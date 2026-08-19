#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
pnm_benchmark_suite_A.py
------------------------
Benchmark reestruturado do PruneN'Merge v3 — Versão A.

Suites:
  A.1 — 12q, 1/2/3/4/6/12 componentes densos (real + cplx)
  A.2 — 2 comp densos, N = 4..16 qubits (real + cplx)
  A.3 — 2 comp médios (densidades 0.3 e 0.7), N = 4..14 (real + cplx)
  A.4 — 2 comp esparsos (nz = N), N = 4..16 (real + cplx)
  A.5 — 3 comp densos, N = 6/9/12/15 qubits (real + cplx)
  A.7 - 3 comp médios (densidades 0.3, 0.5 e 0.7), N = 6/9/12/15 qubits (real + cplx)
  A.6 — 3 comp esparsos (nz = N//3 cada), N = 6/9/12/15 (real + cplx)

Métodos solo:  ucge, ucge_dc, lrsp, baa, svd, pivot, merge*
Métodos pnm+:  pnm+ucge, pnm+ucge_dc, pnm+lrsp, pnm+baa, pnm+svd, pnm+pivot, pnm+merge*
* merge só roda se density(estado) ou density(sub-estado) < MERGE_MAX_DENSITY

Gráficos gerados por suite: CNOTs e Tempo (solo orig/perm vs pnm+ orig/perm).
"""

import os, sys, json, csv, time, gc, warnings, traceback
import multiprocessing as mp
from datetime import datetime
from pathlib import Path
from collections import defaultdict

import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches

sys.path.insert(0, str(Path(__file__).resolve().parent / "pnm"))

warnings.filterwarnings('ignore')

# ===========================================================================
# Configurações
# ===========================================================================

METHOD_TIMEOUT     = 600          # s por método (10 min)
BAA_TIMEOUT        = 1200         # s para baa brute_force (20 min)
N_WORKERS      = max(1, mp.cpu_count() - 1)
FIDELITY_TOL   = 1e-6

# paleta de cores
C = {
    'solo_orig': '#B4B2A9', 'solo_perm': '#5F5E5A',
    'pnm_orig':  '#1D9E75', 'pnm_perm':  '#0F6E56',
}

# ===========================================================================
# Importações do projeto
# ===========================================================================

from qclib.state_preparation import (
    LowRankInitialize, SVDInitialize, UCGEInitialize,
    BaaLowRankInitialize, PivotInitialize, MergeInitialize,
)
from qiskit import QuantumCircuit, transpile
from quantum_state_vector_utils import (
    vector_to_binary_dict,
    normalize_state, generate_random_state_n_m, is_valid_state_vector,
)
from ucge_dc import UCGEDCInitialize
from pnm_main import (          # antes: PNM_main_v3 (renomeado na limpeza)
    build_pnm_with_method,
    apply_bit_permutation_fast,
    generate_kron_state,
    _compute_substates,
)
from pnm_cluster_analysis import analyze_ry_rz_clusters


# ===========================================================================
# Funções de inicialização
# ===========================================================================

def _init_lrsp(sv):
    return LowRankInitialize(sv, opt_params={"unitary_scheme": "ccd"}).definition

def _init_baa(sv):
    return BaaLowRankInitialize(
        sv, opt_params={'max_fidelity_loss': 1e-6, 'strategy': 'brute_force'}
    ).definition

def _init_ucge(sv):    return UCGEInitialize(sv).definition
def _init_ucge_dc(sv): return UCGEDCInitialize(sv).definition

def _init_svd(sv):
    sv2 = sv.copy(); sv2[np.abs(sv2) < 1e-8] = 0
    if np.count_nonzero(sv2) == 1:
        k = int(np.nonzero(sv2)[0][0])
        nq = len(sv2).bit_length() - 1
        qc = QuantumCircuit(nq)
        for i, b in enumerate(bin(k)[2:].zfill(nq)[::-1]):
            if b == '1': qc.x(i)
        return qc
    return SVDInitialize(sv).definition

def _init_pivot(sv):
    sv2 = sv.copy(); sv2[np.abs(sv2) < 1e-8] = 0
    if np.count_nonzero(sv2) == 1:
        k = int(np.nonzero(sv2)[0][0])
        nq = len(sv2).bit_length() - 1
        qc = QuantumCircuit(nq)
        for i, b in enumerate(bin(k)[2:].zfill(nq)[::-1]):
            if b == '1': qc.x(i)
        return qc
    return PivotInitialize(vector_to_binary_dict(sv)).definition

def _init_merge(sv):
    return MergeInitialize(vector_to_binary_dict(sv)).definition

SOLO_METHODS = {
    'ucge':    _init_ucge,
    'ucge_dc': _init_ucge_dc,
    'lrsp':    _init_lrsp,
    'baa':     _init_baa,
    'svd':     _init_svd,
    'pivot':   _init_pivot,
    'merge':   _init_merge,
}

# ordem de exibição nos gráficos
METHODS_ORDER = ['ucge', 'ucge_dc', 'lrsp', 'baa', 'svd', 'pivot', 'merge']
PNM_METHODS   = ['ucge', 'ucge_dc', 'lrsp', 'baa', 'svd', 'pivot', 'merge']

# Reaproveitamento do benchB (opt=0): nas suites A.1-A.6, os metodos baa, merge e
# ucge_dc (+pnm_v3+) ja existem no benchB e serao reusados na fusao; aqui rodamos
# so os faltantes (lrsp, pivot, svd, ucge). Em A.7 (nao coberta pelo benchB),
# rodamos todos os 7. Defina REUSE_FROM_BENCHB=False para rodar tudo do zero.
REUSE_FROM_BENCHB  = True
METHODS_IN_BENCHB  = frozenset({'baa', 'merge', 'ucge_dc'})

def skip_for_suite(suite_name):
    if REUSE_FROM_BENCHB and not suite_name.startswith('A.7'):
        return METHODS_IN_BENCHB
    return frozenset()


# ===========================================================================
# Fidelidade
# ===========================================================================

def fidelity_check(v_alvo, qc):
    from qiskit.quantum_info import Statevector
    try:
        v = np.array(v_alvo, dtype=complex)
        v /= np.linalg.norm(v)
        sv_d = Statevector.from_instruction(qc).data
        sv_r = Statevector.from_instruction(qc).reverse_qargs().data
        best, tag = 0.0, 'FAIL'
        for sv, t in [(sv_d, 'OK'), (sv_r, 'OK-rev')]:
            f = float(abs(np.dot(np.conj(v), np.array(sv, dtype=complex)))**2)
            if f > best: best, tag = f, t
        ok = best >= 1.0 - FIDELITY_TOL
        return best, f"{best:.8f} ({'OK' if ok else 'FAIL'})"
    except Exception as e:
        return 0.0, f"N/A ({e})"


# ===========================================================================
# Workers
# ===========================================================================

def _worker_solo(q, method_name, sv, basis_gates):
    import warnings; warnings.filterwarnings('ignore')
    try:
        t0 = time.time()
        qc = SOLO_METHODS[method_name](sv)
        bt = time.time() - t0
        t1 = time.time()
        qct = transpile(qc, basis_gates=basis_gates, optimization_level=0)
        tt = time.time() - t1
        fid, fstr = fidelity_check(sv, qc)
        q.put({'method': method_name,
               'cnots': qct.count_ops().get('cx', 0), 'depth': qct.depth(),
               'build': round(bt, 6), 'transpile': round(tt, 6),
               'total': round(bt+tt, 6),
               'fidelity': fstr, 'fid_val': fid, 'status': 'ok'})
    except Exception as e:
        q.put({'method': method_name, 'status': 'error', 'error': str(e),
               'cnots': -1, 'depth': -1, 'build': 0, 'transpile': 0,
               'total': 0, 'fidelity': 'ERROR', 'fid_val': 0.0})


def _worker_pnm_plus(q, method_name, sv, basis_gates, cached_result=None):
    import warnings; warnings.filterwarnings('ignore')
    try:
        t0 = time.time()
        qc, _ = build_pnm_with_method(sv, method_name,
                                          cached_result=cached_result)
        bt = time.time() - t0
        if qc is None:
            q.put({'method': f'pnm+{method_name}', 'status': 'not_decomposed'})
            return
        t1 = time.time()
        qct = transpile(qc, basis_gates=basis_gates, optimization_level=0)
        tt = time.time() - t1
        fid, fstr = fidelity_check(sv, qc)
        q.put({'method': f'pnm+{method_name}',
               'cnots': qct.count_ops().get('cx', 0), 'depth': qct.depth(),
               'build': round(bt, 6), 'transpile': round(tt, 6),
               'total': round(bt+tt, 6),
               'fidelity': fstr, 'fid_val': fid, 'status': 'ok'})
    except Exception as e:
        q.put({'method': f'pnm+{method_name}', 'status': 'error',
               'error': str(e), 'cnots': -1, 'depth': -1,
               'build': 0, 'transpile': 0, 'total': 0,
               'fidelity': 'ERROR', 'fid_val': 0.0})


def _run_proc(target, args, label, timeout):
    """Lança processo com timeout garantido."""
    timeout = float(timeout)  # garante tipo numérico
    q = mp.Queue()
    p = mp.Process(target=target, args=(q, *args), daemon=True)
    p.start()
    p.join(timeout)
    if p.is_alive():
        p.terminate(); p.join(5)
        if p.is_alive(): p.kill(); p.join()
        return {'method': label, 'status': 'timeout',
                'cnots': -1, 'depth': -1,
                'build': timeout, 'transpile': 0, 'total': timeout,
                'fidelity': f'TIMEOUT (>{timeout}s)', 'fid_val': 0.0}
    try:
        return q.get_nowait()
    except Exception:
        return {'method': label, 'status': 'error', 'error': 'no result',
                'cnots': -1, 'depth': -1, 'build': 0, 'transpile': 0,
                'total': 0, 'fidelity': 'ERROR', 'fid_val': 0.0}


def _run_batch(tasks):
    """Executa lote de tarefas em paralelo (até N_WORKERS por lote)."""
    results = []
    i = 0
    while i < len(tasks):
        batch = tasks[i:i+N_WORKERS]
        procs, t_launch = [], []
        for target, args, label, tmo in batch:
            tmo = float(tmo)
            q = mp.Queue()
            p = mp.Process(target=target, args=(q, *args), daemon=True)
            p.start()
            procs.append((p, q, label, tmo))
            t_launch.append(time.time())
        for (p, q, label, tmo), t0 in zip(procs, t_launch):
            remaining = max(0, tmo - (time.time() - t0))
            p.join(remaining)
            if p.is_alive():
                p.terminate(); p.join(5)
                if p.is_alive(): p.kill(); p.join()
                results.append({'method': label, 'status': 'timeout',
                                'cnots': -1, 'depth': -1,
                                'build': tmo, 'transpile': 0, 'total': tmo,
                                'fidelity': f'TIMEOUT (>{tmo}s)', 'fid_val': 0.0})
            else:
                try: results.append(q.get_nowait())
                except Exception:
                    results.append({'method': label, 'status': 'error',
                                    'error': 'no result', 'cnots': -1, 'depth': -1,
                                    'build': 0, 'transpile': 0, 'total': 0,
                                    'fidelity': 'ERROR', 'fid_val': 0.0})
        i += N_WORKERS
    return results


# ===========================================================================
# Motor principal
# ===========================================================================

def run_all_methods(sv, basis_gates=['u', 'cx'], skip=frozenset()):
    results = []

    # ── 1. Análise de separabilidade + cache sub-estados ──────────────────
    import io as _io, sys as _sys
    _old = _sys.stdout; _sys.stdout = _io.StringIO()
    try:
        cached = analyze_ry_rz_clusters(sv)
        decomposed = cached['can_decompose']
        if decomposed:
            cached = _compute_substates(sv, cached)
    except Exception as e:
        print(f"[ERRO analyze]: {e}", file=sys.stderr)
        cached, decomposed = None, False
    finally:
        _sys.stdout = _old

    # ── 2. Calcula nz para controle de pivot ──────────────────────────────
    nz = int(np.count_nonzero(np.abs(sv) > 1e-9))

    # ── 3. Solo rápidos + pnm+ rápidos — em paralelo ──────────────────────
    tasks = []
    for m in METHODS_ORDER:
        if m in ('baa', 'merge'): continue
        if m in skip: continue
        if m == 'pivot' and nz > 5000: continue   # pivot catastrófico em denso
        tasks.append((_worker_solo, (m, sv, basis_gates), m, METHOD_TIMEOUT))

    if decomposed:
        for m in PNM_METHODS:
            if m in ('baa', 'merge'): continue
            if m in skip: continue
            tasks.append((_worker_pnm_plus, (m, sv, basis_gates, cached),
                          f'pnm+{m}', METHOD_TIMEOUT))

    results.extend(_run_batch(tasks))

    # ── 4. BAA + merge (lentos) — sequencialmente no final ────────────────
    slow_tasks = []
    if 'baa' not in skip:
        slow_tasks.append((_worker_solo, ('baa', sv, basis_gates), 'baa', BAA_TIMEOUT))
        if decomposed:
            slow_tasks.append((_worker_pnm_plus,
                               ('baa', sv, basis_gates, cached), 'pnm+baa', BAA_TIMEOUT))
    if 'merge' not in skip:
        slow_tasks.append((_worker_solo, ('merge', sv, basis_gates), 'merge', METHOD_TIMEOUT))
        if decomposed:
            slow_tasks.append((_worker_pnm_plus,
                               ('merge', sv, basis_gates, cached), 'pnm+merge', METHOD_TIMEOUT))

    for target, args, label, tmo in slow_tasks:
        results.append(_run_proc(target, args, label, tmo))

    gc.collect()
    return results, decomposed


def run_solo_only(sv, basis_gates=['u', 'cx'], skip=frozenset()):
    """Apenas métodos solo — para casos onde PNM não decompõe."""
    nz = int(np.count_nonzero(np.abs(sv) > 1e-9))
    tasks = []
    for m in METHODS_ORDER:
        if m in ('baa', 'merge'): continue
        if m in skip: continue
        if m == 'pivot' and nz > 5000: continue
        tasks.append((_worker_solo, (m, sv, basis_gates), m, METHOD_TIMEOUT))

    results = _run_batch(tasks)
    if 'baa' not in skip:
        results.append(_run_proc(_worker_solo, ('baa', sv, basis_gates),
                                 'baa', BAA_TIMEOUT))
    if 'merge' not in skip:
        results.append(_run_proc(_worker_solo, ('merge', sv, basis_gates),
                                 'merge', METHOD_TIMEOUT))
    return results


# ===========================================================================
# Checkpoint
# ===========================================================================

def load_ckpt(path):
    return json.load(open(path)) if Path(path).exists() else {'completed': []}

def save_ckpt(path, ck):
    json.dump(ck, open(path, 'w'), indent=2)

def is_done(ck, key): return key in ck['completed']

def mark_done(ck, path, key):
    ck['completed'].append(key); save_ckpt(path, ck)


# ===========================================================================
# CSV
# ===========================================================================

FIELDNAMES = ['Suite', 'Caso', 'Qubits', 'nz_total', 'Run', 'Real',
              'Permutado', 'Method', 'CNOTs', 'Depth',
              'Build (s)', 'Transpile (s)', 'Total (s)', 'Decomposed', 'Fidelity']


def write_results(writer, f_csv, f_fail, suite, caso, nq, nz,
                  run, real, perm_str, results, decomposed):
    base = {'Suite': suite, 'Caso': caso, 'Qubits': nq, 'nz_total': nz,
            'Run': run, 'Real': real, 'Permutado': perm_str}
    for r in results:
        if r.get('status') == 'not_decomposed': continue
        row = {**base,
               'Method':       r['method'],
               'CNOTs':        r['cnots'],
               'Depth':        r['depth'],
               'Build (s)':    r['build'],
               'Transpile (s)':r['transpile'],
               'Total (s)':    r['total'],
               'Decomposed':   decomposed,
               'Fidelity':     r['fidelity']}
        writer.writerow(row)
        if 'FAIL' in r.get('fidelity', ''):
            f_fail.write(f"Suite={suite} Caso={caso} Run={run} Perm={perm_str} "
                         f"Method={r['method']} Fid={r['fidelity']}\n")
            f_fail.flush()
    f_csv.flush()


# ===========================================================================
# Gráficos
# ===========================================================================

def _avg(lst): return round(sum(lst)/len(lst), 2) if lst else 0.0


def make_chart(title, cases, data, ylabel, outpath, logy=False):
    """Gráfico de barras agrupadas: solo orig/perm vs pnm+ orig/perm.
    Se logy=True, usa escala log10 no eixo Y (para grandes diferenças de ordem de grandeza)."""
    fig, ax = plt.subplots(figsize=(max(8, len(cases)*0.9), 4.5))
    n = len(cases); x = np.arange(n); w = 0.2
    offsets = [-1.5*w, -0.5*w, 0.5*w, 1.5*w]
    keys    = ['solo_orig', 'solo_perm', 'pnm_orig', 'pnm_perm']
    labels  = ['solo orig', 'solo perm', 'pnm+ orig', 'pnm+ perm']

    for off, key, lbl in zip(offsets, keys, labels):
        vals = data.get(key, [0]*n)
        # para log, substitui zeros por 0.1 para não sumir do gráfico
        plot_vals = [max(v, 0.1) if logy else v for v in vals]
        ax.bar(x + off, plot_vals, width=w, color=C[key], label=lbl)

    ax.set_xticks(x)
    ax.set_xticklabels(cases, rotation=45, ha='right', fontsize=8)
    ax.set_ylabel(ylabel, fontsize=9)
    ax.set_title(title, fontsize=10, fontweight='bold')

    if logy:
        ax.set_yscale('log')
        ax.set_ylabel(f"{ylabel} [log₁₀]", fontsize=9)
        # não seta ylim(bottom=0) pois conflita com escala log
    else:
        ax.set_ylim(bottom=0)

    patches = [mpatches.Patch(color=C[k], label=l) for k, l in zip(keys, labels)]
    ax.legend(handles=patches, fontsize=8, ncol=2, loc='upper left', framealpha=0.7)
    plt.tight_layout()
    plt.savefig(outpath, dpi=150, bbox_inches='tight')
    plt.close()


def generate_plots(suite_name, rows, outdir):
    """Gera gráficos CNOTs e Tempo para cada método da suite."""
    os.makedirs(outdir, exist_ok=True)

    def _m(r):    return r.get('Method') or r.get('method', '')
    def _cx(r):
        v = r.get('CNOTs') if r.get('CNOTs') is not None else r.get('cnots', -1)
        try: return int(v)
        except: return -1
    def _t(r):
        v = r.get('Total (s)') if r.get('Total (s)') is not None else r.get('total', 0)
        try: return float(v)
        except: return 0.0
    def _caso(r):  return r.get('Caso', '')
    def _perm(r):  return str(r.get('Permutado', ''))

    rows = [r for r in rows if _m(r) and _cx(r) >= 0 and _caso(r)]
    casos = sorted(set(_caso(r) for r in rows))
    case_labels = [c.split('_', 1)[-1] if '_' in c else c for c in casos]

    for m in METHODS_ORDER:
        cx_data = {k: [] for k in ['solo_orig','solo_perm','pnm_orig','pnm_perm']}
        t_data  = {k: [] for k in ['solo_orig','solo_perm','pnm_orig','pnm_perm']}

        has_data = False
        for caso in casos:
            def get(method, perm):
                return [_cx(r) for r in rows
                        if _m(r)==method and _caso(r)==caso and _perm(r)==perm
                        and _cx(r) >= 0]
            def gett(method, perm):
                return [_t(r) for r in rows
                        if _m(r)==method and _caso(r)==caso and _perm(r)==perm]

            cx_data['solo_orig'].append(_avg(get(m, 'False')))
            cx_data['solo_perm'].append(_avg(get(m, 'True')))
            cx_data['pnm_orig'].append(_avg(get(f'pnm+{m}', 'False')))
            cx_data['pnm_perm'].append(_avg(get(f'pnm+{m}', 'True')))

            t_data['solo_orig'].append(_avg(gett(m, 'False')))
            t_data['solo_perm'].append(_avg(gett(m, 'True')))
            t_data['pnm_orig'].append(_avg(gett(f'pnm+{m}', 'False')))
            t_data['pnm_perm'].append(_avg(gett(f'pnm+{m}', 'True')))

            if any(cx_data[k][-1] > 0 for k in cx_data): has_data = True

        if not has_data: continue

        # escala log quando há grande diferença de ordem de grandeza (razão > 50)
        all_cx = [v for vals in cx_data.values() for v in vals if v > 0]
        use_log_cx = len(all_cx) > 0 and max(all_cx) / (min(all_cx) + 1e-9) > 50

        all_t = [v for vals in t_data.values() for v in vals if v > 0]
        use_log_t = len(all_t) > 0 and max(all_t) / (min(all_t) + 1e-9) > 50

        make_chart(f"{suite_name} — {m} — CNOTs",
                   case_labels, cx_data, 'CNOTs (avg)',
                   os.path.join(outdir, f"{suite_name}_{m}_cnots.png"),
                   logy=use_log_cx)
        make_chart(f"{suite_name} — {m} — Tempo total (s)",
                   case_labels, t_data, 'Tempo (s, avg)',
                   os.path.join(outdir, f"{suite_name}_{m}_time.png"),
                   logy=use_log_t)

    print(f"  [plots] {outdir}")


# ===========================================================================
# Permutações
# ===========================================================================

PERM_MAP = {
    2:  [1,0],
    4:  [1,3,0,2],
    6:  [1,3,5,2,0,4],
    8:  [1,3,6,5,2,7,0,4],
    9:  [1,3,6,5,8,2,7,0,4],
    10: [1,3,6,9,5,2,7,0,4,8],
    12: [1,11,10,3,0,8,4,5,6,2,9,7],
    14: [1,13,10,3,0,8,4,5,6,2,9,7,11,12],
    15: [1,13,10,3,0,8,4,5,6,2,9,7,11,12,14],
    16: [1,15,10,3,0,8,4,5,6,2,9,7,11,12,13,14],
}

def get_perm(n):
    if n in PERM_MAP: return PERM_MAP[n]
    evens = list(range(0,n,2)); odds = list(range(1,n,2))
    return odds + evens


# ===========================================================================
# Executor de suite genérico
# ===========================================================================

def run_suite(suite_name, cases, writer, f_csv, f_fail,
              checkpoint, ckpt_path, plot_dir, n_runs=10):
    suite_rows = []

    for case in cases:
        label    = case['label']
        list_nq  = case['list_nq']
        list_nz  = case['list_nz']
        real     = case['real']
        solo_only = case.get('solo_only', False)
        runs     = case.get('n_runs', n_runs)
        nq       = sum(list_nq)
        perm     = get_perm(nq)
        tag      = 'real' if real else 'cplx'

        print(f"\n  [{suite_name}] {label} ({nq}q, {tag}"
              f"{', SOLO ONLY' if solo_only else ''})")

        for run in range(1, runs+1):
            for permutado in [False, True]:
                ck_key = f"{suite_name}|{label}|{run}|{permutado}"
                if is_done(checkpoint, ck_key):
                    print(f"    skip {ck_key}")
                    continue

                seed = hash((suite_name, label, run)) % (2**31)
                state_orig = generate_kron_state(list_nq, list_nz,
                                                 seed=seed, real=real)
                sv = normalize_state(
                    np.array(apply_bit_permutation_fast(state_orig, perm))
                    if permutado else state_orig.copy())

                perm_str = 'True' if permutado else 'False'

                if not is_valid_state_vector(sv):
                    print(f"    run={run} perm={perm_str}  "
                          f"[PULADO: vetor de estado inválido]")
                    f_fail.write(f"Suite={suite_name} Caso={label} Run={run} "
                                 f"Perm={perm_str} Status=invalid_state_vector "
                                 f"(tamanho={len(sv)})\n")
                    f_fail.flush()
                    mark_done(checkpoint, ckpt_path, ck_key)
                    continue

                nz_total = int(np.count_nonzero(np.abs(sv) > 1e-9))

                print(f"    run={run} perm={perm_str} nz={nz_total}",
                      end='  ', flush=True)
                t0 = time.time()

                skip = skip_for_suite(suite_name)
                if solo_only:
                    results = run_solo_only(sv, skip=skip)
                    decomposed = False
                else:
                    results, decomposed = run_all_methods(sv, skip=skip)

                print(f"t={time.time()-t0:.1f}s  decomposed={decomposed}")

                write_results(writer, f_csv, f_fail,
                              suite_name, label, nq, nz_total,
                              run, real, perm_str, results, decomposed)

                for r in results:
                    suite_rows.append({**r,
                        'Suite': suite_name, 'Caso': label,
                        'Qubits': nq, 'Run': run, 'Real': real,
                        'Permutado': perm_str})

                mark_done(checkpoint, ckpt_path, ck_key)

    if suite_rows:
        generate_plots(suite_name, suite_rows, plot_dir)

    return suite_rows


# ===========================================================================
# Definição das suites A.1 – A.6
# ===========================================================================

def make_A1(real):
    """A.1 — 12 qubits, 1/2/3/4/6/12 componentes densos."""
    tag = 'real' if real else 'cplx'
    configs = [
        # (n_comp, list_nq,       list_nz_frac,  solo_only, n_runs)
        (1,  [12],            [1.0],          True,  2),   # baseline: 2 amostras
        (2,  [6,6],           [1.0,1.0],      False, 5),
        (3,  [4,4,4],         [1.0]*3,        False, 5),
        (4,  [3,3,3,3],       [1.0]*4,        False, 5),
        (6,  [2,2,2,2,2,2],   [1.0]*6,        False, 5),
        (12, [1]*12,          [1.0]*12,       False, 5),
    ]
    cases = []
    for n_comp, list_nq, fracs, solo_only, n_runs in configs:
        list_nz = [max(1, int(np.ceil((1<<nq)*f))) for nq,f in zip(list_nq, fracs)]
        cases.append({'label': f'{tag}_12q_{n_comp:02d}comp',
                      'list_nq': list_nq, 'list_nz': list_nz,
                      'real': real, 'solo_only': solo_only, 'n_runs': n_runs})
    return cases


def make_A2(real):
    """A.2 — 2 comp densos, N = 4..16."""
    tag = 'real' if real else 'cplx'
    cases = []
    for N in [4, 6, 8, 10, 12, 14, 16]:
        nq = N // 2
        cases.append({'label': f'{tag}_2x{nq}q_denso',
                      'list_nq': [nq,nq],
                      'list_nz': [2**nq, 2**nq],
                      'real': real})
    return cases


def make_A3(real):
    """A.3 — 2 comp médios (density 0.3 e 0.7), N = 4..14."""
    tag = 'real' if real else 'cplx'
    cases = []
    for N in [4, 6, 8, 10, 12, 14]:
        nq = N // 2
        nz = [max(1, int(np.ceil(d * 2**nq))) for d in [0.3, 0.7]]
        cases.append({'label': f'{tag}_2x{nq}q_medio',
                      'list_nq': [nq,nq], 'list_nz': nz, 'real': real})
    return cases


def make_A4(real):
    """A.4 — 2 comp esparsos (nz = N por componente), N = 4..16."""
    tag = 'real' if real else 'cplx'
    cases = []
    for N in [4, 6, 8, 10, 12, 14, 16]:
        nq = N // 2
        nz_each = max(1, N)   # nz = N total → N/2 por componente aprox.
        nz_each = max(1, nq)  # nz por componente = nq (muito esparso)
        cases.append({'label': f'{tag}_2x{nq}q_esparso',
                      'list_nq': [nq,nq],
                      'list_nz': [nz_each, nz_each],
                      'real': real})
    return cases


def make_A5(real):
    """A.5 — 3 comp densos, N = 6/9/12/15."""
    tag = 'real' if real else 'cplx'
    cases = []
    for N in [6, 9, 12, 15]:
        nq = N // 3
        cases.append({'label': f'{tag}_3x{nq}q_denso',
                      'list_nq': [nq]*3,
                      'list_nz': [2**nq]*3,
                      'real': real})
    return cases

def make_A7(real):
    """A.7 — 3 comp médios (density 0.3, 0.5 e 0.7), N = 6/9/12/15."""
    tag = 'real' if real else 'cplx'
    cases = []
    for N in [6, 9, 12, 15]:
        nq = N // 3
        nz = [max(1, int(np.ceil(d * 2**nq))) for d in [0.3, 0.5, 0.7]]
        cases.append({'label': f'{tag}_3x{nq}q_medio030507',
                      'list_nq': [nq,nq], 'list_nz': nz, 'real': real})
    return cases


def make_A6(real):
    """A.6 — 3 comp esparsos (nz = nq por componente), N = 6/9/12/15."""
    tag = 'real' if real else 'cplx'
    cases = []
    for N in [6, 9, 12, 15]:
        nq = N // 3
        nz_each = max(1, nq)
        cases.append({'label': f'{tag}_3x{nq}q_esparso',
                      'list_nq': [nq]*3,
                      'list_nz': [nz_each]*3,
                      'real': real})
    return cases


# ===========================================================================
# Main
# ===========================================================================

if __name__ == '__main__':
    import psutil

    timestamp = datetime.now().strftime("%d%b%y_%H%M")
    csv_file  = f"benchA_completo_{timestamp}.csv"
    fail_file = f"benchA_falhas_{timestamp}.txt"
    ckpt_file = f"benchA_checkpoint_{timestamp}.json"
    plot_base = f"benchA_plots_{timestamp}"

    if len(sys.argv) > 1 and Path(sys.argv[1]).exists():
        ckpt_file = sys.argv[1]
        ts = Path(ckpt_file).stem.replace('benchA_checkpoint_', '')
        csv_file  = f"benchA_completo_{ts}.csv"
        fail_file = f"benchA_falhas_{ts}.txt"
        plot_base = f"benchA_plots_{ts}"
        print(f"Retomando: {ckpt_file}")
    else:
        print(f"Novo benchmark: {csv_file}")

    checkpoint = load_ckpt(ckpt_file)
    csv_mode   = 'a' if Path(csv_file).exists() else 'w'
    f_csv      = open(csv_file,  csv_mode, newline='', encoding='utf-8')
    f_fail     = open(fail_file, 'a',      encoding='utf-8')
    writer     = csv.DictWriter(f_csv, fieldnames=FIELDNAMES)
    if csv_mode == 'w': writer.writeheader()
    f_fail.write(f"\n# Sessão {timestamp}\n")

    N_RUNS = 10

    SUITES = [
        # (nome,        make_fn(real),   make_fn(cplx))
        ('A.1', make_A1(True),  make_A1(False)),
        ('A.2', make_A2(True),  make_A2(False)),
        ('A.3', make_A3(True),  make_A3(False)),
        ('A.4', make_A4(True),  make_A4(False)),
        ('A.5', make_A5(True),  make_A5(False)),
        ('A.7', make_A7(True),  make_A7(False)),
        ('A.6', make_A6(True),  make_A6(False)),
    ]

    for suite_base, cases_real, cases_cplx in SUITES:
        for suffix, cases in [('_real', cases_real), ('_cplx', cases_cplx)]:
            suite_name = suite_base + suffix
            n_casos = len(cases)
            print(f"\n{'='*62}")
            print(f"  SUITE: {suite_name}  ({n_casos} casos × {N_RUNS} runs × 2 perm)")
            print(f"{'='*62}")

            plot_dir = os.path.join(plot_base, suite_name)
            run_suite(suite_name, cases, writer, f_csv, f_fail,
                      checkpoint, ckpt_file, plot_dir, n_runs=N_RUNS)

            mem = psutil.virtual_memory()
            print(f"  RAM: {mem.percent:.1f}%  |  Livre: {mem.available/1e9:.1f} GB")

    f_csv.close()
    f_fail.close()
    print(f"\n✓ CSV:        {csv_file}")
    print(f"✓ Falhas:     {fail_file}")
    print(f"✓ Checkpoint: {ckpt_file}")
    print(f"✓ Gráficos:   {plot_base}/")
