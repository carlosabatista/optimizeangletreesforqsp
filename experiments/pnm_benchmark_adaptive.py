#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
pnm_benchmark_adaptive.py
-----------------------------
Benchmark completo do PNM + seletor adaptativo (APNM).

Versão 4 — consolidação final:
  - optimization_level=0 em todos os métodos
  - Todos os métodos: ucge_dc, ucge, baa, merge, svd, lrsp, pivot (solos)
    + pnm+ de cada um + pnm+adaptive
  - svd: skip quando nq==1
  - pivot: skip quando nq>=5
  - Teto de qubits: 14q para suites de 2 componentes, 15q para suites de 3 componentes.
  - Suites A.1/A.2/A.5 reformuladas (mistas denso+esparso)
  - rho_log A.1: [0.05, 0.09, 0.15, 0.25, 0.45] 
  - Cache de separabilidade compartilhado entre todos os pnm+ do mesmo caso
  - Checkpoint + --resume
  - Workers configurável (--workers N, padrão 4)
  - spawn em vez de fork (evita herança de memória)
  - del + gc.collect() por caso
  - Tempo total na tela por caso e por run

Uso:
    python pnm_benchmark_adaptive.py
    python pnm_benchmark_adaptive.py --workers 4
    python pnm_benchmark_adaptive.py --resume benchB_vX_XXXX.csv --workers 4
"""

import os, csv, time, gc, warnings, argparse
import sys
from pathlib import Path
import multiprocessing as mp
from datetime import datetime

import numpy as np

warnings.filterwarnings('ignore')

# torna pnm/ e adaptive/ (irmãs de benchmarks/, todas sob a raiz do
# repositório) importáveis — este arquivo mora em benchmarks/, então sobe
# um nível até a raiz e desce em cada uma.
_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_ROOT / "pnm"))
sys.path.insert(0, str(_ROOT / "adaptive"))

from qclib.state_preparation import (
    LowRankInitialize, SVDInitialize, UCGEInitialize,
    BaaLowRankInitialize, PivotInitialize, MergeInitialize,
)
from qiskit import QuantumCircuit, transpile
from qiskit.quantum_info import Statevector

from quantum_state_vector_utils import (
    vector_to_binary_dict, normalize_state, generate_random_state_n_m,
)
from ucge_dc import UCGEDCInitialize
from pnm_main import (
    build_pnm_with_method, apply_bit_permutation_fast,
    generate_kron_state, _compute_substates, prepare_basis_state,
)
from pnm_cluster_analysis import analyze_ry_rz_clusters
from pnm_adaptive_selector import build_pnm_adaptive

# ===========================================================================
# Configuração
# ===========================================================================

METHOD_TIMEOUT = 600
BAA_TIMEOUT    = 600
FIDELITY_TOL   = 1e-6
BASIS_GATES    = ['u', 'cx']
N_RUNS         = 10

# ===========================================================================
# Inicializadores solo
# ===========================================================================

def _init_ucge_dc(sv): return UCGEDCInitialize(sv).definition
def _init_ucge(sv):    return UCGEInitialize(sv).definition

def _init_baa(sv):
    return BaaLowRankInitialize(
        sv, opt_params={'max_fidelity_loss': 1e-6, 'strategy': 'brute_force'}
    ).definition

def _init_lrsp(sv):
    return LowRankInitialize(sv, opt_params={"unitary_scheme": "ccd"}).definition

def _init_svd(sv):
    sv2 = sv.copy(); sv2[np.abs(sv2) < 1e-8] = 0
    if np.count_nonzero(sv2) == 1:
        k = int(np.nonzero(sv2)[0][0])
        return prepare_basis_state(k, len(sv2).bit_length() - 1)
    return SVDInitialize(sv).definition

def _init_pivot(sv):
    sv2 = sv.copy(); sv2[np.abs(sv2) < 1e-8] = 0
    if np.count_nonzero(sv2) == 1:
        k = int(np.nonzero(sv2)[0][0])
        return prepare_basis_state(k, len(sv2).bit_length() - 1)
    return PivotInitialize(vector_to_binary_dict(sv)).definition

def _init_merge(sv):
    sv2 = sv.copy(); sv2[np.abs(sv2) < 1e-8] = 0
    if np.count_nonzero(sv2) == 1:
        k = int(np.nonzero(sv2)[0][0])
        return prepare_basis_state(k, len(sv2).bit_length() - 1)
    return MergeInitialize(vector_to_binary_dict(sv)).definition

# método -> (fn, timeout, skip_fn)
# skip_fn(nq_total, nz_total) -> True = pular este método para este caso
#
# svd: qclib.SVDInitialize não aceita nq=1 (falha com "not a positive
#      power of 2", mesmo 2 sendo potência de 2 — limitação da bipartição
#      interna do método, confirmado empiricamente). Nunca dispara nas
#      suítes atuais (nq_total mínimo = 4), mas é seguro manter — e no
#      caminho pnm+, sub-estados de 1 qubit já caem em UCGE por conta
#      própria (ver build_pnm_with_method), então essa guarda nem
#      precisaria valer ali.
# pivot: fica caro para estados com muitos elementos não-nulos (denso).
#      LIMIAR AINDA NÃO CALIBRADO — 5000 é um valor herdado de uma
#      versão anterior do pipeline (comentário "if nz > 5000: continue"),
#      não validado neste benchmark. Ajustar quando houver dados de
#      calibração próprios.
_PIVOT_NZ_SKIP_THRESHOLD = 5000

SOLO_METHODS = {
    'ucge_dc': (_init_ucge_dc, METHOD_TIMEOUT, lambda nq, nz: False),
    'ucge':    (_init_ucge,    METHOD_TIMEOUT, lambda nq, nz: False),
    'baa':     (_init_baa,     BAA_TIMEOUT,    lambda nq, nz: False),
    'merge':   (_init_merge,   METHOD_TIMEOUT, lambda nq, nz: False),
    'svd':     (_init_svd,     METHOD_TIMEOUT, lambda nq, nz: nq == 1),
    'lrsp':    (_init_lrsp,    METHOD_TIMEOUT, lambda nq, nz: False),
    'pivot':   (_init_pivot,   METHOD_TIMEOUT, lambda nq, nz: nz >= _PIVOT_NZ_SKIP_THRESHOLD),
}
PNM_METHODS = list(SOLO_METHODS.keys())   # mesmos 7 + adaptive

# ===========================================================================
# Fidelidade
# ===========================================================================

def fidelity_check(v_alvo, qc):
    try:
        v    = np.array(v_alvo, dtype=complex)
        v   /= np.linalg.norm(v)
        sv_d = Statevector.from_instruction(qc).data
        sv_r = Statevector.from_instruction(qc).reverse_qargs().data
        best = max(
            float(abs(np.dot(np.conj(v), np.array(s, dtype=complex)))**2)
            for s in [sv_d, sv_r]
        )
        ok = best >= 1.0 - FIDELITY_TOL
        return best, f"{best:.8f} ({'OK' if ok else 'FAIL'})"
    except Exception as e:
        return 0.0, f"N/A ({e})"

# ===========================================================================
# Workers
# ===========================================================================

def _worker_solo(q, method_name, init_fn, sv, basis_gates):
    import warnings; warnings.filterwarnings('ignore')
    try:
        t0  = time.time()
        qc  = init_fn(sv)
        bt  = time.time() - t0
        t1  = time.time()
        qct = transpile(qc, basis_gates=basis_gates, optimization_level=0)
        tt  = time.time() - t1
        fid, fstr = fidelity_check(sv, qc)
        q.put({'method': method_name,
               'cnots': qct.count_ops().get('cx', 0), 'depth': qct.depth(),
               'build': round(bt, 6), 'transpile': round(tt, 6),
               'total': round(bt + tt, 6), 'fidelity': fstr,
               'status': 'ok', 'submethods': '-'})
    except Exception as e:
        q.put({'method': method_name, 'status': 'error',
               'cnots': -1, 'depth': -1, 'build': 0, 'transpile': 0,
               'total': 0, 'fidelity': f'ERROR ({e})', 'submethods': '-'})


def _worker_pnm(q, method_name, sv, basis_gates, cached_result=None):
    import warnings; warnings.filterwarnings('ignore')
    try:
        t0    = time.time()
        qc, _ = build_pnm_with_method(sv, method_name,
                                          cached_result=cached_result)
        bt    = time.time() - t0
        if qc is None:
            q.put({'method': f'pnm+{method_name}', 'status': 'not_decomposed',
                   'cnots': -1, 'depth': -1, 'build': 0, 'transpile': 0,
                   'total': 0, 'fidelity': 'N/A', 'submethods': '-'})
            return
        t1  = time.time()
        qct = transpile(qc, basis_gates=basis_gates, optimization_level=0)
        tt  = time.time() - t1
        fid, fstr = fidelity_check(sv, qc)
        q.put({'method': f'pnm+{method_name}',
               'cnots': qct.count_ops().get('cx', 0), 'depth': qct.depth(),
               'build': round(bt, 6), 'transpile': round(tt, 6),
               'total': round(bt + tt, 6), 'fidelity': fstr,
               'status': 'ok', 'submethods': method_name})
    except Exception as e:
        q.put({'method': f'pnm+{method_name}', 'status': 'error',
               'cnots': -1, 'depth': -1, 'build': 0, 'transpile': 0,
               'total': 0, 'fidelity': f'ERROR ({e})', 'submethods': '-'})


def _worker_adaptive(q, sv, basis_gates, cached_result=None):
    import warnings; warnings.filterwarnings('ignore')
    from pnm_adaptive_selector import build_pnm_adaptive
    try:
        t0 = time.time()
        qc, _, methods_used = build_pnm_adaptive(sv, cached_result=cached_result)
        bt = time.time() - t0
        if qc is None:
            q.put({'method': 'pnm+adaptive', 'status': 'not_decomposed',
                   'cnots': -1, 'depth': -1, 'build': 0, 'transpile': 0,
                   'total': 0, 'fidelity': 'N/A', 'submethods': '-'})
            return
        t1  = time.time()
        qct = transpile(qc, basis_gates=basis_gates, optimization_level=0)
        tt  = time.time() - t1
        fid, fstr = fidelity_check(sv, qc)
        submethods = '+'.join(sorted(set(methods_used.values())))
        q.put({'method': 'pnm+adaptive',
               'cnots': qct.count_ops().get('cx', 0), 'depth': qct.depth(),
               'build': round(bt, 6), 'transpile': round(tt, 6),
               'total': round(bt + tt, 6), 'fidelity': fstr,
               'status': 'ok', 'submethods': submethods})
    except Exception as e:
        q.put({'method': 'pnm+adaptive', 'status': 'error',
               'cnots': -1, 'depth': -1, 'build': 0, 'transpile': 0,
               'total': 0, 'fidelity': f'ERROR ({e})', 'submethods': '-'})


def _run_proc(target, args, label, timeout):
    q = mp.Queue()
    p = mp.Process(target=target, args=(q, *args), daemon=True)
    p.start()
    p.join(float(timeout))
    if p.is_alive():
        p.terminate(); p.join(5)
        if p.is_alive(): p.kill(); p.join()
        return {'method': label, 'status': 'timeout',
                'cnots': -1, 'depth': -1,
                'build': timeout, 'transpile': 0, 'total': timeout,
                'fidelity': f'TIMEOUT (>{timeout}s)', 'submethods': '-'}
    try:
        return q.get_nowait()
    except Exception:
        return {'method': label, 'status': 'error',
                'cnots': -1, 'depth': -1, 'build': 0, 'transpile': 0,
                'total': 0, 'fidelity': 'ERROR', 'submethods': '-'}

# ===========================================================================
# Execução paralela por janela de n_workers
# ===========================================================================

def _run_batch_parallel(tasks, n_workers):
    """
    tasks: lista de (target, args, label, timeout)
    Retorna dict {label: result}
    """
    results = {}
    i = 0
    while i < len(tasks):
        batch = tasks[i:i + n_workers]
        procs = []
        for target, args, label, timeout in batch:
            q = mp.Queue()
            p = mp.Process(target=target, args=(q, *args), daemon=True)
            p.start()
            procs.append((p, q, label, timeout))

        for p, q, label, timeout in procs:
            p.join(float(timeout))
            if p.is_alive():
                p.terminate(); p.join(5)
                if p.is_alive(): p.kill(); p.join()
                results[label] = {
                    'method': label, 'status': 'timeout',
                    'cnots': -1, 'depth': -1,
                    'build': timeout, 'transpile': 0, 'total': timeout,
                    'fidelity': f'TIMEOUT (>{timeout}s)', 'submethods': '-'}
            else:
                try:
                    results[label] = q.get_nowait()
                except Exception:
                    results[label] = {
                        'method': label, 'status': 'error',
                        'cnots': -1, 'depth': -1, 'build': 0,
                        'transpile': 0, 'total': 0,
                        'fidelity': 'ERROR', 'submethods': '-'}
        i += n_workers
    return results

# ===========================================================================
# Suites
# ===========================================================================

_perm_map = {
    4:  [1, 3, 0, 2],
    6:  [1, 3, 5, 2, 0, 4],
    8:  [1, 3, 6, 5, 2, 7, 0, 4],
    9:  [1, 3, 6, 5, 8, 2, 7, 0, 4],
    10: [1, 3, 6, 9, 5, 2, 7, 0, 4, 8],
    12: [1, 11, 10, 3, 0, 8, 4, 5, 6, 2, 9, 7],
    14: [1, 13, 10, 3, 0, 8, 4, 5, 6, 2, 9, 7, 11, 12],
    15: [1, 13, 10, 3, 14, 8, 4, 5, 6, 2, 9, 7, 11, 12, 0],
}


def _build_suites():
    suites = []

    # -------------------------------------------------------------------
    # A.1 — denso + esparso variável, 2 comp, até 14q (7q+7q)
    # rho_log: 5 pontos cobrindo zona esparsa e transição
    # -------------------------------------------------------------------
    _rho_log = [0.05, 0.09, 0.15, 0.25, 0.45]
    _a1_dims = [(2,2),(2,3),(3,3),(3,4),(4,4),(4,5),(5,5),(6,6),(7,7)]
    for real, prefix in [(True,'real'),(False,'cplx')]:
        for (nq_d, nq_s) in _a1_dims:
            for rho_s in _rho_log:
                nz_d = 2**nq_d
                nz_s = max(1, round(rho_s * (2**nq_s)))
                suites.append({
                    'suite':   f'A.1_{prefix}',
                    'caso':    f'{prefix}_{nq_d}qD_{nq_s}qS_rho{rho_s:.2f}',
                    'nq_list': [nq_d, nq_s],
                    'nz_list': [nz_d, nz_s],
                    'real':    real,
                })

    # -------------------------------------------------------------------
    # A.2 — zona de transição, 2 comp, até 14q (7q+7q)
    # rho em {0.10, 0.15, 0.20, 0.25} — produto cartesiano
    # -------------------------------------------------------------------
    _rho_trans = [0.10, 0.15, 0.20, 0.25]
    _a2_dims   = [(2,2),(3,3),(4,4),(5,5),(6,6),(7,7)]
    for real, prefix in [(True,'real'),(False,'cplx')]:
        for (nq1, nq2) in _a2_dims:
            for rho1 in _rho_trans:
                for rho2 in _rho_trans:
                    suites.append({
                        'suite':   f'A.2_{prefix}',
                        'caso':    f'{prefix}_{nq1}q_rho{rho1:.2f}_{nq2}q_rho{rho2:.2f}',
                        'nq_list': [nq1, nq2],
                        'nz_list': [max(1, round(rho1*(2**nq1))),
                                    max(1, round(rho2*(2**nq2)))],
                        'real':    real,
                    })

    # -------------------------------------------------------------------
    # A.3 — 2 comp médios (rho=0.3, 0.7), até 14q (7q+7q)
    # -------------------------------------------------------------------
    for total_nq in [4, 6, 8, 10, 12, 14]:
        nq = total_nq // 2
        for real, prefix in [(True,'real'),(False,'cplx')]:
            suites.append({
                'suite':   f'A.3_{prefix}',
                'caso':    f'{prefix}_2x{nq}q_medio',
                'nq_list': [nq, nq],
                'nz_list': [max(1, int(np.ceil(0.3*2**nq))),
                            max(1, int(np.ceil(0.7*2**nq)))],
                'real':    real,
            })

    # -------------------------------------------------------------------
    # A.4 — 2 comp esparsos (nz=nq), até 14q (7q+7q)
    # -------------------------------------------------------------------
    for total_nq in [4, 6, 8, 10, 12, 14]:
        nq = total_nq // 2
        for real, prefix in [(True,'real'),(False,'cplx')]:
            suites.append({
                'suite':   f'A.4_{prefix}',
                'caso':    f'{prefix}_2x{nq}q_esparso',
                'nq_list': [nq, nq],
                'nz_list': [nq, nq],
                'real':    real,
            })

    # -------------------------------------------------------------------
    # A.5 — gradiente ordenado, 3 comp, até 15q (5q+5q+5q)
    # Triplas fixas de rho: (esparso, médio, denso)
    # Variantes: all-real e mixed (comp esparso complexo)
    # -------------------------------------------------------------------
    _a5_triplas = [
        (0.05, 0.20, 1.0),
        (0.05, 0.50, 1.0),
        (0.10, 0.40, 1.0),
        (0.10, 0.70, 1.0),
        (0.15, 0.50, 1.0),
        (0.20, 0.60, 1.0),
    ]
    for nq in [2, 3, 4, 5]:
        for (rho1, rho2, rho3) in _a5_triplas:
            for variant in ['all-real', 'mixed']:
                real1 = True if variant == 'all-real' else False
                suites.append({
                    'suite':        f'A.5_{variant}',
                    'caso':         f'{variant}_{nq}q_rho{rho1:.2f}_{rho2:.2f}_{rho3:.2f}',
                    'nq_list':      [nq, nq, nq],
                    'nz_list':      [max(1, round(rho1*(2**nq))),
                                     max(1, round(rho2*(2**nq))),
                                     2**nq],
                    'real':         real1,
                    '_real_list':   [real1, True, True],
                })

    # -------------------------------------------------------------------
    # A.6 — 3 comp esparsos (nz=nq), até 15q (5q+5q+5q)
    # -------------------------------------------------------------------
    for total_nq in [6, 9, 12, 15]:
        nq = total_nq // 3
        for real, prefix in [(True,'real'),(False,'cplx')]:
            suites.append({
                'suite':   f'A.6_{prefix}',
                'caso':    f'{prefix}_3x{nq}q_esparso',
                'nq_list': [nq, nq, nq],
                'nz_list': [nq, nq, nq],
                'real':    real,
            })

    return suites


SUITES = _build_suites()

# ===========================================================================
# Checkpoint
# ===========================================================================

def load_checkpoint(csv_path):
    done = set()
    if not csv_path or not os.path.exists(csv_path):
        return done
    try:
        with open(csv_path, newline='', encoding='utf-8') as f:
            for row in csv.DictReader(f):
                key = (row['Suite'], row['Caso'],
                       row['Run'], row['Permutado'], row['Method'])
                done.add(key)
        print(f"[checkpoint] {len(done)} registros carregados de '{csv_path}'")
    except Exception as e:
        print(f"[checkpoint] aviso: {e}")
    return done


def is_done(done, suite, caso, run, permutado, method):
    return (suite, caso, str(run), str(permutado), method) in done

# ===========================================================================
# Runner de um caso
# ===========================================================================

def run_case(sv, suite, caso, nq_total, run, real, permutado,
             writer, f_csv, done_set, n_workers):

    sv = normalize_state(np.array(sv, dtype=complex))
    nz = int(np.count_nonzero(np.abs(sv) > 1e-9))

    base = {
        'Suite': suite, 'Caso': caso, 'Qubits': nq_total,
        'nz_total': nz, 'Run': run, 'Real': real, 'Permutado': permutado,
    }

    # pré-computa separabilidade — cache compartilhado por todos os pnm+
    result = analyze_ry_rz_clusters(sv)
    if result['can_decompose']:
        result     = _compute_substates(sv, result)
        decomposed = True
    else:
        decomposed = False

    # monta lista de tarefas pendentes
    tasks = []

    # solos
    for mname, (fn, tmo, skip_fn) in SOLO_METHODS.items():
        if skip_fn(nq_total, nz):
            # registra skip direto sem processo
            if not is_done(done_set, suite, caso, run, permutado, mname):
                writer.writerow({**base,
                    'Method': mname, 'CNOTs': -1, 'Depth': -1,
                    'Build (s)': 0, 'Transpile (s)': 0, 'Total (s)': 0,
                    'Submethods': '-', 'Decomposed': False,
                    'Fidelity': 'SKIP',
                })
                print(f"    {mname:<16} SKIP")
            continue
        if not is_done(done_set, suite, caso, run, permutado, mname):
            tasks.append((_worker_solo, (mname, fn, sv, BASIS_GATES),
                          mname, tmo))

    # pnm+ individuais
    if decomposed:
        for mname, (fn, tmo, skip_fn) in SOLO_METHODS.items():
            label = f'pnm+{mname}'
            if not is_done(done_set, suite, caso, run, permutado, label):
                tasks.append((_worker_pnm,
                               (mname, sv, BASIS_GATES, result),
                               label, tmo))

        # pnm+adaptive
        if not is_done(done_set, suite, caso, run, permutado, 'pnm+adaptive'):
            tasks.append((_worker_adaptive,
                           (sv, BASIS_GATES, result),
                           'pnm+adaptive', METHOD_TIMEOUT))

    if not tasks:
        print(f"    [skip] caso já completo no checkpoint")
        del sv, result
        gc.collect()
        return 0.0

    # executa em paralelo
    t0      = time.time()
    results = _run_batch_parallel(tasks, n_workers)
    t_caso  = time.time() - t0

    for label, r in results.items():
        writer.writerow({**base,
            'Method': label, 'CNOTs': r['cnots'], 'Depth': r['depth'],
            'Build (s)': r['build'], 'Transpile (s)': r['transpile'],
            'Total (s)': r['total'], 'Submethods': r.get('submethods', '-'),
            'Decomposed': decomposed, 'Fidelity': r['fidelity'],
        })
        print(f"    {label:<16} cnots={r['cnots']:>7}  "
              f"{r['fidelity']}  [{r.get('status','?')}]  "
              f"build={r['build']:.2f}s transp={r['transpile']:.2f}s")

    print(f"    {'TOTAL CASO':<16} {t_caso:>7.2f}s")
    f_csv.flush()

    del sv, result, results
    gc.collect()
    return t_caso

# ===========================================================================
# Main
# ===========================================================================

def main():
    parser = argparse.ArgumentParser(description='PNM Benchmark Adaptive v4')
    parser.add_argument('--workers', type=int, default=4)
    parser.add_argument('--resume',  type=str, default=None)
    args = parser.parse_args()

    n_workers = max(1, args.workers)
    print(f"[config] workers={n_workers}  resume='{args.resume}'")
    print(f"[config] suites: {len(SUITES)} definições × {N_RUNS} runs × 2 (orig+perm)")

    done_set = load_checkpoint(args.resume)

    timestamp  = datetime.now().strftime('%d%b%y_%H%M')
    csv_file   = f'benchB_adaptive_{timestamp}.csv'
    fieldnames = ['Suite','Caso','Qubits','nz_total','Run','Real','Permutado',
                  'Method','CNOTs','Depth','Build (s)','Transpile (s)',
                  'Total (s)','Submethods','Decomposed','Fidelity']

    f_csv  = open(csv_file, 'w', newline='', encoding='utf-8')
    writer = csv.DictWriter(f_csv, fieldnames=fieldnames)
    writer.writeheader()

    if args.resume and os.path.exists(args.resume):
        print(f"[checkpoint] copiando registros anteriores para '{csv_file}'...")
        with open(args.resume, newline='', encoding='utf-8') as f_old:
            for row in csv.DictReader(f_old):
                writer.writerow({k: row.get(k, '') for k in fieldnames})
        f_csv.flush()
        print("[checkpoint] ok.")

    tempo_total = 0.0
    total_iter  = 0

    for suite_def in SUITES:
        suite     = suite_def['suite']
        caso      = suite_def['caso']
        nq_list   = suite_def['nq_list']
        nz_list   = suite_def['nz_list']
        real      = suite_def['real']
        nq_total  = sum(nq_list)
        perm_list = _perm_map.get(nq_total, list(range(nq_total)))

        print(f"\n{'='*64}")
        print(f"  {suite} | {caso}  ({nq_total}q)")
        print(f"{'='*64}")

        for run in range(1, N_RUNS + 1):
            total_iter += 1
            seed = 42 + total_iter

            state_orig = generate_kron_state(nq_list, nz_list,
                                             seed=seed, real=real)
            state_perm = np.array(
                apply_bit_permutation_fast(state_orig.tolist(), perm_list))

            print(f"\n  Run {run}/{N_RUNS}  seed={seed}")
            t_run_start = time.time()

            print("\n\n")
            print("  [original]")
            run_case(state_orig, suite, caso, nq_total,
                     run, real, False, writer, f_csv, done_set, n_workers)

            print("\n\n")
            print("  [permutado]")
            run_case(state_perm, suite, caso, nq_total,
                     run, real, True, writer, f_csv, done_set, n_workers)

            t_run = time.time() - t_run_start
            tempo_total += t_run
            print(f"  >> Run {run} concluída em {t_run:.1f}s  |"
                  f"  acumulado: {tempo_total/60:.1f} min")

            del state_orig, state_perm
            gc.collect()

    f_csv.close()
    print(f"\n[done] CSV: {csv_file}")
    print(f"[done] tempo total: {tempo_total/60:.1f} min")


if __name__ == '__main__':
    mp.set_start_method('spawn', force=True)
    main()
