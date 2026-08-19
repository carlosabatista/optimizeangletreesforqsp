#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
pnm_benchmark_retry.py
----------------------
Reexecuta apenas os runs faltantes identificados em runs_faltantes.csv.

Uso:
    python pnm_benchmark_retry.py runs_faltantes.csv [checkpoint.json]

Saída:
    benchA_retry_<timestamp>.csv   — resultados dos runs reexecutados
    benchA_retry_falhas_<timestamp>.txt — erros persistentes (após MAX_RETRIES)
    benchA_retry_checkpoint_<timestamp>.json — checkpoint para retomada

Após validação, o CSV de retry pode ser mesclado ao CSV original com:
    python pnm_benchmark_retry.py --merge benchA_completo_orig.csv benchA_retry_<ts>.csv

Lógica de retry:
    MAX_RETRIES = 2  → se falhar, tenta mais 2 vezes (3 tentativas no total).
    Considera falha: status 'error' ou 'timeout'.
    Não considera falha: status 'ok' ou 'not_decomposed'.
"""

import os, sys, json, csv, time, gc, warnings, traceback
import multiprocessing as mp
from datetime import datetime
from pathlib import Path

import numpy as np
import matplotlib
matplotlib.use('Agg')

warnings.filterwarnings('ignore')

# ===========================================================================
# Re-importações do projeto original (idênticas ao benchmark original)
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
# Constantes (idênticas ao original)
# ===========================================================================

METHOD_TIMEOUT = 600
BAA_TIMEOUT    = 1200
N_WORKERS      = max(1, mp.cpu_count() - 1)
FIDELITY_TOL   = 1e-6
MAX_RETRIES    = 2   # tentativas extras após a primeira falha

FIELDNAMES = ['Suite', 'Caso', 'Qubits', 'nz_total', 'Run', 'Real',
              'Permutado', 'Method', 'CNOTs', 'Depth',
              'Build (s)', 'Transpile (s)', 'Total (s)', 'Decomposed', 'Fidelity']

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

sys.path.insert(0, str(Path(__file__).resolve().parent / "pnm"))


def get_perm(n):
    if n in PERM_MAP: return PERM_MAP[n]
    evens = list(range(0, n, 2)); odds = list(range(1, n, 2))
    return odds + evens

# ===========================================================================
# Definição das suites — re-usa make_* do original para obter list_nq/list_nz
# Copiadas aqui para o script ser auto-contido.
# ===========================================================================

def _make_cases_map():
    """Devolve dict label -> {'list_nq', 'list_nz', 'real', 'solo_only'}
    para todas as suites A.1..A.7, real e cplx."""
    cases = {}

    def add(label, list_nq, list_nz, real, solo_only=False):
        cases[label] = {'list_nq': list_nq, 'list_nz': list_nz,
                        'real': real, 'solo_only': solo_only}

    for real in [True, False]:
        tag = 'real' if real else 'cplx'

        # A.1
        cfgs = [
            (1,  [12],            [1.0],       True,  2),
            (2,  [6,6],           [1.0,1.0],   False, 5),
            (3,  [4,4,4],         [1.0]*3,     False, 5),
            (4,  [3,3,3,3],       [1.0]*4,     False, 5),
            (6,  [2,2,2,2,2,2],   [1.0]*6,     False, 5),
            (12, [1]*12,          [1.0]*12,    False, 5),
        ]
        for n_comp, list_nq, fracs, solo_only, _n_runs in cfgs:
            list_nz = [max(1, int(np.ceil((1 << nq) * f)))
                       for nq, f in zip(list_nq, fracs)]
            add(f'{tag}_12q_{n_comp:02d}comp', list_nq, list_nz, real, solo_only)

        # A.2
        for N in [4, 6, 8, 10, 12, 14, 16]:
            nq = N // 2
            add(f'{tag}_2x{nq}q_denso', [nq, nq], [2**nq, 2**nq], real)

        # A.3
        for N in [4, 6, 8, 10, 12, 14]:
            nq = N // 2
            nz = [max(1, int(np.ceil(d * 2**nq))) for d in [0.3, 0.7]]
            add(f'{tag}_2x{nq}q_medio', [nq, nq], nz, real)

        # A.4
        for N in [4, 6, 8, 10, 12, 14, 16]:
            nq = N // 2
            nz_each = max(1, nq)
            add(f'{tag}_2x{nq}q_esparso', [nq, nq], [nz_each, nz_each], real)

        # A.5
        for N in [6, 9, 12, 15]:
            nq = N // 3
            add(f'{tag}_3x{nq}q_denso', [nq]*3, [2**nq]*3, real)

        # A.6
        for N in [6, 9, 12, 15]:
            nq = N // 3
            nz_each = max(1, nq)
            add(f'{tag}_3x{nq}q_esparso', [nq]*3, [nz_each]*3, real)

        # A.7
        for N in [6, 9, 12, 15]:
            nq = N // 3
            nz = [max(1, int(np.ceil(d * 2**nq))) for d in [0.3, 0.5, 0.7]]
            add(f'{tag}_3x{nq}q_medio030507', [nq, nq], nz, real)

    return cases

CASES_MAP = _make_cases_map()

# ===========================================================================
# Workers (idênticos ao original)
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

METHODS_ORDER = ['ucge', 'ucge_dc', 'lrsp', 'baa', 'svd', 'pivot', 'merge']
PNM_METHODS   = ['ucge', 'ucge_dc', 'lrsp', 'baa', 'svd', 'pivot', 'merge']


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


def _worker_solo(q, method_name, sv, basis_gates):
    import warnings; warnings.filterwarnings('ignore')
    try:
        t0 = time.time()
        qc = SOLO_METHODS[method_name](sv)
        bt = time.time() - t0
        t1 = time.time()
        qct = transpile(qc, basis_gates=basis_gates, optimization_level=1)
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
        qct = transpile(qc, basis_gates=basis_gates, optimization_level=1)
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
    timeout = float(timeout)
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


def _is_failure(result):
    """Retorna True se o resultado deve ser retentado."""
    return result.get('status') in ('error', 'timeout')

# ===========================================================================
# Execução de um único método com retry
# ===========================================================================

def run_single_method_with_retry(method, sv, basis_gates, cached, decomposed):
    """Executa um método com até MAX_RETRIES tentativas extras em caso de falha.
    Retorna (result, attempts, gave_up).
    """
    is_pnm = method.startswith('pnm+')
    solo_name = method.replace('pnm+', '') if is_pnm else method

    # timeout adequado
    tmo = BAA_TIMEOUT if solo_name == 'baa' else METHOD_TIMEOUT

    for attempt in range(1 + MAX_RETRIES):
        if is_pnm:
            if not decomposed:
                return {'method': method, 'status': 'not_decomposed'}, attempt + 1, False
            result = _run_proc(_worker_pnm_plus,
                               (solo_name, sv, basis_gates, cached),
                               method, tmo)
        else:
            result = _run_proc(_worker_solo,
                               (solo_name, sv, basis_gates),
                               method, tmo)

        if not _is_failure(result):
            return result, attempt + 1, False

        if attempt < MAX_RETRIES:
            wait = 2 ** attempt  # backoff: 1s, 2s
            print(f"      [retry {attempt+1}/{MAX_RETRIES}] {method} falhou "
                  f"({result.get('status')}) — aguardando {wait}s...", flush=True)
            time.sleep(wait)
            gc.collect()

    # esgotou as tentativas
    return result, 1 + MAX_RETRIES, True

# ===========================================================================
# Reconstrução do estado vetorial (mesma seed do original)
# ===========================================================================

def rebuild_sv(caso, run, permutado):
    """Reconstrói o state vector exato usando a mesma seed do benchmark original."""
    info = CASES_MAP.get(caso)
    if info is None:
        raise ValueError(f"Caso não encontrado no CASES_MAP: '{caso}'. "
                         f"Verifique se o label está correto.")

    list_nq  = info['list_nq']
    list_nz  = info['list_nz']
    real     = info['real']
    nq_total = sum(list_nq)
    perm     = get_perm(nq_total)

    # ── seed idêntica ao original ──────────────────────────────────────────
    # O original usa: suite_name|label|run  (sem permutado na seed)
    # suite_name é deduzido do label
    suite_name = _label_to_suite(caso)
    seed = hash((suite_name, caso, run)) % (2**31)

    state_orig = generate_kron_state(list_nq, list_nz, seed=seed, real=real)
    if permutado:
        sv = normalize_state(np.array(apply_bit_permutation_fast(state_orig, perm)))
    else:
        sv = normalize_state(state_orig.copy())

    nz_total = int(np.count_nonzero(np.abs(sv) > 1e-9))
    return sv, nq_total, nz_total, real


def _label_to_suite(label):
    """Infere o nome da suite a partir do label do caso."""
    # Exemplos: real_12q_01comp -> A.1, cplx_2x7q_denso -> A.2, etc.
    if '12q' in label and 'comp' in label:
        return 'A.1_real' if label.startswith('real') else 'A.1_cplx'
    if '_2x' in label and 'denso' in label:
        return 'A.2_real' if label.startswith('real') else 'A.2_cplx'
    if '_2x' in label and 'medio' in label:
        return 'A.3_real' if label.startswith('real') else 'A.3_cplx'
    if '_2x' in label and 'esparso' in label:
        return 'A.4_real' if label.startswith('real') else 'A.4_cplx'
    if '_3x' in label and 'denso' in label:
        return 'A.5_real' if label.startswith('real') else 'A.5_cplx'
    if '_3x' in label and 'esparso' in label:
        return 'A.6_real' if label.startswith('real') else 'A.6_cplx'
    if '_3x' in label and 'medio' in label:
        return 'A.7_real' if label.startswith('real') else 'A.7_cplx'
    return 'UNKNOWN'

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
# CSV write
# ===========================================================================

def write_result(writer, f_csv, f_fail, suite, caso, nq, nz,
                 run, real, perm_str, result, decomposed, gave_up, attempts):
    if result.get('status') == 'not_decomposed':
        return
    row = {
        'Suite':        suite,
        'Caso':         caso,
        'Qubits':       nq,
        'nz_total':     nz,
        'Run':          run,
        'Real':         real,
        'Permutado':    perm_str,
        'Method':       result['method'],
        'CNOTs':        result['cnots'],
        'Depth':        result['depth'],
        'Build (s)':    result['build'],
        'Transpile (s)':result['transpile'],
        'Total (s)':    result['total'],
        'Decomposed':   decomposed,
        'Fidelity':     result['fidelity'],
    }
    writer.writerow(row)

    if gave_up or 'FAIL' in result.get('fidelity', ''):
        status_tag = f"GAVE_UP_after_{attempts}_attempts" if gave_up else "FIDELITY_FAIL"
        f_fail.write(
            f"Suite={suite} Caso={caso} Run={run} Perm={perm_str} "
            f"Method={result['method']} Status={result.get('status')} "
            f"Tag={status_tag} Attempts={attempts}\n"
        )
        f_fail.flush()
    f_csv.flush()

# ===========================================================================
# Modo merge — mescla retry CSV no CSV original
# ===========================================================================

def merge_csvs(original_csv, retry_csv):
    """Adiciona as linhas do retry_csv ao original_csv.
    Só insere linhas onde status é 'ok' (CNOTs >= 0).
    """
    import pandas as pd

    df_orig  = pd.read_csv(original_csv)
    df_retry = pd.read_csv(retry_csv)

    # filtra apenas sucessos do retry
    df_ok = df_retry[df_retry['CNOTs'] >= 0].copy()

    if len(df_ok) == 0:
        print("Nenhum resultado bem-sucedido no retry CSV. Nada foi mesclado.")
        return

    # chave de deduplicação
    key_cols = ['Suite', 'Caso', 'Run', 'Permutado', 'Method']
    df_merged = pd.concat([df_orig, df_ok], ignore_index=True)
    df_merged = df_merged.drop_duplicates(subset=key_cols, keep='last')
    df_merged = df_merged.sort_values(['Suite', 'Caso', 'Run', 'Permutado', 'Method'])

    ts = datetime.now().strftime("%d%b%y_%H%M")
    out = original_csv.replace('.csv', f'_merged_{ts}.csv')
    df_merged.to_csv(out, index=False)
    print(f"✓ Mesclado: {out}")
    print(f"  Original: {len(df_orig)} linhas")
    print(f"  Retry ok: {len(df_ok)} linhas adicionadas")
    print(f"  Resultado: {len(df_merged)} linhas")

# ===========================================================================
# Main
# ===========================================================================

if __name__ == '__main__':

    # ── modo merge ────────────────────────────────────────────────────────
    if len(sys.argv) >= 4 and sys.argv[1] == '--merge':
        merge_csvs(sys.argv[2], sys.argv[3])
        sys.exit(0)

    # ── modo retry ────────────────────────────────────────────────────────
    if len(sys.argv) < 2:
        print(__doc__)
        sys.exit(1)

    missing_csv = sys.argv[1]
    if not Path(missing_csv).exists():
        print(f"Arquivo não encontrado: {missing_csv}")
        sys.exit(1)

    timestamp = datetime.now().strftime("%d%b%y_%H%M")
    csv_out   = f"benchA_retry_{timestamp}.csv"
    fail_out  = f"benchA_retry_falhas_{timestamp}.txt"
    ckpt_file = sys.argv[2] if len(sys.argv) > 2 else f"benchA_retry_checkpoint_{timestamp}.json"

    # ── carrega lista de runs faltantes ───────────────────────────────────
    with open(missing_csv, newline='', encoding='utf-8') as f:
        reader = csv.DictReader(f)
        missing_runs = list(reader)

    print(f"Runs faltantes carregados: {len(missing_runs)}")
    print(f"CSV de saída:  {csv_out}")
    print(f"Checkpoint:    {ckpt_file}")
    print(f"Max retries:   {MAX_RETRIES} (total de {1+MAX_RETRIES} tentativas)")

    checkpoint = load_ckpt(ckpt_file)
    csv_mode   = 'a' if Path(csv_out).exists() else 'w'
    f_csv      = open(csv_out,  csv_mode, newline='', encoding='utf-8')
    f_fail     = open(fail_out, 'a', encoding='utf-8')
    writer     = csv.DictWriter(f_csv, fieldnames=FIELDNAMES)
    if csv_mode == 'w':
        writer.writeheader()
    f_fail.write(f"\n# Sessão retry {timestamp}\n")

    basis_gates = ['u', 'cx']

    # agrupa por (Suite, Caso, Run, Permutado) para processar um estado por vez
    # (múltiplos métodos podem compartilhar o mesmo sv e cached)
    from itertools import groupby

    def group_key(r):
        return (r['Suite'], r['Caso'], r['Run'], r['Permutado'])

    missing_sorted = sorted(missing_runs, key=group_key)
    groups = [(k, list(v)) for k, v in groupby(missing_sorted, key=group_key)]

    total_groups  = len(groups)
    total_methods = len(missing_runs)
    done_methods  = 0
    gave_up_count = 0

    for g_idx, ((suite, caso, run_str, perm_str), methods_in_group) in enumerate(groups, 1):
        run      = int(run_str)
        permutado = perm_str.strip().lower() in ('true', '1')

        ck_key_group = f"{suite}|{caso}|{run}|{perm_str}"

        print(f"\n[{g_idx}/{total_groups}] {suite} | {caso} | run={run} | perm={perm_str}")

        # ── reconstrói sv ─────────────────────────────────────────────────
        try:
            sv, nq, nz_total, real = rebuild_sv(caso, run, permutado)
        except Exception as e:
            print(f"  ERRO ao reconstruir estado: {e}")
            traceback.print_exc()
            for mr in methods_in_group:
                f_fail.write(f"Suite={suite} Caso={caso} Run={run} Perm={perm_str} "
                             f"Method={mr['Method']} Status=rebuild_error Error={e}\n")
            f_fail.flush()
            continue

        if not is_valid_state_vector(sv):
            print(f"  [PULADO: vetor de estado inválido, tamanho={len(sv)}]")
            for mr in methods_in_group:
                f_fail.write(f"Suite={suite} Caso={caso} Run={run} Perm={perm_str} "
                             f"Method={mr['Method']} Status=invalid_state_vector "
                             f"(tamanho={len(sv)})\n")
                ck_key = f"{suite}|{caso}|{run}|{perm_str}|{mr['Method']}"
                mark_done(checkpoint, ckpt_file, ck_key)
            f_fail.flush()
            continue

        # ── análise de separabilidade (uma vez por estado) ─────────────────
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

        perm_str_out = 'True' if permutado else 'False'

        # ── itera métodos faltantes deste estado ─────────────────────────
        for mr in methods_in_group:
            method = mr['Method']
            ck_key = f"{suite}|{caso}|{run}|{perm_str}|{method}"

            if is_done(checkpoint, ck_key):
                print(f"  skip {method} (já feito)")
                done_methods += 1
                continue

            print(f"  {method} ...", end=' ', flush=True)
            t_start = time.time()

            result, attempts, gave_up = run_single_method_with_retry(
                method, sv, basis_gates, cached, decomposed
            )

            elapsed = time.time() - t_start
            status  = result.get('status', '?')

            if gave_up:
                gave_up_count += 1
                print(f"GAVE UP após {attempts} tentativas "
                      f"({status}) — {elapsed:.1f}s")
            else:
                print(f"{status} em {attempts} tentativa(s) — {elapsed:.1f}s")

            write_result(writer, f_csv, f_fail,
                         suite, caso, nq, nz_total,
                         run, real, perm_str_out,
                         result, decomposed, gave_up, attempts)

            mark_done(checkpoint, ckpt_file, ck_key)
            done_methods += 1

        gc.collect()

    f_csv.close()
    f_fail.close()

    print(f"\n{'='*62}")
    print(f"✓ Retry concluído")
    print(f"  Métodos processados : {done_methods}/{total_methods}")
    print(f"  Gave up (falha def.): {gave_up_count}")
    print(f"  CSV retry:           {csv_out}")
    print(f"  Falhas:              {fail_out}")
    print(f"  Checkpoint:          {ckpt_file}")
    print(f"\nPara mesclar ao CSV original:")
    print(f"  python pnm_benchmark_retry.py --merge benchA_completo_orig.csv {csv_out}")
