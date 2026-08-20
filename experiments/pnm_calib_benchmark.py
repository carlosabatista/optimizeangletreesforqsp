#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
pnm_calib_benchmark.py
-----------------------
Benchmark de calibração do seletor adaptativo PNM v3.

Roda todos os 7 métodos QSP sobre sub-estados isolados (não estados produto),
coletando features descritivas de cada sub-estado para posterior análise e
recalibração do algoritmo de seleção.

Features coletadas por sub-estado:
    nq          — número de qubits
    nz          — número de amplitudes não-nulas
    rho         — esparsidade global (nz / 2^nq)
    real        — booleano (True = amplitudes reais)
    gini        — coeficiente de Gini das |amplitudes|²
    entropy     — entropia de Shannon das |amplitudes|² (bits)
    schmidt_rank — rank numérico da bipartição balanceada (SVD, tol=1e-10)

Métodos comparados por sub-estado:
    ucge_dc, baa, merge, ucge, svd, lrsp, pivot

Grade de parâmetros:
    nq    : 1..8
    nz    : 8 pontos log de 1 até 2^nq (inclusive)
    real  : True, False
    seeds : 20 por combinação (nq, nz, real)

CSV de saída (formato longo): calib_bench_<timestamp>.csv
    Uma linha por (sub-estado × método).

Uso:
    python pnm_calib_benchmark.py
    python pnm_calib_benchmark.py --workers 4
    python pnm_calib_benchmark.py --resume calib_bench_XXXXX.csv --workers 4
"""

import os, csv, time, gc, warnings, argparse
import sys
from pathlib import Path
import multiprocessing as mp
from datetime import datetime

import numpy as np

warnings.filterwarnings('ignore')

# torna a pasta pnm/ (irmã de benchmarks/, ambas sob a raiz do repositório)
# importável — este arquivo mora em benchmarks/, então sobe um nível até a
# raiz e desce em pnm/.
sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "pnm"))

from qclib.state_preparation import (
    BaaLowRankInitialize, MergeInitialize, UCGEInitialize,
    LowRankInitialize, SVDInitialize, PivotInitialize,
)
from qiskit import QuantumCircuit, transpile
from qiskit.quantum_info import Statevector

from quantum_state_vector_utils import (
    vector_to_binary_dict, normalize_state, generate_random_state_n_m,
)
from ucge_dc import UCGEDCInitialize
from pnm_main import prepare_basis_state

# ===========================================================================
# Configuração
# ===========================================================================

METHOD_TIMEOUT = 600
BAA_TIMEOUT    = 600       # mesmo patamar para sub-estados isolados ≤8q
FIDELITY_TOL   = 1e-6
BASIS_GATES    = ['u', 'cx']
N_SEEDS        = 20

# ===========================================================================
# Features descritivas
# ===========================================================================

def gini(probs: np.ndarray) -> float:
    """Coeficiente de Gini das probabilidades (|amp|²). 0=uniforme, 1=concentrado."""
    p = np.sort(probs)
    n = len(p)
    if n == 0 or p.sum() == 0:
        return 0.0
    cumsum = np.cumsum(p)
    return float(1 - 2 * cumsum[:-1].sum() / (n * p.sum()) - 1/n)


def shannon_entropy(probs: np.ndarray) -> float:
    """Entropia de Shannon em bits."""
    p = probs[probs > 0]
    return float(-np.sum(p * np.log2(p)))


def schmidt_rank(sv: np.ndarray, nq: int) -> int:
    """Rank numérico da bipartição balanceada (tol=1e-10)."""
    k  = nq // 2
    if k == 0:
        return 1
    rows = 2 ** k
    cols = 2 ** (nq - k)
    try:
        M   = sv.reshape(rows, cols)
        sv_ = np.linalg.svd(M, compute_uv=False)
        return int(np.sum(sv_ > 1e-10))
    except Exception:
        return -1


def extract_features(sv: np.ndarray, nq: int, real: bool) -> dict:
    probs = np.abs(sv) ** 2
    nz    = int(np.count_nonzero(np.abs(sv) > 1e-10))
    rho   = nz / (2 ** nq)
    return {
        'nq':           nq,
        'nz':           nz,
        'rho':          round(rho, 6),
        'real':         real,
        'gini':         round(gini(probs), 6),
        'entropy':      round(shannon_entropy(probs), 6),
        'schmidt_rank': schmidt_rank(sv, nq),
    }

# ===========================================================================
# Inicializadores
# ===========================================================================

def _init_ucge_dc(sv): return UCGEDCInitialize(sv).definition

def _init_ucge(sv):    return UCGEInitialize(sv).definition

def _init_baa(sv):
    return BaaLowRankInitialize(
        sv, opt_params={'max_fidelity_loss': 1e-6, 'strategy': 'brute_force'}
    ).definition

def _init_merge(sv):
    sv2 = sv.copy(); sv2[np.abs(sv2) < 1e-8] = 0
    if np.count_nonzero(sv2) == 0:
        return QuantumCircuit(len(sv2).bit_length() - 1)
    return MergeInitialize(vector_to_binary_dict(sv)).definition

def _init_svd(sv):
    sv2 = sv.copy()
    sv2[np.abs(sv2) < 1e-8] = 0
    if np.count_nonzero(sv2) == 1:
        k = np.nonzero(sv2)[0][0]
        return prepare_basis_state(k, len(sv).bit_length() - 1)
    return SVDInitialize(sv).definition

def _init_lrsp(sv):
    return LowRankInitialize(sv, opt_params={"unitary_scheme": "ccd"}).definition

def _init_pivot(sv):
    sv2 = sv.copy()
    sv2[np.abs(sv2) < 1e-8] = 0
    if np.count_nonzero(sv2) == 1:
        k = np.nonzero(sv2)[0][0]
        return prepare_basis_state(k, len(sv).bit_length() - 1)
    return PivotInitialize(vector_to_binary_dict(sv)).definition

ALL_METHODS = {
    'ucge_dc': (_init_ucge_dc, METHOD_TIMEOUT),
    'ucge':    (_init_ucge,    METHOD_TIMEOUT),
    'baa':     (_init_baa,     BAA_TIMEOUT),
    'merge':   (_init_merge,   METHOD_TIMEOUT),
    'svd':     (_init_svd,     METHOD_TIMEOUT),
    'lrsp':    (_init_lrsp,    METHOD_TIMEOUT),
    'pivot':   (_init_pivot,   METHOD_TIMEOUT),
}

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
            float(abs(np.dot(np.conj(v), np.array(sv, dtype=complex)))**2)
            for sv in [sv_d, sv_r]
        )
        ok = best >= 1.0 - FIDELITY_TOL
        return best, f"{best:.8f} ({'OK' if ok else 'FAIL'})"
    except Exception as e:
        return 0.0, f"N/A ({e})"

# ===========================================================================
# Worker
# ===========================================================================

def _worker(q, method_name, init_fn, sv, basis_gates):
    import warnings; warnings.filterwarnings('ignore')
    try:
        t0  = time.time()
        qc  = init_fn(sv)
        bt  = time.time() - t0
        t1  = time.time()
        qct = transpile(qc, basis_gates=basis_gates, optimization_level=0)
        tt  = time.time() - t1
        fid, fstr = fidelity_check(sv, qc)
        q.put({
            'method':    method_name,
            'cnots':     qct.count_ops().get('cx', 0),
            'depth':     qct.depth(),
            'build_s':   round(bt, 6),
            'transp_s':  round(tt, 6),
            'total_s':   round(bt + tt, 6),
            'fidelity':  fstr,
            'status':    'ok',
        })
    except Exception as e:
        q.put({
            'method':   method_name,
            'cnots':    -1, 'depth': -1,
            'build_s':  0,  'transp_s': 0, 'total_s': 0,
            'fidelity': f'ERROR ({e})',
            'status':   'error',
        })


# ===========================================================================
# Execução paralela dos métodos de um sub-estado
# ===========================================================================

def run_methods_parallel(sv, n_workers):
    """Roda todos os métodos em janelas de n_workers processos simultâneos."""
    tasks = [(name, fn, tmo) for name, (fn, tmo) in ALL_METHODS.items()]
    results = []
    i = 0
    while i < len(tasks):
        batch = tasks[i:i + n_workers]
        procs = []
        for name, fn, tmo in batch:
            q = mp.Queue()
            p = mp.Process(target=_worker,
                           args=(q, name, fn, sv, BASIS_GATES),
                           daemon=True)
            p.start()
            procs.append((p, q, name, tmo))

        for p, q, name, tmo in procs:
            p.join(float(tmo))
            if p.is_alive():
                p.terminate(); p.join(5)
                if p.is_alive(): p.kill(); p.join()
                results.append({
                    'method':   name,
                    'cnots':    -1, 'depth': -1,
                    'build_s':  tmo, 'transp_s': 0, 'total_s': tmo,
                    'fidelity': f'TIMEOUT (>{tmo}s)',
                    'status':   'timeout',
                })
            else:
                try:
                    results.append(q.get_nowait())
                except Exception:
                    results.append({
                        'method':   name,
                        'cnots':    -1, 'depth': -1,
                        'build_s':  0, 'transp_s': 0, 'total_s': 0,
                        'fidelity': 'ERROR',
                        'status':   'error',
                    })
        i += n_workers
    return results

# ===========================================================================
# Grade de parâmetros
# ===========================================================================

def build_grid():
    """
    Retorna lista de (nq, nz, real, seed).
    nz: 8 pontos log de 1 até 2^nq, sempre incluindo 1 e 2^nq.
    """
    grid = []
    for nq in range(1, 9):
        max_nz = 2 ** nq
        # pontos log entre 1 e max_nz
        if max_nz <= 8:
            nz_points = list(range(1, max_nz + 1))
        else:
            log_pts = np.exp(np.linspace(np.log(1), np.log(max_nz), 8))
            nz_points = sorted(set(
                [1] + [int(round(x)) for x in log_pts] + [max_nz]
            ))
            # garante no máximo 8 pontos (remove duplicatas pelo set)
            # se ficou mais que 8, subamostra mantendo extremos
            if len(nz_points) > 8:
                interior = nz_points[1:-1]
                step = max(1, len(interior) // 6)
                nz_points = [1] + interior[::step][:6] + [max_nz]
                nz_points = sorted(set(nz_points))

        for nz in nz_points:
            for real in (True, False):
                for seed in range(N_SEEDS):
                    grid.append((nq, nz, real, seed))
    return grid

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
                key = (row['nq'], row['nz'], row['real'], row['seed'], row['method'])
                done.add(key)
        print(f"[checkpoint] {len(done)} registros carregados de '{csv_path}'")
    except Exception as e:
        print(f"[checkpoint] aviso: {e}")
    return done


def is_done(done, nq, nz, real, seed, method):
    return (str(nq), str(nz), str(real), str(seed), method) in done

# ===========================================================================
# Main
# ===========================================================================

def main():
    parser = argparse.ArgumentParser(description='PNM Calibration Benchmark')
    parser.add_argument('--workers', type=int, default=2)
    parser.add_argument('--resume',  type=str, default=None)
    args = parser.parse_args()

    n_workers = max(1, args.workers)
    grid      = build_grid()

    print(f"[config] workers={n_workers}  resume='{args.resume}'")
    print(f"[config] grid: {len(grid)} combinações × {len(ALL_METHODS)} métodos = "
          f"{len(grid)*len(ALL_METHODS)} execuções estimadas")

    done = load_checkpoint(args.resume)

    timestamp  = datetime.now().strftime('%d%b%y_%H%M')
    csv_file   = f'calib_bench_{timestamp}.csv'
    fieldnames = [
        'nq', 'nz', 'rho', 'real', 'seed',
        'gini', 'entropy', 'schmidt_rank',
        'method', 'cnots', 'depth',
        'build_s', 'transp_s', 'total_s',
        'fidelity', 'status',
    ]

    f_csv  = open(csv_file, 'w', newline='', encoding='utf-8')
    writer = csv.DictWriter(f_csv, fieldnames=fieldnames)
    writer.writeheader()

    # copia checkpoint anterior para o novo CSV
    if args.resume and os.path.exists(args.resume):
        print(f"[checkpoint] copiando registros anteriores para '{csv_file}'...")
        with open(args.resume, newline='', encoding='utf-8') as f_old:
            for row in csv.DictReader(f_old):
                writer.writerow({k: row.get(k, '') for k in fieldnames})
        f_csv.flush()
        print("[checkpoint] ok.")

    tempo_total = 0.0
    n_done_prev = len(done) // len(ALL_METHODS)   # sub-estados já completos
    n_total     = len(grid)

    for idx, (nq, nz, real, seed) in enumerate(grid):
        # verifica se todos os métodos deste sub-estado já estão no checkpoint
        metodos_pendentes = [
            m for m in ALL_METHODS
            if not is_done(done, nq, nz, real, seed, m)
        ]
        if not metodos_pendentes:
            continue

        # gera sub-estado
        sv = generate_random_state_n_m(nq, nz, seed, real)
        sv = normalize_state(np.array(sv, dtype=complex))

        feats = extract_features(sv, nq, real)

        tipo  = 'real' if real else 'cplx'
        print(f"\n[{idx+1}/{n_total}] nq={nq} nz={nz} rho={feats['rho']:.3f} "
              f"{tipo} seed={seed} | "
              f"gini={feats['gini']:.3f} H={feats['entropy']:.3f} "
              f"rank={feats['schmidt_rank']}")

        t0      = time.time()
        results = run_methods_parallel(sv, n_workers)
        t_caso  = time.time() - t0
        tempo_total += t_caso

        best_cnots = min(
            (r['cnots'] for r in results if r['cnots'] >= 0),
            default=-1
        )

        for r in results:
            if r['method'] not in metodos_pendentes:
                continue
            marker = '*' if r['cnots'] == best_cnots and best_cnots >= 0 else ' '
            print(f"  {marker}{r['method']:<10} cnots={r['cnots']:>6}  "
                  f"{r['fidelity']}  [{r['status']}]  "
                  f"build={r['build_s']:.3f}s transp={r['transp_s']:.3f}s")
            writer.writerow({
                **feats,
                'seed':    seed,
                'method':  r['method'],
                'cnots':   r['cnots'],
                'depth':   r['depth'],
                'build_s': r['build_s'],
                'transp_s':r['transp_s'],
                'total_s': r['total_s'],
                'fidelity':r['fidelity'],
                'status':  r['status'],
            })

        f_csv.flush()
        print(f"  TOTAL CASO {t_caso:>7.2f}s  |  acumulado: {tempo_total/60:.1f} min")

        del sv, results
        gc.collect()

    f_csv.close()
    print(f"\n[done] CSV: {csv_file}")
    print(f"[done] tempo total: {tempo_total/60:.1f} min")


if __name__ == '__main__':
    mp.set_start_method('spawn', force=True)
    main()
