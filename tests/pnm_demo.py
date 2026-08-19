# -*- coding: utf-8 -*-
"""
pnm_demo.py
------------
Demonstração mínima do PNM (Prune N' Merge) — roda em poucos segundos e
mostra, lado a lado, o ganho de CNOTs do PNM sobre os métodos "solo" de
preparação de estado, em estados produto pequenos e aleatórios.

Este script existe para você (leitor/revisor) confirmar rapidamente que
o código funciona, antes de alterar os parâmetros abaixo para
experimentos maiores (mais qubits, mais rodadas, outros métodos).

Uso:
    python3 pnm_demo.py

Requer que a pasta pnm/ (com angle_tree_core.py, angle_tree_reduction.py,
pnm_cluster_analysis.py, pnm_main.py, quantum_state_vector_utils.py,
ucge_dc.py) esteja no mesmo diretório deste script — é adicionada
automaticamente ao caminho de import abaixo, não precisa configurar
PYTHONPATH manualmente.
Dependências externas: numpy, networkx, qiskit, qiskit_aer, qclib.
"""

import io
import sys
import time
import warnings
from pathlib import Path

warnings.filterwarnings('ignore')

sys.path.insert(0, str(Path(__file__).resolve().parent / "pnm"))

import numpy as np
from qiskit import transpile
from qiskit.quantum_info import Statevector

from pnm_main import (
    generate_kron_state,
    build_pnm_with_method,
    _compute_substates,
    initializeLR,
    initializeUCGE,
    initializeSVD,
)
from pnm_cluster_analysis import analyze_ry_rz_clusters


# ===========================================================================
# CONFIGURAÇÃO — altere aqui para experimentos maiores
# ===========================================================================

# Número de qubits de cada componente do estado produto. O exemplo usa um
# estado de 2 componentes (3+3=6 qubits) para rodar em menos de 1 segundo.
# Para um caso maior, tente por exemplo LIST_NQ = [6, 6] (12 qubits) ou
# LIST_NQ = [4, 4, 4] (12 qubits, 3 componentes).
LIST_NQ = [3, 3]

# Número de amplitudes não-nulas por componente (densidade do estado).
# LIST_NZ[i] = 2**LIST_NQ[i] → componente totalmente denso.
# Para estados esparsos, use algo bem menor, ex.: LIST_NZ = [3, 3].
LIST_NZ = [2**n for n in LIST_NQ]

# Estado real (True) ou complexo (False).
REAL = True

# Métodos "solo" a comparar contra o PNM. Métodos disponíveis:
# 'lrsp', 'baa', 'ucge', 'ucge_dc', 'svd', 'pivot', 'merge'.
# Mantidos poucos aqui para o demo rodar rápido — baa/merge tendem a ser
# mais lentos, especialmente em estados maiores.
METHODS = ['lrsp', 'ucge', 'svd']

# Número de estados aleatórios distintos a testar (aumente para uma
# amostra estatística maior).
N_RUNS = 5

# Seed inicial (cada run usa SEED_BASE + run, de forma determinística).
SEED_BASE = 42


# ===========================================================================
# Funções auxiliares do demo
# ===========================================================================

SOLO_FNS = {
    'lrsp': initializeLR,
    'ucge': initializeUCGE,
    'svd':  initializeSVD,
}


def _silent(fn, *args, **kwargs):
    """Executa fn suprimindo os relatórios impressos por analyze_ry_rz_clusters."""
    old_stdout = sys.stdout          # 1. guarda o "terminal atual" numa variável
    sys.stdout = io.StringIO()       # 2. troca o terminal por uma caixa vazia na memória
    try:
        return fn(*args, **kwargs)   # 3. chama fn com os argumentos recebidos, e devolve o resultado
    finally:
        sys.stdout = old_stdout      # 4. devolve o terminal de verdade, sempre (mesmo se der erro)


def _fidelity(state, qc):
    v = np.array(state, dtype=complex)
    v /= np.linalg.norm(v)
    sv_d = Statevector.from_instruction(qc).data
    sv_r = Statevector.from_instruction(qc).reverse_qargs().data
    fd = float(abs(np.dot(np.conj(v), np.array(sv_d))) ** 2)
    fr = float(abs(np.dot(np.conj(v), np.array(sv_r))) ** 2)
    return max(fd, fr)


def _cnots(qc):
    qct = transpile(qc, basis_gates=['u', 'cx'], optimization_level=0)
    return qct.count_ops().get('cx', 0)


def run_one_state(state, methods):
    """
    Roda os métodos solo e os pnm+ equivalentes num único estado.
    Retorna uma lista de dicts com CNOTs/fidelidade/tempo de cada método,
    para as duas versões (solo e pnm+).
    """
    rows = []

    # solo: cada método direto no estado inteiro, sem decompor
    for method in methods:
        t0 = time.perf_counter()
        qc = SOLO_FNS[method](state)
        bt = time.perf_counter() - t0
        rows.append({
            'method': method, 'variant': 'solo',
            'cnots': _cnots(qc), 'fidelity': _fidelity(state, qc), 'build_s': bt,
        })

    # pnm+: decompõe uma vez (em cache), reaproveitando para todos os métodos
    result = _silent(analyze_ry_rz_clusters, state)
    decomposed = result['can_decompose']
    if decomposed:
        result = _compute_substates(state, result)

    for method in methods:
        if not decomposed:
            rows.append({
                'method': method, 'variant': 'pnm+',
                'cnots': None, 'fidelity': None, 'build_s': None,
            })
            continue
        qc, bt = build_pnm_with_method(state, method, cached_result=result)
        rows.append({
            'method': method, 'variant': 'pnm+',
            'cnots': _cnots(qc), 'fidelity': _fidelity(state, qc), 'build_s': bt,
        })

    return rows, decomposed


# ===========================================================================
# Programa principal
# ===========================================================================

def main():
    print("=" * 72)
    print("PNM — demonstração mínima")
    print("=" * 72)
    print(f"Estado: {len(LIST_NQ)} componente(s) de {LIST_NQ} qubits "
          f"(total={sum(LIST_NQ)}q), nz={LIST_NZ}, "
          f"{'real' if REAL else 'complexo'}")
    print(f"Métodos: {METHODS}    Rodadas: {N_RUNS}")
    print("=" * 72)
    print()

    all_rows = []
    t_start = time.perf_counter()

    for run in range(1, N_RUNS + 1):
        state = generate_kron_state(LIST_NQ, LIST_NZ,
                                     seed=SEED_BASE + run, real=REAL)
        rows, decomposed = run_one_state(state, METHODS)
        for r in rows:
            r['run'] = run
        all_rows.extend(rows)

        tag = "separável" if decomposed else "NÃO separável"
        print(f"--- run {run}  ({tag}) ---")
        for method in METHODS:
            solo = next(r for r in rows if r['method'] == method and r['variant'] == 'solo')
            pnm  = next(r for r in rows if r['method'] == method and r['variant'] == 'pnm+')
            if pnm['cnots'] is None:
                print(f"  {method:6s}  solo: CNOTs={solo['cnots']:4d}  "
                      f"fid={solo['fidelity']:.6f}   pnm+: não separável, pulado")
            else:
                ganho = solo['cnots'] / pnm['cnots'] if pnm['cnots'] > 0 else float('inf')
                print(f"  {method:6s}  solo: CNOTs={solo['cnots']:4d}  "
                      f"fid={solo['fidelity']:.6f}   |   "
                      f"pnm+: CNOTs={pnm['cnots']:4d}  fid={pnm['fidelity']:.6f}   "
                      f"[ganho {ganho:.2f}x]")
        print()

    t_total = time.perf_counter() - t_start

    # resumo por método
    print("=" * 72)
    print("RESUMO (média sobre as rodadas onde o estado foi separável)")
    print("=" * 72)
    print(f"{'método':8s} {'CNOTs solo':>12s} {'CNOTs pnm+':>12s} {'ganho médio':>12s}")
    for method in METHODS:
        solo_vals = [r['cnots'] for r in all_rows
                     if r['method'] == method and r['variant'] == 'solo']
        pnm_vals = [r['cnots'] for r in all_rows
                    if r['method'] == method and r['variant'] == 'pnm+' and r['cnots'] is not None]
        if not pnm_vals:
            print(f"{method:8s}  (nenhuma rodada separável para este método)")
            continue
        solo_mean = sum(solo_vals) / len(solo_vals)
        pnm_mean = sum(pnm_vals) / len(pnm_vals)
        ganho = solo_mean / pnm_mean if pnm_mean > 0 else float('inf')
        print(f"{method:8s} {solo_mean:12.1f} {pnm_mean:12.1f} {ganho:11.2f}x")

    n_ok = sum(1 for r in all_rows if r['fidelity'] is not None and r['fidelity'] >= 1 - 1e-6)
    n_total = sum(1 for r in all_rows if r['fidelity'] is not None)
    print()
    print(f"Fidelidade >= 1-1e-6 em {n_ok}/{n_total} construções de circuito.")
    print(f"Tempo total do demo: {t_total:.2f}s")
    print()
    print("Para experimentos maiores, edite o bloco CONFIGURAÇÃO no topo")
    print("deste arquivo (LIST_NQ, LIST_NZ, METHODS, N_RUNS).")


if __name__ == "__main__":
    main()
