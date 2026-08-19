# -*- coding: utf-8 -*-
"""
quantum_state_vector_utils.py
-------------------------------
Versão mínima: mantém apenas normalize_state, generate_random_state_n_m,
vector_to_binary_dict e is_valid_state_vector.

REMOVIDO nesta rodada de enxugamento: a classe clStateVector. Ela
calculava ~8 estatísticas do vetor (is_real, num_unique_amplitudes,
density, uniformity, simetry, non_zeros_concentration, fidelity) toda
vez que um estado era carregado — varreduras O(n) no vetor inteiro.
Levantamento em todo o pipeline (pnm_main.py, pnm_cluster_analysis.py,
pnm_benchmark_suite_a.py, pnm_benchmark_retry.py) confirmou que só dois
"atributos" eram de fato usados em algum lugar: o vetor em si e o número
de qubits — ambos triviais sem a classe (o vetor já normalizado, e
num_qubits = len(vetor).bit_length() - 1). Nenhum dos benchmarks lê
density/uniformity/simetry/etc. em lugar nenhum (nz_total no CSV vem de
np.count_nonzero direto, não da classe). Os dois benchmarks chegavam a
embrulhar o vetor na classe DUAS VEZES separadas por chamada, sem
reaproveitar nada entre elas — sinal de que não guardava nada que valesse
a pena reaproveitar.

is_valid_state_vector substitui a validação que a classe fazia no
__init__ (tamanho potência de 2, norma não-nula) — mas como função pura,
que retorna True/False em vez de levantar exceção. A decisão do que
fazer com um vetor inválido fica com o chamador: as funções de biblioteca
(analyze_ry_rz_clusters, build_pnm_with_method) levantam ValueError com
mensagem clara; os laços de benchmark verificam antes de disparar
qualquer método e pulam o estado (registrando como inválido), sem
interromper a rodada inteira.
"""

import numpy as np


def normalize_state(state_vector):
    return state_vector / np.linalg.norm(state_vector)


def is_valid_state_vector(v):
    """
    Validação barata: True se v tem tamanho potência de 2 (>0) e norma
    finita e não-nula. Não lança exceção.
    """
    n = len(v)
    if n == 0 or (n & (n - 1)) != 0:
        return False
    norm = np.linalg.norm(v)
    return bool(np.isfinite(norm) and norm > 1e-12)


def generate_random_state_n_m(num_qubits, m, seed, bRealState):
    """
    Parameters
    ----------
    num_qubits (int)        : number of qubits of the state to be created.
    m          (int)        : number of non-zero amplitudes, 0 <= m <= 2**num_qubits.
    seed       (int or None): seed for random number generator.
    bRealState (bool)       : if True, generate real-valued state; otherwise, complex-valued.

    Returns
    -------
    v : array of complex or real values; a state of num_qubits qubits, m of which are non-zero.
    """
    if num_qubits < 1:
        raise ValueError("N must be greater or equal 1.")

    szState = 1 << num_qubits
    if m < 0 or m > szState:
        raise ValueError(f"m must be in interval [0,{szState}], given {m}")

    if seed is not None:
        np.random.seed(seed)

    if bRealState:
        v = np.random.rand(szState) * np.random.choice([1, -1], size=szState)
    else:
        v = np.random.rand(szState) + np.random.rand(szState) * 1j

    expulsar = szState - m
    indices = np.random.choice(szState, size=expulsar, replace=False)
    v[indices] = 0

    return normalize_state(v)


def vector_to_binary_dict(vector, precision=None, absvalue=None):
    result = {}
    szVector = len(vector)
    iNumQubits = szVector.bit_length() - 1
    if absvalue:
        vector = abs(vector)
    for i in range(szVector):
        if not np.isclose(vector[i], 0):
            binary_index = format(i, f'0{iNumQubits}b')
            if precision:
                result[binary_index] = np.round(vector[i], precision)
            else:
                result[binary_index] = vector[i]
    return result
