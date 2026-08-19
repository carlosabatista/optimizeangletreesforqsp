#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
pnm_adaptive_selector.py
------------------------
Seletor de método QSP por sub-estado, calibrado a partir dos
resultados do benchmark Suite A (benchA_completo_21Mar26_2228.csv).

Padrões observados na análise exploratória:
  nq_sub <= 2            → ucge_dc  (vence 100% dos casos)
  nq_sub == 3, esparso   → ucge_dc  (vence 51%)
  nq_sub == 3, outros    → ucge_dc  (vence 54% denso, empate com baa)
  nq_sub >= 4, rho<0.15  → merge    (vence 55-100% a partir de nq=5)
  nq_sub == 4, esparso   → baa      (vence 57%, merge 5%)
  nq_sub >= 4, rho>=0.15 → baa      (vence 80-100%)

Uso:
    from pnm_adaptive_selector import select_method, build_pnm_adaptive

    # método para um sub-estado individual
    method = select_method(nq_sub, rho_sub)

    # circuito completo com seleção adaptativa
    qc, bt = build_pnm_adaptive(state_vector, cached_result=result)
"""

import time
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "pnm"))

import numpy as np
from qiskit import QuantumCircuit
from quantum_state_vector_utils import normalize_state, vector_to_binary_dict, is_valid_state_vector
from angle_tree_core import get_state_tree
from pnm_main import (
    initializeUCGE, initializeUCGEDC, initializeLR, initializeBAA,
    initializeSVD, initializePivot,
    _fix_subgroup_phase, _fix_global_phase,
    _compute_substates,
)
from pnm_cluster_analysis import analyze_ry_rz_clusters
from qclib.state_preparation import MergeInitialize


# ===========================================================================
# Limiares calibrados pelo benchmark Suite A
# ===========================================================================

_RHO_MERGE_THRESHOLD = {
    2: 0.0,    # ucge_dc
    3: 0.0,    # ucge_dc
    4: 0.15,   # merge só vence se rho < 15% (e mesmo assim baa ainda compete)
    5: 0.20,   # merge domina (55%) abaixo de 20%
    6: 0.20,   # merge domina (95%) abaixo de 20%
    7: 0.20,   # merge domina (100%) abaixo de 20%
    8: 0.15,   # merge domina (100%) abaixo de 15%
}
_RHO_MERGE_DEFAULT = 0.15   # para nq_sub > 8


def select_method(nq_sub: int, rho_sub: float) -> str:
    """
    Retorna o nome do método QSP recomendado para um sub-estado com
    nq_sub qubits e densidade de amplitudes não-nulas rho_sub ∈ [0,1].

    Baseado nos padrões de dominância do benchmark Suite A:
      - nq_sub <= 3          → ucge_dc  (universal, overhead mínimo)
      - nq_sub >= 4, esparso → merge    (domina 55-100% se rho < limiar)
      - nq_sub >= 4, outros  → baa      (domina 80-100%)

    Parâmetros
    ----------
    nq_sub  : int    número de qubits do sub-estado
    rho_sub : float  densidade = nz / 2^nq_sub

    Retorna
    -------
    str  nome do método: 'ucge_dc', 'merge' ou 'baa'
    """
    if nq_sub <= 3:
        return 'ucge_dc'

    threshold = _RHO_MERGE_THRESHOLD.get(nq_sub, _RHO_MERGE_DEFAULT)
    if rho_sub < threshold:
        return 'merge'

    return 'baa'


# ===========================================================================
# Funções de inicialização
# =++========================================================================

def _init_merge(sv):
    return MergeInitialize(vector_to_binary_dict(sv)).definition


_METHOD_FNS = {
    'ucge_dc': initializeUCGEDC,
    'ucge':    initializeUCGE,
    'lrsp':    initializeLR,
    'baa':     initializeBAA,
    'svd':     initializeSVD,
    'pivot':   initializePivot,
    'merge':   _init_merge,
}


# ===========================================================================
# Circuito adaptativo completo
# ===========================================================================

def build_pnm_adaptive(state_vector, cached_result=None):
    """
    Decompõe o estado via PNM e aplica o método selecionado adaptativamente
    para cada sub-estado com base em (nq_sub, rho_sub).

    Parâmetros
    ----------
    state_vector   : vetor de estado (array numpy complexo)
    cached_result  : resultado de analyze_ry_rz_clusters (opcional).
                     Se None, é calculado internamente.
                     Pode conter 'substates_cache' pré-computado via
                     _compute_substates para evitar recomputação.

    Retorna
    -------
    QC             : QuantumCircuit  ou None se não separável
    bt             : float  tempo de construção em segundos (build time)
    methods_used   : dict   {group_id: método escolhido}

    Levanta ValueError se state_vector não for um vetor de estado válido.
    """
    start = time.time()

    if not is_valid_state_vector(state_vector):
        raise ValueError(
            f"vetor de estado inválido (tamanho={len(state_vector)}): "
            f"esperado potência de 2 e norma não-nula")

    state      = normalize_state(np.asarray(state_vector, dtype=complex))
    iNumQubits = len(state).bit_length() - 1

    # análise de separabilidade
    if cached_result is not None:
        result = cached_result
    else:
        result = analyze_ry_rz_clusters(state)

    if not result['can_decompose']:
        return None, time.time() - start, {}

    # pré-computa sub-estados se ainda não feito
    if 'substates_cache' not in result:
        result = _compute_substates(state, result)

    state_tree   = get_state_tree(state, iNumQubits)
    global_phase = state_tree.arg

    QC           = QuantumCircuit(iNumQubits)
    methods_used = {}

    for g in result['groups']:
        if not g['valid']:
            continue

        gid        = g['group_id']
        lvs        = sorted(g['levels_ry'])
        qiskit_lvs = [iNumQubits - 1 - lv for lv in reversed(lvs)]
        nq_sub     = len(lvs)

        # recupera sub-estado do cache
        cache_entry = result['substates_cache'].get(gid)
        if cache_entry is None:
            return None, time.time() - start, methods_used
        v_complex, _, _ = cache_entry
        if v_complex is None:
            return None, time.time() - start, methods_used

        # densidade do sub-estado
        nz_sub  = int(np.count_nonzero(np.abs(v_complex) > 1e-9))
        rho_sub = nz_sub / (2 ** nq_sub)

        # seleciona método
        if nq_sub == 1:
            method_name = 'ucge_dc'          # merge/baa não aceitam 1 qubit
        else:
            method_name = select_method(nq_sub, rho_sub)

        methods_used[gid] = method_name
        fn = _METHOD_FNS[method_name]

        try:
            qc_sub = fn(v_complex)
            # merge usa convenção reverse_qargs — converte para convenção direta
            if method_name == 'merge':
                qc_sub = qc_sub.reverse_bits()
            qc_sub = _fix_subgroup_phase(qc_sub, v_complex)
            QC.compose(qc_sub, qubits=qiskit_lvs, inplace=True)
        except Exception as e:
            print(f"  [adaptive] {method_name} falhou no grupo {gid} "
                  f"(nq={nq_sub}, rho={rho_sub:.3f}): {e}")
            # fallback para ucge_dc
            try:
                qc_sub = initializeUCGEDC(v_complex)
                qc_sub = _fix_subgroup_phase(qc_sub, v_complex)
                QC.compose(qc_sub, qubits=qiskit_lvs, inplace=True)
                methods_used[gid] = 'ucge_dc (fallback)'
            except Exception as e2:
                print(f"  [adaptive] fallback ucge_dc também falhou: {e2}")
                return None, time.time() - start, methods_used

    # correção de fase global residual
    QC.global_phase = global_phase
    QC = _fix_global_phase(QC, state, iNumQubits)

    return QC, time.time() - start, methods_used


# ===========================================================================
# Integração com o benchmark 
# ===========================================================================

def build_pnm_adaptive_compat(state_vector, cached_result=None):
    """
    retorna (QC, build_time).
    """
    qc, bt, _ = build_pnm_adaptive(state_vector, cached_result=cached_result)
    return qc, bt


# ===========================================================================
# Diagnóstico: mostra seleção por sub-estado
# ===========================================================================

def explain_selection(state_vector, cached_result=None):
    """
    Imprime qual método seria escolhido para cada sub-estado e por quê.
    Útil para depuração e análise.

    Levanta ValueError se state_vector não for um vetor de estado válido.
    """
    if not is_valid_state_vector(state_vector):
        raise ValueError(
            f"vetor de estado inválido (tamanho={len(state_vector)}): "
            f"esperado potência de 2 e norma não-nula")

    state = normalize_state(np.asarray(state_vector, dtype=complex))

    if cached_result is None:
        cached_result = analyze_ry_rz_clusters(state)

    if not cached_result['can_decompose']:
        print("Estado não separável — nenhuma decomposição.")
        return

    if 'substates_cache' not in cached_result:
        cached_result = _compute_substates(state, cached_result)

    iNumQubits = len(state).bit_length() - 1
    print(f"Estado: {iNumQubits}q | grupos: {cached_result['num_groups']}")
    print(f"{'Grupo':<6} {'nq':>4} {'nz':>6} {'rho':>7}  Método")
    print(f"{'─'*6} {'─'*4} {'─'*6} {'─'*7}  {'─'*14}")

    for g in cached_result['groups']:
        if not g['valid']:
            continue
        gid         = g['group_id']
        lvs         = sorted(g['levels_ry'])
        nq_sub      = len(lvs)
        cache_entry = cached_result['substates_cache'].get(gid)
        if cache_entry is None:
            continue
        v_complex, _, _ = cache_entry
        if v_complex is None:
            continue
        nz_sub      = int(np.count_nonzero(np.abs(v_complex) > 1e-9))
        rho_sub     = nz_sub / (2 ** nq_sub)
        method_name = 'ucge_dc' if nq_sub == 1 else select_method(nq_sub, rho_sub)
        threshold   = _RHO_MERGE_THRESHOLD.get(nq_sub, _RHO_MERGE_DEFAULT)
        reason      = (f"nq≤3" if nq_sub <= 3
                       else f"rho={rho_sub:.3f}<{threshold:.2f}" if method_name == 'merge'
                       else f"rho={rho_sub:.3f}≥{threshold:.2f}")
        print(f"{gid:<6} {nq_sub:>4} {nz_sub:>6} {rho_sub:>7.3f}  {method_name:<14} ({reason})")
