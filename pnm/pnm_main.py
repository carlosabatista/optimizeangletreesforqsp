# -*- coding: utf-8 -*-
"""
pnm_main.py
----------------------------------------------------------------
Versão enxuta: mantém apenas o que os benchmarks de fato importam e
chamam:

  build_pnm_with_method, apply_bit_permutation_fast,
  generate_kron_state, _compute_substates

REMOVIDO nesta limpeza (confirmado, via inspeção do gerador canônico do
benchA, que nada disso é chamado no caminho de benchmark):

  - build_prune_n_merge_circuit, prune_and_merge_graph_with_dc_roots:
    o fluxo "circuito único" original nunca é chamado pelo benchmark; só
    existia no __main__ de demonstração do arquivo original.
  - AllQSPMethods, FinalTable, generate_n_kron_states, format_datetime,
    pick_single_method, normalize_subgraph_levels_min,
    get_sorted_unique_levels, remap_by_sorted_values, initializeUCG,
    initializeIso: usadas só pelas funções acima ou pelo __main__ de
    demonstração.
  - todo o import de quantum_state_preparation_helper_functions (7 funções
    — nenhuma chamada em lugar nenhum, e uma delas, SpaceTimeInitialize,
    está quebrada: referencia QubitEfficientQSP/braket_to_qiskit, que nem
    estão importados no arquivo original).
  - import de graph_to_multicontrolled_circuit / quantum_circuit_builder:
    não é mais necessário — build_pnm_with_method extrai sub-estados
    via _extract_substate_from_global (SVD direto no tensor), não via
    construção de circuito.
  - imports não usados do qclib: CvoqramInitialize, DcspInitialize,
    IsometryInitialize, UCGInitialize, BdspInitialize.

Se o fluxo de "circuito único" fizer falta de novo (ex.: para gerar uma
figura de circuito no artigo), ele segue disponível no arquivo original
pré-limpeza.
"""

import numpy as np
import time
import warnings

warnings.filterwarnings('ignore')

from functools import reduce
from typing import List, Sequence, TypeVar

from angle_tree_core import get_state_tree
from quantum_state_vector_utils import (
    vector_to_binary_dict, generate_random_state_n_m,
    normalize_state, is_valid_state_vector,
)
from pnm_cluster_analysis import analyze_ry_rz_clusters
from ucge_dc import UCGEDCInitialize

from qclib.state_preparation import (
    LowRankInitialize,
    SVDInitialize,
    PivotInitialize,
    MergeInitialize,
    BaaLowRankInitialize,
    UCGEInitialize,
)

from qiskit import QuantumCircuit
from qiskit.quantum_info import Statevector


def initializeUCGEDC(state_vector):
    return UCGEDCInitialize(state_vector).definition


# ===========================================================================
# Preparação de estado base (basis state) — usada por SVD/Pivot no caso
# degenerado de 1 amplitude não-nula
# ===========================================================================

def prepare_basis_state(k: int, n_qubits: int = None):
    if n_qubits is None:
        n_qubits = int(np.ceil(np.log2(k + 1))) if k > 0 else 1
    if k >= 2**n_qubits:
        raise ValueError(f"k={k} requer pelo menos {int(np.log2(k+1))} qubits")
    qc = QuantumCircuit(n_qubits)
    bin_k = bin(k)[2:].zfill(n_qubits)[::-1]
    for qubit, bit in enumerate(bin_k):
        if bit == '1':
            qc.x(qubit)
    qc.name = f"|{k}>"
    return qc


# ===========================================================================
# Wrappers de método (solo, usados dentro de build_pnm_with_method)
# ===========================================================================

def initializeLR(state_vector):
    opt_params = {"unitary_scheme": "ccd"}
    return LowRankInitialize(state_vector, opt_params=opt_params).definition


def initializeSVD(state_vector):
    sv = state_vector.copy()
    for i in range(len(sv)):
        if abs(sv[i]) < 1e-8:
            sv[i] = 0
    non_zero_counts = np.count_nonzero(sv)
    if non_zero_counts == 1:
        k = np.nonzero(sv)[0][0]
        num_qubits = len(state_vector).bit_length() - 1
        return prepare_basis_state(k, num_qubits)
    else:
        return SVDInitialize(state_vector).definition


def initializePivot(state_vector):
    sv = state_vector.copy()
    for i in range(len(sv)):
        if abs(sv[i]) < 1e-8:
            sv[i] = 0
    non_zero_counts = np.count_nonzero(sv)
    if non_zero_counts == 1:
        k = np.nonzero(sv)[0][0]
        num_qubits = len(state_vector).bit_length() - 1
        return prepare_basis_state(k, num_qubits)
    else:
        state_dict = vector_to_binary_dict(state_vector)
        return PivotInitialize(state_dict).definition


def initializeUCGE(state_vector): return UCGEInitialize(state_vector).definition
def initializeBAA(state_vector):  return BaaLowRankInitialize(state_vector).definition


# ===========================================================================
# Correção de fase global / de sub-circuito
# ===========================================================================

def _estimate_global_phase(v_target, v_prep):
    """
    Estima a fase global φ tal que e^{iφ}·|prep⟩ ≈ |target⟩.

    φ = angle(<v_target|v_prep>) = angle(Σ conj(v_target[i]) · v_prep[i])

    Estimador de máxima verossimilhança para fase global — robusto para
    qualquer estado, incluindo esparsos e complexos.
    """
    inner = np.dot(np.conj(v_target), v_prep)
    return float(np.angle(inner))


def _fix_global_phase(QC, state, nqubits):
    """Corrige a fase global do circuito QC comparando com o estado alvo."""
    try:
        sv_prep = Statevector.from_instruction(QC).data
        phi = _estimate_global_phase(
            np.array(state, dtype=complex) / np.linalg.norm(state),
            np.array(sv_prep, dtype=complex))
        QC.global_phase += phi
    except Exception:
        pass
    return QC


def _fix_subgroup_phase(qc_sub, v_target_sub):
    """
    Corrige a fase global de um sub-circuito comparando com o sub-estado
    alvo. Testa convenção direta e reverse_qargs, usa a melhor —
    compatível com ucge/ucge_dc (direta) e merge (reverse_qargs).
    """
    try:
        sv_d = Statevector.from_instruction(qc_sub).data
        sv_r = Statevector.from_instruction(qc_sub).reverse_qargs().data
        v_t  = np.array(v_target_sub, dtype=complex)
        v_t  = v_t / np.linalg.norm(v_t)
        fid_d = float(abs(np.dot(np.conj(v_t), sv_d))**2)
        fid_r = float(abs(np.dot(np.conj(v_t), sv_r))**2)
        sv_best = sv_d if fid_d >= fid_r else sv_r
        phi = _estimate_global_phase(v_t, np.array(sv_best, dtype=complex))
        qc_sub.global_phase += phi
    except Exception:
        pass
    return qc_sub


# ===========================================================================
# Extração de sub-estado a partir do vetor global (via SVD/tensor)
# ===========================================================================

def _extract_substate_from_global(state, lvs_group, n_total):
    """
    Extrai o sub-estado de um grupo diretamente do vetor de estado global.

    Estratégia: reorganiza o vetor por transposição de eixos para que os
    qubits do grupo fiquem contíguos, depois extrai o sub-estado por
    reshape + SVD.

    Os níveis da árvore Ry usam nível 0 = MSB (qubit mais significativo),
    o que corresponde ao eixo 0 do tensor após reshape.
    """
    tensor = np.array(state, dtype=complex).reshape([2] * n_total)

    lvs_other = [lv for lv in range(n_total) if lv not in lvs_group]
    tensor    = tensor.transpose(list(lvs_group) + lvs_other)

    nA  = len(lvs_group)
    nB  = n_total - nA
    mat = tensor.reshape(2**nA, 2**nB)

    U, S, Vh = np.linalg.svd(mat, full_matrices=False)
    return normalize_state(U[:, 0])


def _compute_substates(state_vector, result):
    """
    Pré-computa os sub-estados de todos os grupos e armazena em
    cached_result. Evita recomputar _extract_substate_from_global para
    cada método.

    Levanta ValueError se state_vector não for um vetor de estado válido.
    """
    if not is_valid_state_vector(state_vector):
        raise ValueError(
            f"vetor de estado inválido (tamanho={len(state_vector)}): "
            f"esperado potência de 2 e norma não-nula")

    state     = normalize_state(np.asarray(state_vector, dtype=complex))
    n_total   = len(state).bit_length() - 1
    substates = {}  # group_id -> (v_complex, lvs, qiskit_lvs)
    for g in result['groups']:
        if not g['valid']:
            continue
        gid       = g['group_id']
        lvs       = sorted(g['levels_ry'])
        qiskit_lvs = [n_total - 1 - lv for lv in reversed(lvs)]
        try:
            v = _extract_substate_from_global(state, lvs, n_total)
            substates[gid] = (v, lvs, qiskit_lvs)
        except Exception:
            substates[gid] = (None, lvs, qiskit_lvs)
    result['substates_cache'] = substates
    return result


# ===========================================================================
# Construção do circuito PNM decompondo por método, grupo a grupo
# ===========================================================================

def build_pnm_with_method(state_vector, method_name, cached_result=None):
    """
    Decompõe o estado via PNM e aplica 'method_name' em cada sub-estado
    complexo. Retorna (QC, construction_time) ou (None, None) se não
    separável.

    cached_result: resultado de analyze_ry_rz_clusters, opcionalmente com
    'substates_cache' pré-computado via _compute_substates.

    Levanta ValueError se state_vector não for um vetor de estado válido.
    """
    if not is_valid_state_vector(state_vector):
        raise ValueError(
            f"vetor de estado inválido (tamanho={len(state_vector)}): "
            f"esperado potência de 2 e norma não-nula")

    state = normalize_state(np.asarray(state_vector, dtype=complex))

    start = time.time()

    if cached_result is not None:
        result = cached_result
    else:
        result = analyze_ry_rz_clusters(state)

    if not result['can_decompose']:
        return None, None

    iNumQubits   = len(state).bit_length() - 1
    state_tree   = get_state_tree(state, iNumQubits)
    global_phase = state_tree.arg

    QC = QuantumCircuit(iNumQubits)

    method_fns = {
        'lrsp':    initializeLR,
        'baa':     initializeBAA,
        'ucge':    initializeUCGE,
        'ucge_dc': initializeUCGEDC,
        'svd':     initializeSVD,
        'pivot':   initializePivot,
        'merge':   lambda sv: MergeInitialize(vector_to_binary_dict(sv)).definition,
    }
    fn = method_fns.get(method_name)
    if fn is None:
        return None, None

    substates_cache = result.get('substates_cache', {})

    for g in result['groups']:
        if not g['valid']:
            continue

        gid = g['group_id']

        if gid in substates_cache:
            v_complex, lvs, qiskit_lvs = substates_cache[gid]
        else:
            lvs        = sorted(g['levels_ry'])
            qiskit_lvs = [iNumQubits - 1 - lv for lv in reversed(lvs)]
            try:
                v_complex = _extract_substate_from_global(state, lvs, iNumQubits)
            except Exception as e:
                print(f"  [aviso] extração sub-estado grupo {gid}: {e}")
                return None, None

        if v_complex is None:
            return None, None

        try:
            # sub-estados de 1 qubit: usa ucge
            if len(v_complex) == 2:
                fn_use = initializeUCGE
            else:
                fn_use = fn
            qc_sub = fn_use(v_complex)
            # merge usa convenção reverse_qargs — converte para convenção direta
            if method_name == 'merge':
                qc_sub = qc_sub.reverse_bits()
            qc_sub = _fix_subgroup_phase(qc_sub, v_complex)
            QC.compose(qc_sub, qubits=qiskit_lvs, inplace=True)
        except Exception as e:
            print(f"  [aviso] {method_name} falhou no grupo {g['group_id']}: {e}")
            return None, None

    QC.global_phase = global_phase
    QC = _fix_global_phase(QC, state, iNumQubits)

    return QC, time.time() - start


# ===========================================================================
# Geração de estados produto (para o benchmark)
# ===========================================================================

def generate_kron_state(list_nq, list_nz, seed, real):
    """Gera estado produto com controle de real/complexo."""
    subs = [
        generate_random_state_n_m(nq, nz, seed=seed + i, bRealState=real)
        for i, (nq, nz) in enumerate(zip(list_nq, list_nz))
    ]
    return normalize_state(reduce(np.kron, subs))


T = TypeVar('T')

def apply_bit_permutation_fast(state: Sequence[T], perm: List[int]) -> List[T]:
    n    = len(state)
    logn = n.bit_length() - 1
    new_idx_from_old = [0] * n
    for i in range(n):
        new_idx = 0
        for new_pos, old_pos in enumerate(perm):
            if i & (1 << old_pos):
                new_idx |= (1 << new_pos)
        new_idx_from_old[i] = new_idx
    return [state[i] for i in new_idx_from_old]
