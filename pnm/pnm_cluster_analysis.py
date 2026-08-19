# -*- coding: utf-8 -*-
"""
pnm_cluster_analysis.py  (enxuto)
-----------------------------------------
Análise de separabilidade baseada exclusivamente na árvore Ry.

Princípio:
    A separabilidade é determinada APENAS pelos clusters Ry.
    Se find_clusters(graph_ry) retorna N > 1 raízes, o estado é
    candidato a ser separável em N sub-estados. Os grupos passam então
    por uma validação/refinamento via SVD direto no vetor de estado
    COMPLEXO (_svd_rank1): isso já captura qualquer quebra de
    separabilidade por fase, sem precisar de uma segunda árvore de
    ângulos (Rz) para comparar contra a Ry.

LIMPEZA (rodada de enxugamento do PNM):
  - Removidas _extract_substate_real e extract_group_substate_complex
    (construíam sub-circuito via graph_to_multicontrolled_circuit só
    para popular g['substates'], usado unicamente pelo relatório
    _print_report — cujo stdout é descartado no caminho de benchmark;
    o resultado nunca era lido por build_pnm_with_method). Isso já
    tinha eliminado a necessidade de quantum_circuit_builder.py no
    caminho de benchmark.
  - Removida a construção do grafo Rz por completo (graph_rz,
    _project_rz, _rz_nontrivial, rz_fully_negligible, e os campos
    'rz_subgraph'/'rz_nontrivial'/'rz_roots'/'levels_rz' em cada
    grupo). Motivo: a decisão de separabilidade (can_decompose) nunca
    dependeu de graph_rz — vem inteiramente de find_clusters(graph_ry)
    + _svd_rank1. O grafo Rz era resto de uma versão anterior do
    algoritmo (quando a separabilidade era decidida comparando as
    FORMAS dos grafos Ry e Rz), e nada em pnm_main.py (build_pnm_with_
    method, _compute_substates) ou nos scripts de benchmark lê qualquer
    campo derivado de Rz — confirmado por varredura em todos os
    arquivos do pipeline antes de remover. Era, além de código morto,
    a maior fatia do tempo de analyze_ry_rz_clusters (no caso de teste
    "8x8 denso", a fusão do grafo Rz sozinha custava mais que o resto
    da função inteira).

MERGE_FN: analyze_ry_rz_clusters aceita um parâmetro opcional merge_fn
(padrão: merge_equal_sibling_subtrees) para permitir comparar variantes
do algoritmo de fusão de subárvores (ex.: merge_equal_sibling_subtrees_
recursive) sem duplicar este pipeline inteiro — ver
compare_miss_variants.py.

VALIDAÇÃO: clStateVector foi removida (ver quantum_state_vector_utils.py
para a justificativa completa). analyze_ry_rz_clusters agora recebe o
vetor de estado bruto (array complexo), valida com is_valid_state_vector
e normaliza com normalize_state — levanta ValueError com mensagem clara
se o vetor for inválido (tamanho não é potência de 2, ou norma nula).
Não tenta "pular e continuar" sozinha — isso é decisão de quem estiver
num laço sobre múltiplos estados (ver os laços principais dos scripts
de benchmark, que verificam a validade ANTES de chamar esta função).
"""

import numpy as np

from angle_tree_core import (
    state_to_graph_with_dcs,
    find_clusters,
    get_subgraph,
)
from angle_tree_reduction import (
    prune_graph_dontcares_with_roots,
    merge_equal_sibling_subtrees,
)
from quantum_state_vector_utils import normalize_state, is_valid_state_vector


# ---------------------------------------------------------------------------
# Construção do grafo Ry
# ---------------------------------------------------------------------------

def _build_graph_ry(state, nqubits, merge_fn=merge_equal_sibling_subtrees):
    import io, sys
    old = sys.stdout; sys.stdout = io.StringIO()
    g, dc = state_to_graph_with_dcs(state, nqubits, 'y')
    g = prune_graph_dontcares_with_roots(g, dc)
    g = merge_fn(g)
    sys.stdout = old
    return g


# ---------------------------------------------------------------------------
# Auxiliares
# ---------------------------------------------------------------------------

def _levels_of(graph, root):
    return frozenset(
        d['level'] for _, d in get_subgraph(graph, root).nodes(data=True)
        if 'level' in d
    )


# ---------------------------------------------------------------------------
# Função principal
# ---------------------------------------------------------------------------

def analyze_ry_rz_clusters(state_vector, svd_tol=1e-9,
                            merge_fn=merge_equal_sibling_subtrees):
    """
    Separabilidade baseada exclusivamente nos clusters Ry + validação SVD
    no vetor complexo.

    merge_fn: função de fusão de subárvores irmãs a usar na construção
    do grafo Ry (padrão: merge_equal_sibling_subtrees). Passe
    merge_equal_sibling_subtrees_recursive para comparar a variante em
    fila de trabalho — ver compare_miss_variants.py.

    Levanta ValueError se state_vector não for um vetor de estado válido
    (tamanho não é potência de 2, ou norma nula).

    Retorna dict com:
        'can_decompose', 'num_groups', 'groups', 'graph_ry'
    """
    if not is_valid_state_vector(state_vector):
        raise ValueError(
            f"vetor de estado inválido (tamanho={len(state_vector)}): "
            f"esperado potência de 2 e norma não-nula")

    state   = normalize_state(np.asarray(state_vector, dtype=complex))
    nqubits = len(state).bit_length() - 1

    graph_ry = _build_graph_ry(state, nqubits, merge_fn=merge_fn)

    roots_ry      = find_clusters(graph_ry)
    can_decompose = len(roots_ry) > 1

    # ------------------------------------------------------------------
    # Validação e refinamento SVD dos grupos Ry.
    #
    # A árvore Ry só enxerga módulos das amplitudes, podendo detectar
    # mais grupos do que existem no vetor complexo (quando a fase quebra
    # a separabilidade de alguns grupos individualmente).
    #
    # Estratégia:
    #   1. Para cada grupo G detectado pelo Ry, testa SVD de G vs todos
    #      os outros qubits (não apenas outros grupos Ry).
    #   2. Grupos que passam (S[0] >= 1-tol) são mantidos como grupos
    #      independentes.
    #   3. Grupos que falham são mesclados em um único grupo residual.
    #   4. O grupo residual mesclado é validado uma última vez. Se ainda
    #      falhar (rank > 1), can_decompose = False.
    # ------------------------------------------------------------------
    SVD_TOL = 0.01

    def _svd_rank1(lvs_grp, vec, nq):
        """True se o grupo lvs_grp está separado do restante (rank-1)."""
        lvs_rest = [lv for lv in range(nq) if lv not in lvs_grp]
        if not lvs_rest:
            return True
        try:
            T = np.array(vec, dtype=complex).reshape([2] * nq)
            T = T.transpose(lvs_grp + lvs_rest)
            S = np.linalg.svd(
                T.reshape(2**len(lvs_grp), 2**len(lvs_rest)),
                compute_uv=False)
            return float(S[0]) >= 1.0 - SVD_TOL
        except Exception:
            return True  # em caso de falha numérica, não rejeita

    partitions_ry = [sorted(_levels_of(graph_ry, r)) for r in roots_ry]

    if can_decompose:
        passing  = []   # grupos que passam individualmente
        failing  = []   # grupos que falham — serão mesclados

        for grp in partitions_ry:
            if _svd_rank1(grp, state, nqubits):
                passing.append(grp)
            else:
                failing.append(grp)

        if failing:
            merged = sorted(lv for grp in failing for lv in grp)
            if len(passing) == 0 or not _svd_rank1(merged, state, nqubits):
                can_decompose = False
                partitions_ry = [sorted(lv for grp in partitions_ry for lv in grp)]
            else:
                partitions_ry = passing + [merged]

        can_decompose = len(partitions_ry) > 1

    validated_partitions = partitions_ry

    groups = []
    for lvs_ry in validated_partitions:
        orig_roots = [r for r in roots_ry
                      if any(lv in lvs_ry for lv in _levels_of(graph_ry, r))]
        groups.append({
            'ry_roots'  : orig_roots,
            'levels_ry' : set(lvs_ry),
            'valid'     : True,
        })

    for gid, g in enumerate(groups):
        g['group_id'] = gid

    _print_report(groups, can_decompose)

    return {
        'can_decompose' : can_decompose,
        'num_groups'    : len(groups),
        'groups'        : groups,
        'graph_ry'      : graph_ry,
    }


# ---------------------------------------------------------------------------
# Relatório
# ---------------------------------------------------------------------------

def _print_report(groups, can_decompose):
    sep = "=" * 70
    print(sep)
    print("  ANÁLISE DE CLUSTERS Ry  (separabilidade por Ry)")
    print(sep)
    print(f"  Clusters Ry encontrados: {len(groups)}")
    print()
    for g in groups:
        print(f"  Grupo {g['group_id']}  [OK]")
        print(f"    Raízes Ry    : {sorted(g['ry_roots'])}")
        print(f"    Níveis Ry    : {sorted(g['levels_ry'])}")
        print()
    print()
    if can_decompose:
        print("  >>> ESTADO PODE SER DECOMPOSTO em sub-estados independentes.")
        if any(len(g['ry_roots']) > 1 for g in groups):
            print("      (Grupos Ry mesclados após validação SVD complexo)")
    else:
        if len(groups) == 1 and len(groups[0]['ry_roots']) > 1:
            print("  >>> Grupos Ry detectados mas SVD complexo invalida separabilidade.")
        else:
            print("  >>> Estado NÃO separável (1 cluster Ry) — fluxo original.")
    print(sep)
