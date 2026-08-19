# -*- coding: utf-8 -*-
"""
angle_tree_core.py
-------------------
Núcleo mínimo de construção/consulta da árvore de ângulos, usado pelo
pipeline de benchmark do PNM (pnm_main.py, pnm_cluster_analysis.py).

Fusão enxuta dos dois arquivos originais de utilidades de árvore de ângulos
(versões originais). Mantém apenas as funções efetivamente referenciadas
pelo caminho canônico de benchmark (pnm_benchmark_suite_A_new.py):

  - get_state_tree            (fase global, via qclib.state_decomposition)
  - state_to_graph_with_dcs (construção do grafo de ângulos a partir do
                              vetor de estado)
  - find_clusters, get_subgraph, get_graph_depth, remove_subgraph,
    check_node_weights_equal   (usadas por angle_tree_reduction.py e por
                                 pnm_cluster_analysis.py)

REMOVIDO nesta limpeza (não referenciado no caminho de benchmark):
  create_angles_tree_with_dont_cares[_with_dont_cares_list], tree_to_
  graph_weighted_edges, display_graph, display_graph_yz, get_levels_graph,
  GraphDensityDistribution, remove_subtrees_zero. Isso também elimina as
  dependências de matplotlib e plotly, que só existiam por causa das
  funções display_*.

Se alguma dessas funções fizer falta depois (ex.: visualização de grafo
para debug), elas seguem disponíveis no histórico dos arquivos
originais, antes da limpeza.
"""

import math
import networkx as nx

from qclib.state_preparation.util.state_tree_preparation import (
    Amplitude,
    state_decomposition,
)

DC_VALUE = -9.0
TWO_PI = 2.0 * math.pi


# ---------------------------------------------------------------------------
# Árvore de estado (fase global) — via qclib
# ---------------------------------------------------------------------------

def get_state_tree(state, Num_qubits):
    data = [Amplitude(i, a) for i, a in enumerate(state)]
    return state_decomposition(Num_qubits, data)


# ---------------------------------------------------------------------------
# Consulta / manipulação de grafo
# ---------------------------------------------------------------------------

def get_graph_depth(graph, no_inicial):
    profundidade = 0
    fila = [(no_inicial, 0)]
    visitados = set()

    while fila:
        no_atual, nivel_atual = fila.pop(0)
        if no_atual not in visitados:
            visitados.add(no_atual)
            profundidade = max(profundidade, nivel_atual)
            for vizinho in graph[no_atual]:
                fila.append((vizinho, nivel_atual + 1))

    return profundidade


def get_subgraph(graph, root):
    raio = get_graph_depth(graph, root)
    return nx.ego_graph(graph, root, radius=raio)


def remove_subgraph(graph, subgraph):
    sorted_list_graph = list(graph.nodes())
    sorted_list_graph.sort()
    if subgraph in sorted_list_graph:
        raio = get_graph_depth(graph, subgraph)
        sub = list(nx.ego_graph(graph, subgraph, radius=raio))
        sub.sort(reverse=True)
        for node in sub:
            graph.remove_node(node)


def check_node_weights_equal(graph, subgraph1, subgraph2):
    """
    Verifica se os pesos dos nós em subgraph1 são iguais (dentro de
    tolerância) aos de subgraph2, nó a nó, na mesma ordem de índice.
    """
    Err = 1e-6

    if len(subgraph1.nodes()) != len(subgraph2.nodes()):
        return False

    list_subgraph1 = sorted(subgraph1.nodes)
    list_subgraph2 = sorted(subgraph2.nodes)

    node_map = {node: i for i, node in enumerate(list_subgraph1)}
    for node in list_subgraph1:
        other = list_subgraph2[node_map[node]]
        if 'weight' not in graph.nodes[node] or 'weight' not in graph.nodes[other]:
            return False
        if abs(graph.nodes[node]['weight'] - graph.nodes[other]['weight']) > Err:
            return False
        if graph.nodes[node]['level'] != graph.nodes[other]['level']:
            return False

    return True


def find_clusters(graph):
    """Retorna os nós-raiz (sem predecessor) — um por componente/cluster."""
    return [no for no in graph if not graph._pred[no]]


# ---------------------------------------------------------------------------
# Construção do grafo de ângulos a partir do vetor de estado
# ---------------------------------------------------------------------------

def _unwrap_phase(phi_new, phi_ref):
    """
    Retorna phi_new ajustado para ser o mais próximo de phi_ref,
    eliminando saltos de 2π. Garante continuidade entre fases vizinhas.
    """
    diff = phi_new - phi_ref
    diff = diff - TWO_PI * math.floor((diff + math.pi) / TWO_PI)
    return phi_ref + diff


def _build_state_tree(state, nqubits):
    """
    Constrói a árvore de estado bottom-up.

    Para cada par (left, right) ao subir:
      - arg_right_local = unwrap(arg_right, ref=arg_left)
        garante continuidade entre irmãos sem modificar arg_right.
      - arg_parent = (arg_left + arg_right_local) / 2
        média consistente, usada para calcular angle_z.
      - arg_right_local é guardado no nó pai para uso posterior.

    Retorna (all_levels, root).
    """
    import cmath

    level_nodes = []
    for i, a in enumerate(state):
        mag = abs(a)
        arg = cmath.phase(a) if mag > 1e-12 else 0.0
        level_nodes.append({
            "index": i,
            "level": nqubits,
            "mag":   mag,
            "arg":   arg,
            "left":  None,
            "right": None,
        })

    all_levels = {nqubits: level_nodes}
    level = nqubits
    while level > 0:
        nodes     = all_levels[level]
        new_nodes = []
        k = 0
        while k < len(nodes):
            left  = nodes[k]
            right = nodes[k + 1]
            mag   = math.sqrt(left["mag"]**2 + right["mag"]**2)

            arg_right_local = _unwrap_phase(right["arg"], left["arg"])
            arg             = (left["arg"] + arg_right_local) / 2.0

            new_nodes.append({
                "index":           left["index"] // 2,
                "level":           level - 1,
                "mag":             mag,
                "arg":             arg,
                "arg_right_local": arg_right_local,
                "left":            left,
                "right":           right,
            })
            k += 2
        all_levels[level - 1] = new_nodes
        level -= 1

    return all_levels, all_levels[0][0]


def state_to_graph_with_dcs(state, nqubits, axis='y'):
    """
    Constrói o grafo direcionado de ângulos Ry ou Rz a partir do vetor
    de estado.

    Retorna
    -------
    graph    : nx.DiGraph  com atributos 'weight' e 'level' por nó
    dc_roots : list        nós don't care que são raízes de sub-árvore dc
    """
    _, state_root = _build_state_tree(state, nqubits)

    graph         = nx.DiGraph()
    dc_candidates = []

    def process_node(st_node):
        lv  = st_node["level"]
        idx = st_node["index"]

        if st_node["left"] is None:
            return

        g_node_index = (2 ** lv) + idx

        right      = st_node["right"]
        mag_parent = st_node["mag"]
        arg_parent = st_node["arg"]

        arg_right_local = st_node.get("arg_right_local", right["arg"])
        arg_diff        = arg_right_local - arg_parent

        if mag_parent != 0.0:
            mag_ratio = right["mag"] / mag_parent
            angle_z   = 2.0 * arg_diff
        else:
            mag_ratio = DC_VALUE
            angle_z   = DC_VALUE

        if mag_ratio == DC_VALUE:
            angle_y = DC_VALUE
            angle_z = DC_VALUE
        elif mag_ratio < -1.0:
            angle_y = -math.pi
        elif mag_ratio > 1.0:
            angle_y =  math.pi
        else:
            angle_y = 2.0 * math.asin(mag_ratio)

        nodeangle = angle_y if axis == 'y' else angle_z

        graph.add_node(g_node_index, weight=nodeangle, level=lv)

        if lv > 0:
            parent_index = g_node_index // 2
            edge_weight  = int(((-1) ** (g_node_index + 1) + 1) / 2)
            if parent_index not in graph:
                graph.add_node(parent_index, weight=0.0, level=lv - 1)
            graph.add_edge(parent_index, g_node_index, weight=edge_weight)

        if nodeangle == DC_VALUE:
            dc_candidates.append(g_node_index)

        process_node(st_node["left"])
        process_node(st_node["right"])

    process_node(state_root)

    dc_roots = []
    for v in dc_candidates:
        is_leaf_node = (graph.out_degree(v) == 0)
        parents      = list(graph.predecessors(v))
        parent_is_dc = False
        if parents:
            parent_is_dc = (graph.nodes[parents[0]].get("weight") == DC_VALUE)
        is_dc_root = (not is_leaf_node) and parents and (not parent_is_dc)
        if is_leaf_node or is_dc_root:
            dc_roots.append(v)

    return graph, dc_roots
