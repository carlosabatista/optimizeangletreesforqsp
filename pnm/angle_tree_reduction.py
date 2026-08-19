# -*- coding: utf-8 -*-
"""
angle_tree_reduction.py
------------------------
Versão enxuta de angle_tree_reduction_methods.py: mantém apenas as duas
rotinas usadas pelo caminho canônico de benchmark (via
pnm_cluster_analysis._build_graph):

  - merge_equal_sibling_subtrees      (fusão de subárvores irmãs idênticas)
  - prune_graph_dontcares_with_roots  (poda de don't cares, usando a lista
                                        de raízes dc já calculada por
                                        state_to_graph_with_dcs_v3)

REMOVIDO nesta limpeza (não referenciado no caminho de benchmark):
  remove_first_zero_node, prune_graph_dontcares (versão sem roots,
  substituída por prune_graph_dontcares_with_roots), dontcare_condition,
  remove_duplicates.

merge_equal_sibling_subtrees_recursive foi reincorporada nesta rodada:
mesmo critério de fusão e mesma semântica de aplicação de
merge_subgraphs (inclusive o comportamento de manter o nó pai pendurado
em gp sem remover a aresta gp->parent — comportamento intencional, não
é bug), mas sem reiniciar a varredura completa do grafo a cada fusão
individual. Já foi validada (mesmos clusters de registradores que a
versão original, em todos os estados reais testados) antes de ser
temporariamente deixada de fora da limpeza; ver compare_miss_variants.py
para a comparação ponta a ponta neste conjunto de arquivos.
"""

import gzip  # ← PREVENIR o bug de networkx
import bz2   # ← PREVENIR o bug de networkx
import time
import networkx as nx

from collections import defaultdict

from angle_tree_core import (
    get_graph_depth,
    check_node_weights_equal,
    remove_subgraph,
)


def merge_subgraphs(graph):
    graph = graph.copy()
    if len(graph) <= 1:
        return graph, False

    changed = False
    parents = [n for n in sorted(graph) if graph.out_degree(n) >= 2]

    for parent in parents:
        if parent not in graph: continue

        children = sorted(graph.successors(parent))
        if len(children) < 2: continue

        groups = defaultdict(list)
        for child in children:
            rounded_weight = round(graph.nodes[child]['weight'], 6)
            key = (graph.nodes[child]['level'], rounded_weight)
            groups[key].append(child)

        for group in groups.values():
            if len(group) < 2: continue

            for i in range(len(group)-1):
                no_1, no_2 = group[i], group[i+1]

                raio1 = get_graph_depth(graph, no_1)
                raio2 = get_graph_depth(graph, no_2)

                if raio1 != raio2: continue

                sub_1 = nx.ego_graph(graph, no_1, radius=raio1)
                sub_2 = nx.ego_graph(graph, no_2, radius=raio2)

                if check_node_weights_equal(graph, sub_1, sub_2):
                    grandparent = list(graph.predecessors(parent))
                    if grandparent:
                        gp = grandparent[0]
                        weight = graph[gp][parent].get('weight', 1.0)
                        graph.add_edge(gp, no_1, weight=weight)

                    graph.remove_edge(parent, no_1)
                    remove_subgraph(graph, no_2)
                    changed = True
                    break
            if changed: break

    return graph, changed


def merge_equal_sibling_subtrees(graph):
    graphs_merged = True
    while graphs_merged:
        graph, graphs_merged = merge_subgraphs(graph)
    return graph


def _find_merge_pair_in_parent(graph, parent):
    """
    Reproduz EXATAMENTE a lógica de busca de par fundível usada dentro do
    laço de merge_subgraphs (mesmo agrupamento por (level, weight
    arredondado), mesmo critério de igualdade estrutural via
    check_node_weights_equal). Isolada aqui apenas para ser reaproveitada
    pela varredura em fila de trabalho (worklist), sem alterar
    merge_subgraphs em si.
    """
    if parent not in graph:
        return None

    children = sorted(graph.successors(parent))
    if len(children) < 2:
        return None

    groups = defaultdict(list)
    for child in children:
        rounded_weight = round(graph.nodes[child]['weight'], 6)
        key = (graph.nodes[child]['level'], rounded_weight)
        groups[key].append(child)

    for group in groups.values():
        if len(group) < 2:
            continue
        for i in range(len(group) - 1):
            no_1, no_2 = group[i], group[i + 1]

            raio1 = get_graph_depth(graph, no_1)
            raio2 = get_graph_depth(graph, no_2)
            if raio1 != raio2:
                continue

            sub_1 = nx.ego_graph(graph, no_1, radius=raio1)
            sub_2 = nx.ego_graph(graph, no_2, radius=raio2)

            if check_node_weights_equal(graph, sub_1, sub_2):
                return no_1, no_2

    return None


def _apply_merge(graph, parent, no_1, no_2):
    """
    Executa a MESMA operação de promoção/remoção de merge_subgraphs
    (inclusive o mesmo comportamento de manter `parent` pendurado em `gp`
    sem remover a aresta gp->parent -- comportamento intencional,
    confirmado, não é bug). Retorna gp (ou None se parent já era raiz).
    """
    grandparent = list(graph.predecessors(parent))
    gp = grandparent[0] if grandparent else None
    if gp is not None:
        weight = graph[gp][parent].get('weight', 1.0)
        graph.add_edge(gp, no_1, weight=weight)

    graph.remove_edge(parent, no_1)
    remove_subgraph(graph, no_2)
    return gp


def merge_equal_sibling_subtrees_recursive(graph):
    """
    Versão em pós-ordem (DFS até ponto fixo local) de
    merge_equal_sibling_subtrees: mesmo critério de fusão e mesma
    semântica de aplicação (via _find_merge_pair_in_parent / _apply_merge,
    espelhando merge_subgraphs linha a linha), mas SEM reiniciar a
    varredura completa do grafo a cada fusão individual.

    Ideia (proposta e traçada à mão pelo usuário sobre o exemplo
    A[B[D[E,F],D[E,F]], C[D[E,F],D[E,F]]]):
      - Para cada nó, resolve PRIMEIRO cada filho por completo
        (recursão) — só depois de todo filho estar em forma final é que
        os filhos do próprio nó são comparados entre si.
      - Enquanto houver par fundível entre os filhos (já resolvidos) do
        nó atual, funde e repete a checagem no mesmo nó (cobre grupos
        com >2 membros e o efeito cascata: ex. D[E,F] promovido de B e
        de C vira dois filhos novos e iguais em A, fundem de novo, viram
        raiz independente).

    Por que isso é suficiente (e por que corrige o bug da versão anterior
    sem reintroduzir o custo do reinício completo): check_node_weights_
    equal compara a SUBÁRVORE INTEIRA de cada filho. Resolvendo cada
    filho até ponto fixo ANTES de comparar os filhos entre si, a
    comparação em qualquer nó já opera sobre formas finais e estáveis —
    nunca precisa ser refeita depois, porque nunca é feita cedo demais.
    Cada nó é visitado exatamente uma vez (um nó que perdeu todos os
    filhos não pode ganhar filho novo depois, então nunca precisa ser
    revisitado), o que evita a rechecagem redundante que fazia a versão
    anterior (correção via subida de cadeia de ancestrais a cada fusão)
    explodir para minutos em casos densos permutados grandes (ex.: 8x8
    denso permutado, 144s -> mais de 1h com a versão anterior).

    Limitação compartilhada com merge_equal_sibling_subtrees (não é
    regressão nova): raízes originalmente distintas de uma floresta (sem
    um pai em comum) nunca são comparadas entre si, pois a fusão só
    acontece entre filhos de um mesmo nó.
    """
    graph = graph.copy()
    if len(graph) <= 1:
        return graph

    def resolve(node):
        if node not in graph:
            return
        for child in sorted(graph.successors(node)):
            resolve(child)

        changed = True
        while changed:
            changed = False
            pair = _find_merge_pair_in_parent(graph, node)
            if pair is not None:
                no_1, no_2 = pair
                _apply_merge(graph, node, no_1, no_2)
                changed = True

    roots = sorted(n for n in graph if not list(graph.predecessors(n)))
    for r in roots:
        resolve(r)

    return graph


def prune_graph_dontcares_with_roots(graph, dc_roots):
    """
    Poda don't cares usando diretamente a lista dc_roots (nós dc que são
    raízes de subárvore ou folhas), evitando a busca completa por
    weight == -9.
    """
    valor = -9
    graphy = graph.copy()

    list_dontcares = sorted(dc_roots)
    dc_visitados = []

    if len(list_dontcares) > 0:
        start_time = time.time()

        for no_dc in list_dontcares:
            if no_dc not in graphy:
                continue
            if graphy.nodes[no_dc].get('weight') != valor:
                continue
            if no_dc in dc_visitados:
                continue

            raio = get_graph_depth(graphy, no_dc)
            subgrafo_dc_a_eliminar_Y = nx.ego_graph(graphy, no_dc, radius=raio)

            parent_dc = graphy._pred.get(no_dc, {})
            if parent_dc:
                parent = list(parent_dc)[0]

                children = list(graphy.successors(parent))
                if no_dc in children:
                    children.remove(no_dc)

                if len(children) == 0:
                    remove_subgraph(graphy, no_dc)
                    for n in list(subgrafo_dc_a_eliminar_Y):
                        dc_visitados.append(n)
                    continue

                brother = children[0]

                grand_parent = graphy._pred.get(parent, {})

                remove_subgraph(graphy, no_dc)
                for n in list(subgrafo_dc_a_eliminar_Y):
                    dc_visitados.append(n)

                if grand_parent:
                    gp = list(grand_parent)[0]
                    if gp in graphy and parent in graphy and graphy.has_edge(gp, parent):
                        w = graphy[gp][parent]['weight']
                        graphy.add_edge(gp, brother, weight=w)

                if parent in graphy and brother in graphy and graphy.has_edge(parent, brother):
                    graphy.remove_edge(parent, brother)

        dc_time = time.time() - start_time
        print("="*80)
        print(f"don't cares Identification+prune time (roots only): {dc_time} ")
        print("="*80)

    return graphy
