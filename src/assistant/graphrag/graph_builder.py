"""Сборка графа знаний из результатов извлечения. Требование FR-2."""

import networkx as nx

from assistant.infra.schemas import ExtractionResult
from collections import defaultdict


def build_graph(results: list[ExtractionResult]) -> nx.DiGraph:
    """Объединяет результаты по всем чанкам в один направленный граф.

    Узел: идентификатор `Entity.node_id`, атрибуты `type`, `name`, `sources` (список chunk_id).
    Ребро: атрибуты `type`, `weight` (число разных чанков, где связь найдена), `sources`.
    Связи, не соответствующие схеме (`Relation.matches_schema`), отбрасываются. Если между двумя
    деталями есть `REPLACED_BY` в обе стороны, остаётся связь с большим весом, при равенстве
    отбрасываются обе. Число таких конфликтов записывается в `graph.graph["conflicts"]`.
    """

    g = nx.DiGraph()
    g.graph["conflicts"] = 0
    nodes = {}
    node_sources = defaultdict(set)
    edge_sources = defaultdict(set)

    for result in results:
        curr_chunk_id = result.chunk_id

        for e in result.entities:
            nodes[e.node_id] = (e.type, e.name)
            node_sources[e.node_id].add(curr_chunk_id)

        for r in result.relations:
            if not r.matches_schema():
                continue

            subj, obj = r.subject, r.object
            nodes.setdefault(subj.node_id, (subj.type, subj.name))
            nodes.setdefault(obj.node_id, (obj.type, obj.name))
            node_sources[subj.node_id].add(curr_chunk_id)
            node_sources[obj.node_id].add(curr_chunk_id)
            edge_sources[(subj.node_id, obj.node_id, r.type)].add(curr_chunk_id)

    for node_id, (n_type, n_name) in nodes.items():
        g.add_node(node_id, type=n_type, name=n_name, sources=node_sources[node_id])

    for (subj_id, obj_id, r_type), srcs in edge_sources.items():
        reverse_key = (obj_id, subj_id, r_type)
        w = len(srcs)

        if r_type == "REPLACED_BY" and reverse_key in edge_sources:
            if subj_id < obj_id:
                g.graph["conflicts"] += 1

            reverse_w = len(edge_sources[reverse_key])
            if w <= reverse_w:
                continue

        g.add_edge(subj_id, obj_id, type=r_type, weight=w, sources=sorted(srcs))

    return g
