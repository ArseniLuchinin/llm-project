"""Сообщества графа и их сводки. Требование FR-3."""

import networkx as nx
import networkx.algorithms.community as nx_comm

from assistant.infra.llm import LLMClient, LLMResponse, LLMError
from assistant.infra.schemas import Community

from collections import defaultdict


def detect_communities(graph: nx.DiGraph) -> list[Community]:
    """Разбивает узлы на сообщества алгоритмом Лувена с фиксированным `seed`.

    Каждый узел входит ровно в одно сообщество. Поле `summary` здесь остаётся пустым.
    """

    seed = 42
    communities_raw = nx_comm.louvain_communities(graph, seed=seed)
    return [Community(community_id=i, nodes=list(raw)) for i, raw in enumerate(communities_raw)]


def summarize_community(community: Community, graph: nx.DiGraph, llm: LLMClient) -> str:
    """Сводка сообщества не длиннее 150 слов: какие сущности входят, как связаны, что встречается чаще.

    Все числа считает код по весам рёбер и передаёт модели готовыми. Модель их только формулирует.
    """

    SYSTEM_PROMPT = """
    Ты составляешь краткую сводку сообщества в графе на основе уже посчитанных фактов.

    Задача:
    написать связную, нейтральную, информативную сводку сообщества длиной не более 150 слов.
    
    Что нужно отразить в сводке:
    1. Какие сущности входят в сообщество.
    2. Какие типы сущностей преобладают.
    3. Как сущности связаны между собой внутри сообщества.
    4. Какие типы связей встречаются чаще всего.
    5. Какие узлы являются наиболее заметными или центральными, если такие переданы во входных данных.
    
    Правила:
    - Используй только факты и числа из входных данных.
    - Ничего не выдумывай и не делай предположений, которых нет во входе.
    - Не пересчитывай числа и не интерпретируй их как приблизительные.
    - Если каких-то данных нет, просто не упоминай их.
    - Пиши коротко, ясно и без воды.
    - Не используй маркированный список.
    - Не используй вводные фразы вроде "Вот сводка" или "Данное сообщество".
    - Тон нейтральный, аналитический.
    - Если во входных данных есть названия конкретных сущностей, упоминай наиболее важные из них.
    - Если переданы только типы, описывай сообщество через типы, не придумывая имена.
    - Итог должен быть одним абзацем.
    """

    def get_sorted_text(stat: dict) -> str:
        return "\n".join([f"- {pair[0]}:{pair[1]}" for pair in sorted(stat.items(), key=lambda x: x[1], reverse=True)])

    internal_nodes = { node_id: 0 for node_id in community.nodes }
    stats = {
        "num_nodes": len(community.nodes),
        "num_internal_edges": 0,
        "entity_types": defaultdict(int),
        "relation_types": defaultdict(int),
        "top_heavy_edges": {},
        "top_by_internal_in_degree": defaultdict(int),
        "top_by_internal_out_degree": defaultdict(int)
    }

    for node_id in community.nodes:
        meta = graph.nodes[node_id]
        stats["entity_types"][meta["type"]] += 1
        node_internal_degree = 0

        for u, v in graph.edges(node_id):
            if not v in internal_nodes:
                continue

            e = graph.edges[u, v]
            stats["num_internal_edges"] += 1
            stats["relation_types"][e["type"]] += 1
            stats["top_heavy_edges"][f"{u}--{e['type']}->{v}"] = e['weight']
            stats["top_by_internal_in_degree"][v] += 1
            node_internal_degree += 1

        stats["top_by_internal_out_degree"][node_id] += node_internal_degree

    USER_PROMPT = f"""
Составь краткую сводку сообщества по следующим данным.

ID сообщества: {community.community_id}

Размер сообщества:
- число узлов: {stats["num_nodes"]}
- число внутренних направленных рёбер: {stats["num_internal_edges"]}

Распределение типов сущностей:
{get_sorted_text(stats["entity_types"])}

Распределение типов связей:
{get_sorted_text(stats["relation_types"])}

Самые тяжелые связи:
{get_sorted_text(stats["top_heavy_edges"])}

Узлы с наибольшим количеством внутренних входящих связей:
{get_sorted_text(stats["top_by_internal_in_degree"])}

Узлы с наибольшим количеством внутренних исходящих связей:
{get_sorted_text(stats["top_by_internal_out_degree"])}

Требования к ответу:
- не более 150 слов (ВАЖНО!);
- один абзац;
- упомяни только то, что следует из этих данных.
"""

    MAX_RESP_LEN = 150

    try:
        resp: LLMResponse = llm.chat(
            messages=[
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user", "content": USER_PROMPT}
            ]
        )
        return resp.content[:MAX_RESP_LEN]
    except LLMError as err:
        print(f"LLM Client error {err}")
        return ""
