"""Два режима поиска по графу. Требования FR-4, FR-5 и FR-10."""

import json
from itertools import combinations

import networkx as nx

from assistant.infra.grounding import code_type, find_codes, find_names
from assistant.infra.llm import LLMClient
from assistant.infra.schemas import MAP_SCHEMA, Answer, KnowledgeBase


def local_search(question: str, kb: KnowledgeBase, llm: LLMClient) -> Answer:
    """Отвечает на вопрос о конкретных сущностях по фактам из графа.

    Опорные сущности ищутся без модели: коды через `find_codes`, названия через `find_names`.
    При двух и более опорных сущностях берутся факты на путях длиной до трёх связей между ними
    и цепочки `REPLACED_BY` от деталей на этих путях. При одной берётся окрестность на глубину
    до двух связей. Модель получает пронумерованный список фактов и отвечает только по нему.
    Если в вопросе есть код, которого нет в графе, или фактов нет, возвращается
    `Answer(..., refused=True)` без вызова модели.
    """
    graph = kb.graph
    codes = find_codes(question)
    if any(f"{code_type(code)}:{code}" not in graph for code in codes):
        return Answer("в базе знаний не найдено", refused=True)

    anchors = [f"{code_type(code)}:{code}" for code in codes]
    anchors.extend(f"{kind}:{name}" for kind, name in find_names(question, kb.vocabulary))
    anchors = list(dict.fromkeys(node for node in anchors if node in graph))
    if not anchors:
        return Answer("в базе знаний не найдено", refused=True)

    undirected = graph.to_undirected()
    selected_edges: set[tuple[str, str]] = set()
    path_nodes: set[str] = set()
    if len(anchors) == 1:
        neighbourhood = set(nx.single_source_shortest_path_length(undirected, anchors[0], cutoff=2))
        path_nodes.update(neighbourhood)
        selected_edges.update(
            (source, target)
            for source, target in graph.edges
            if source in neighbourhood and target in neighbourhood
        )
    else:
        for start, end in combinations(anchors, 2):
            for path in nx.all_simple_paths(undirected, start, end, cutoff=3):
                path_nodes.update(path)
                for source, target in zip(path, path[1:], strict=False):
                    if graph.has_edge(source, target):
                        selected_edges.add((source, target))
                    if graph.has_edge(target, source):
                        selected_edges.add((target, source))

        pending_parts = sorted(node for node in path_nodes if graph.nodes[node].get("type") == "Part")
        visited_parts: set[str] = set()
        while pending_parts:
            part = pending_parts.pop(0)
            if part in visited_parts:
                continue
            visited_parts.add(part)
            for replacement in sorted(graph.successors(part)):
                if graph.edges[part, replacement].get("type") != "REPLACED_BY":
                    continue
                selected_edges.add((part, replacement))
                if graph.nodes[replacement].get("type") == "Part" and replacement not in visited_parts:
                    pending_parts.append(replacement)

    edges = sorted(
        selected_edges,
        key=lambda edge: (edge[0], graph.edges[edge[0], edge[1]].get("type", ""), edge[1]),
    )
    facts: list[str] = []
    numbered_facts: list[str] = []
    sources: list[str] = []
    for source, target in edges:
        relation = graph.edges[source, target]
        fact = (
            f"{graph.nodes[source].get('name', source)} "
            f"-{relation.get('type', 'RELATED_TO')}-> {graph.nodes[target].get('name', target)}"
        )
        numbered_fact = f"{len(facts) + 1}. {fact}"
        if len("\n".join([*numbered_facts, numbered_fact])) > 8000:
            continue
        facts.append(fact)
        numbered_facts.append(numbered_fact)
        for chunk_id in relation.get("sources", []):
            document = chunk_id.split("#", 1)[0]
            if document not in sources:
                sources.append(document)

    if not facts:
        return Answer("в базе знаний не найдено", refused=True)

    reply = llm.chat(
        [
            {
                "role": "system",
                "content": (
                    "Ты ассистент сервисной службы. Отвечай только по приведённым фактам и не добавляй "
                    "сведения, которых в них нет. Если данных недостаточно, ответь: «в базе знаний не найдено». "
                    "Если факты содержат цепочку REPLACED_BY, назови последнюю деталь цепочки как актуальную."
                ),
            },
            {
                "role": "user",
                "content": f"Факты:\n{chr(10).join(numbered_facts)}\n\nВопрос: {question}",
            },
        ]
    )
    return Answer(
        reply.content,
        sources=sources,
        facts=facts,
        refused="в базе знаний не найдено" in reply.content.lower(),
    )


def global_search(question: str, kb: KnowledgeBase, llm: LLMClient) -> Answer:
    """Отвечает на обзорный вопрос по сводкам сообществ в два шага.

    Шаг map: для каждой сводки модель даёт частичный ответ и оценку полезности от 0 до 10.
    Шаг reduce: частичные ответы с оценкой выше 0 объединяются в итоговый ответ.
    Если таких ответов нет, возвращается `Answer(..., refused=True)`.
    """
    partial_answers: list[str] = []
    sources: list[str] = []
    graph = kb.graph
    for community in kb.communities:
        response = llm.chat(
            [
                {
                    "role": "system",
                    "content": (
                        "Ответь на вопрос только по сводке сообщества. Если сводка не помогает ответить, "
                        "верни пустой ответ и score 0. Иначе дай краткий частичный ответ и оцени его полезность "
                        "целым числом от 1 до 10."
                    ),
                },
                {
                    "role": "user",
                    "content": f"Вопрос: {question}\n\nСводка сообщества:\n{community.summary}",
                },
            ],
            json_schema=MAP_SCHEMA,
        )
        try:
            partial = json.loads(response.content)
        except json.JSONDecodeError:
            continue
        if not isinstance(partial, dict):
            continue
        answer = partial.get("answer")
        score = partial.get("score")
        if (
            not isinstance(answer, str)
            or not answer.strip()
            or not isinstance(score, int)
            or isinstance(score, bool)
            or not 0 <= score <= 10
            or score == 0
        ):
            continue

        partial_answers.append(answer.strip())
        community_nodes = set(community.nodes)
        for source, target, relation in graph.edges(data=True):
            if source not in community_nodes or target not in community_nodes:
                continue
            for chunk_id in relation.get("sources", []):
                document = chunk_id.split("#", 1)[0]
                if document not in sources:
                    sources.append(document)

    if not partial_answers:
        return Answer("в базе знаний не найдено", refused=True)

    reply = llm.chat(
        [
            {
                "role": "system",
                "content": (
                    "Объедини частичные ответы в один точный ответ на вопрос. Используй только сведения из "
                    "частичных ответов. Если они не дают ответа, напиши: «в базе знаний не найдено»."
                ),
            },
            {
                "role": "user",
                "content": (
                    f"Вопрос: {question}\n\nПолезные частичные ответы:\n"
                    + "\n".join(f"{index}. {answer}" for index, answer in enumerate(partial_answers, start=1))
                ),
            },
        ]
    )
    return Answer(
        reply.content,
        sources=sources,
        facts=partial_answers,
        refused="в базе знаний не найдено" in reply.content.lower(),
    )
