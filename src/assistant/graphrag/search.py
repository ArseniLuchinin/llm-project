"""Два режима поиска по графу. Требования FR-4, FR-5 и FR-10."""

import json
from itertools import combinations

import networkx as nx

from assistant.infra.grounding import code_type, find_codes, find_names
from assistant.infra.llm import LLMClient
from assistant.infra.schemas import MAP_SCHEMA, Answer, KnowledgeBase

MAX_REPLACEMENT_DEPTH = 5
MAX_FACTS_CHARS = 8000
REFUSAL_TEXT = "в базе знаний не найдено"


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

    pending_parts = [(node, 0) for node in sorted(path_nodes) if graph.nodes[node].get("type") == "Part"]
    visited_parts: set[str] = set()
    while pending_parts:
        part, depth = pending_parts.pop(0)
        if part in visited_parts:
            continue
        visited_parts.add(part)

        for source, target in (*graph.in_edges(part), *graph.out_edges(part)):
            if graph.edges[source, target].get("type") != "REPLACED_BY":
                selected_edges.add((source, target))

        if depth >= MAX_REPLACEMENT_DEPTH:
            continue
        for replacement in sorted(graph.successors(part)):
            if graph.edges[part, replacement].get("type") != "REPLACED_BY":
                continue
            selected_edges.add((part, replacement))
            if graph.nodes[replacement].get("type") == "Part" and replacement not in visited_parts:
                pending_parts.append((replacement, depth + 1))

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
        if len("\n".join([*numbered_facts, numbered_fact])) > MAX_FACTS_CHARS:
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
                    "Ты ассистент сервисной службы. Сначала проверь, дают ли приведённые факты ответ на вопрос. "
                    "Если ответ следует из фактов, обязательно ответь прямо, не отказывайся излишне осторожно. "
                    "Направленные связи FIXED_BY, SUPPLIED_BY, BELONGS_TO и REPLACED_BY являются достаточными "
                    "доказательствами для соответствующих вопросов; используй тип и направление связи, даже если "
                    "ответ не повторён дословно. Используй также остальные свойства связанных узлов. "
                    "Если есть REPLACED_BY, проследи цепочку и назови последнюю деталь. "
                    f"Отказывайся и пиши «{REFUSAL_TEXT}» только если ответ действительно нельзя определить "
                    "из фактов. Не добавляй отсутствующих в фактах сведений."
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


def _global_graph_facts(question: str, kb: KnowledgeBase) -> list[str]:
    """Готовит точные расчёты по имеющимся рёбрам, не достраивая отсутствующие связи."""
    graph = kb.graph
    facts: list[str] = []
    selected_lines = []
    for kind, name in find_names(question, kb.vocabulary):
        line = f"ProductLine:{name}"
        if kind == "ProductLine" and line in graph:
            selected_lines.append(line)
    if not selected_lines:
        selected_lines = sorted(
            node for node, attrs in graph.nodes(data=True) if attrs.get("type") == "ProductLine"
        )

    for line in selected_lines:
        line_name = graph.nodes[line].get("name", line)
        models = {
            source
            for source, _, relation in graph.in_edges(line, data=True)
            if relation.get("type") == "BELONGS_TO"
        }
        failures_by_component: dict[str, int] = {}
        failures_by_model: dict[str, int] = {}
        parts: set[str] = set()
        for model in models:
            for _, target, relation in graph.out_edges(model, data=True):
                if relation.get("type") == "FAILED":
                    weight = relation.get("weight", 1)
                    failures_by_component[target] = failures_by_component.get(target, 0) + weight
                    failures_by_model[model] = failures_by_model.get(model, 0) + weight
                elif relation.get("type") == "HAS_PART":
                    parts.add(target)

        if failures_by_component:
            counts = sorted(
                failures_by_component.items(),
                key=lambda item: (-item[1], graph.nodes[item[0]].get("name", item[0])),
            )
            values = ", ".join(f"{graph.nodes[node].get('name', node)} — {count}" for node, count in counts)
            facts.append(f"Отказы по узлам линейки «{line_name}»: {values}")

        if failures_by_model:
            counts = sorted(
                failures_by_model.items(),
                key=lambda item: (-item[1], graph.nodes[item[0]].get("name", item[0])),
            )
            values = ", ".join(f"{graph.nodes[model].get('name', model)} — {count}" for model, count in counts)
            facts.append(f"Отказы по моделям линейки «{line_name}»: {values}")

        replaced_parts = []
        for part in sorted(parts):
            if any(relation.get("type") == "REPLACED_BY" for _, _, relation in graph.out_edges(part, data=True)):
                replaced_parts.append(graph.nodes[part].get("name", part))
        if replaced_parts:
            facts.append(f"Детали с заменой в линейке «{line_name}»: {', '.join(replaced_parts)}")

    for code in find_codes(question):
        node = f"{code_type(code)}:{code}"
        if node not in graph:
            continue
        for source, _, relation in graph.in_edges(node, data=True):
            facts.append(
                f"{graph.nodes[source].get('name', source)} -{relation.get('type', 'RELATED_TO')}-> "
                f"{graph.nodes[node].get('name', node)}"
            )
        for _, target, relation in graph.out_edges(node, data=True):
            facts.append(
                f"{graph.nodes[node].get('name', node)} -{relation.get('type', 'RELATED_TO')}-> "
                f"{graph.nodes[target].get('name', target)}"
            )

    return list(dict.fromkeys(facts))


def global_search(question: str, kb: KnowledgeBase, llm: LLMClient) -> Answer:
    """Отвечает на обзорный вопрос по сводкам сообществ в два шага.

    Шаг map: для каждой сводки модель даёт частичный ответ и оценку полезности от 0 до 10.
    Шаг reduce: частичные ответы с оценкой выше 0 объединяются в итоговый ответ.
    Если таких ответов нет, возвращается `Answer(..., refused=True)`.
    """
    codes = find_codes(question)
    if any(f"{code_type(code)}:{code}" not in kb.graph for code in codes):
        return Answer(REFUSAL_TEXT, refused=True)

    partial_answers: list[str] = []
    sources: list[str] = []
    graph = kb.graph
    structured_facts = _global_graph_facts(question, kb)
    for community in kb.communities:
        context = f"Сводка сообщества:\n{community.summary}"
        if structured_facts:
            context += "\n\nТочные факты и расчёты по графу, подготовленные кодом:\n"
            context += "\n".join(f"- {fact}" for fact in structured_facts)
        response = llm.chat(
            [
                {
                    "role": "system",
                    "content": (
                        "Ответь на вопрос только по сводке и точным фактам графа. Рассматривай направленные "
                        "связи и рассчитанные значения как доказательства; не пересчитывай и не заменяй их "
                        "приблизительными оценками. Если данные позволяют ответить, верни конкретный частичный "
                        "ответ и score от 1 до 10. Если данных недостаточно, верни пустой answer и score 0."
                    ),
                },
                {
                    "role": "user",
                    "content": f"Вопрос: {question}\n\n{context}",
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

        if answer.strip() not in partial_answers:
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
                    "частичных ответов и точных фактов графа ниже. Рассчитанные значения считай точными, "
                    "не заменяй их предположениями. Если сведений недостаточно, напиши: "
                    f"«{REFUSAL_TEXT}»."
                ),
            },
            {
                "role": "user",
                "content": (
                    f"Вопрос: {question}\n\n"
                    + (
                        "Точные факты и расчёты по графу:\n"
                        + "\n".join(f"- {fact}" for fact in structured_facts)
                        + "\n\n"
                        if structured_facts
                        else ""
                    )
                    + "Полезные частичные ответы:\n"
                    + "\n".join(f"{index}. {answer}" for index, answer in enumerate(partial_answers, start=1))
                ),
            },
        ]
    )
    return Answer(
        reply.content,
        sources=sources,
        facts=partial_answers,
        refused=REFUSAL_TEXT in reply.content.lower(),
    )
