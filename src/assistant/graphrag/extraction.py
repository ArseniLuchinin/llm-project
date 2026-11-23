"""Извлечение сущностей и связей из чанка. Требование FR-1."""

import json
from assistant.infra.llm import LLMClient, LLMResponse, LLMError
from assistant.infra.config import vocabulary
from assistant.infra.grounding import find_codes, normalize_code, canonical_name, CODE_TYPES
from assistant.infra.schemas import Chunk, ExtractionResult, EXTRACTION_SCHEMA, Entity, Relation, RELATION_SIGNATURES


def extract_chunk(chunk: Chunk, llm: LLMClient) -> ExtractionResult:
    """Возвращает сущности и связи чанка строго по схеме графа.

    Модель вызывается через `llm.chat(..., json_schema=EXTRACTION_SCHEMA)`. Запрос описывает схему
    с направлением каждой связи и содержит хотя бы один пример. Всё, что не подтверждается текстом,
    отбрасывается и считается в `dropped`: код, которого нет в чанке, название не из словаря,
    связь с отброшенным концом. Пригодятся `find_codes` и `canonical_name` из `assistant.infra.grounding`.
    Функция не падает ни на каком ответе модели: одна повторная попытка, затем пустой результат.
    """

    SYSTEM_PROMPT = """Ты — система извлечения знаний для построения графа для компании, 
    которая производит и обслуживает холодильное оборудование для магазинов и складов. 
    У неё три линейки: витрины «Арктика», моноблоки для холодильных камер «Бриз» и чиллеры «Гольфстрим», по четыре модели в каждой.
    Извлеки из текста сущности и связи строго по схеме. Не выдумывай факты, которых нет в тексте.

    ### Схема графа и направление связей:
    
    ### Типы сущностей
    ENTITY_TYPES = ("ProductLine", "Model", "Component", "Part", "ErrorCode", "Procedure", "Supplier")

    ### Тип связи -> (тип начала, тип конца). Связь с другими типами концов в граф не попадает.
    
    "BELONGS_TO" -> ("Model", "ProductLine")
    "HAS_PART" -> ("Model", "Part")
    "IS_A" -> ("Part", "Component")
    "INDICATES" -> ("ErrorCode", "Component")
    "FIXED_BY" -> ("ErrorCode", "Procedure")
    "REPLACED_BY" -> ("Part", "Part")
    "SUPPLIED_BY" -> ("Part", "Supplier")
    "FAILED" -> ("Model", "Component")
    
    ### Префикс кода сущности -> (число цифр, тип сущности).

    "AR" -> (3, "Model")
    "BZ" -> (3, "Model")
    "GS" -> (3, "Model")
    "P" -> (4, "Part")
    "E" -> (2, "ErrorCode")
    "SP" -> (3, "Procedure")
    
    ***Важно: ответ должен представлять собой полностью валидный JSON***

    ### Пример работы:
    Текст: "Датчик температуры является одним из компонентов модели AR-310. При неисправности датчика назначается ошибка E-24."
    Ответ:
    {
      "entities": [
        {"type": "Model", "name": "AR-310"},
        {"type": "ErrorCode", "name": "E-24"},
        {"type": "Component", "name": "датчик температуры"}
      ],
      "relations": [
        {
          "type": "FAILED",
          "subject_type": "Model",
          "subject": "AR-310",
          "object_type": "Component",
          "object": "датчик температуры"
        },
        {
          "type": "INDICATES",
          "subject_type": "ErrorCode",
          "subject": "E-24",
          "object_type": "Component",
          "object": "датчик температуры"
        }
      ]
    }
    """

    vocab = vocabulary()
    unique_codes = set(find_codes(chunk.text))
    dropped = 0

    for _ in range(2):
        try:
            resp: LLMResponse = llm.chat(
                messages=[
                    {"role": "system", "content": SYSTEM_PROMPT},
                    {"role": "user", "content": chunk.text}
                ],
                json_schema=EXTRACTION_SCHEMA,
                max_tokens=2048
            )

            print(f"extraction response: {resp.content}")
        except LLMError as e:
            print(f"llm_error={e}")
            continue

        try:
            content = json.loads(resp.content)
        except json.JSONDecodeError:
            continue

        if not "entities" in content or not "relations" in content:
            continue

        entities = content["entities"]
        relations = content["relations"]
        if not isinstance(entities, list) or not isinstance(relations, list):
            continue

        def resolve_entity(entity_type: str | None, entity_name: str | None) -> tuple[str, str] | None:
            if entity_type is None or entity_name is None:
                return None

            if entity_type in CODE_TYPES and normalize_code(entity_name) not in unique_codes:
                return None

            canon_name = canonical_name(entity_type, entity_name, vocab)
            if canon_name is None:
                return None

            return entity_type, canon_name

        valid_entities = set()
        valid_relations = set()

        for e in entities:
            if not isinstance(e, dict):
                dropped += 1
                continue

            resolved = resolve_entity(e.get("type"), e.get("name"))
            if resolved is None:
                dropped += 1
                continue

            valid_entities.add(resolved)

        for r in relations:
            if not isinstance(r, dict):
                dropped += 1
                continue

            rel_type = r.get("type")
            if rel_type is None:
                dropped += 1
                continue

            subj_resolved = resolve_entity(r.get("subject_type"), r.get("subject"))
            obj_resolved = resolve_entity(r.get("object_type"), r.get("object"))

            if subj_resolved is None or obj_resolved is None:
                dropped += 1
                continue

            if not rel_type in RELATION_SIGNATURES:
                dropped += 1
                continue

            valid_relations.add((subj_resolved, rel_type, obj_resolved))

        return ExtractionResult(
            chunk_id=chunk.chunk_id,
            doc_id=chunk.doc_id,
            entities=[Entity(*e) for e in valid_entities],
            relations=[Relation(Entity(*r[0]), r[1], Entity(*r[2])) for r in valid_relations],
            dropped=dropped
        )

    return ExtractionResult(chunk_id=chunk.chunk_id, doc_id=chunk.doc_id, dropped=dropped)
