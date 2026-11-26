"""Извлечение сущностей и связей из чанка. Требование FR-1."""

import json

from assistant.infra.config import vocabulary
from assistant.infra.grounding import CODE_TYPES, canonical_name, find_codes, normalize_code
from assistant.infra.llm import LLMClient, LLMError, LLMResponse
from assistant.infra.schemas import EXTRACTION_SCHEMA, Chunk, Entity, ExtractionResult, Relation


def extract_chunk(chunk: Chunk, llm: LLMClient) -> ExtractionResult:
    """Возвращает сущности и связи чанка строго по схеме графа.

    Модель вызывается через `llm.chat(..., json_schema=EXTRACTION_SCHEMA)`. Запрос описывает схему
    с направлением каждой связи и содержит хотя бы один пример. Всё, что не подтверждается текстом,
    отбрасывается и считается в `dropped`: код, которого нет в чанке, название не из словаря,
    связь с отброшенным концом. Пригодятся `find_codes` и `canonical_name` из `assistant.infra.grounding`.
    Функция не падает ни на каком ответе модели: одна повторная попытка, затем пустой результат.
    """

    def normalize_code_extra(value: str) -> str | None:
        if (after_reg_norm := normalize_code(value)) is not None:
            return after_reg_norm

        return normalize_code("".join(value.split()))


    def resolve_entity(entity_type: str | None, entity_name: str | None) -> tuple[str, str] | None:
        if entity_type is None or entity_name is None:
            return None

        if entity_type in CODE_TYPES:
            entity_name = normalize_code_extra(entity_name)
            if entity_name is None or entity_name not in unique_codes:
                return None

        canon_name = canonical_name(entity_type, entity_name, vocab)
        if canon_name is None:
            return None

        return entity_type, canon_name

    SYSTEM_PROMPT = """Ты — система извлечения знаний для построения графа для компании, 
    которая производит и обслуживает холодильное оборудование для магазинов и складов. 
    У неё три линейки: витрины «Арктика», моноблоки для холодильных камер «Бриз»
    и чиллеры «Гольфстрим», по четыре модели в каждой.
    Извлеки из текста сущности и связи строго по схеме. Не выдумывай факты, которых нет в тексте.
    
    ### Типы сущностей
    ENTITY_TYPES = ("ProductLine", "Model", "Component", "Part", "ErrorCode", "Procedure", "Supplier")
    
    ### Словарь названий
    "ProductLine": "Арктика", "Бриз", "Гольфстрим"
    "Component": "компрессор", "вентилятор конденсатора", "датчик температуры",
    "контроллер", "реле давления", "пусковое реле", "вентилятор испарителя",
    "блок освещения", "нагреватель оттайки", "терморегулирующий вентиль",
    "циркуляционный насос", "реле протока"
    "Supplier": "АвтоматикаСевер", "ВентПром", "ГидроТех", "КомпрессорМаш", "ТермоПрибор", "ХолодКомплект"
   
     ### Префикс кода сущности -> (число цифр, тип сущности).
    "AR" -> (3, "Model")
    "BZ" -> (3, "Model")
    "GS" -> (3, "Model")
    "P" -> (4, "Part")
    "E" -> (2, "ErrorCode")
    "SP" -> (3, "Procedure")
    
    ***
    Важно: в ответе коды сущностей должны быть валидными, формат: [ПРЕФИКС]-[2-4 цифры][необязательная 1 буква] 
    Примеры валидных кодов: AR-210, P-1122, P-2054B, P-3062C, E-19, SP-101, GS-510, BZ-220 
    Примеры невалидных кодов: P-2 034B, SP-1_101, E19
    ***
    
    ### Схема графа и направление связей:

    ### Тип связи: (subject_type -> object_type). Связь с другими типами концов в граф не попадает.
    "BELONGS_TO": ("Model" -> "ProductLine")
    "HAS_PART": ("Model" -> "Part")
    "IS_A": ("Part" -> "Component")
    "INDICATES": ("ErrorCode" -> "Component")
    "FIXED_BY": ("ErrorCode" -> "Procedure")
    "REPLACED_BY": ("Part" -> "Part")
    "SUPPLIED_BY": ("Part" -> "Supplier")
    "FAILED": ("Model" -> "Component")
    
    ### Правила извлечения связей для графа
    1. Извлекай только то, что явно следует из текста.
    2. Не дублируй одинаковые сущности и одинаковые связи.
    3. Связи между одинаковыми типами разрешены только для Part (REPLACED_BY).
       Связи типа Model -> Model, Component -> Component запрещены.
    4. Если в тексте перечислены узлы/детали для конкретной модели, то:
       - создается связь с КАЖДОЙ деталью через HAS_PART (Model -> Part)
       - каждая деталь связывается с компонентом через IS_A
       - каждая деталь связывается с поставщиком через SUPPLIED_BY, если поставщик указан
    5. Если текст говорит о неисправности компонента у модели, используй FAILED: Model -> Component.
    6. Если принадлежность модели к линейке указана в тексте, используй тип связи BELONGS_TO.
    
    ***Важно: ответ должен представлять собой полностью валидный JSON***

    ### Примеры работы:
    Текст: "Датчик температуры является одним из компонентов модели AR-310.
    При неисправности датчика назначается ошибка E-24."
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
    
    Текст: "В BZ-170 установлены следующие узлы.
    компрессор – деталь P-1020, поставщик «КомпрессорМаш»
    вентилятор конденсатора – деталь P-2014, поставщик «ВентПром»"
    Ответ:
    {
      "entities": [
        {"type": "Model", "name": "BZ-170"},
        {"type": "Component", "name": "компрессор"},
        {"type": "Component", "name": "вентилятор конденсатора"},
        {"type": "Part", "name": "P-1020"},
        {"type": "Part", "name": "P-2014"},
        {"type": "Supplier", "name": "КомпрессорМаш"},
        {"type": "Supplier", "name": "ВентПром"}
      ],
      "relations": [
        {
          "type": "HAS_PART",
          "subject_type": "Model",
          "subject": "BZ-170",
          "object_type": "Part",
          "object": "P-1020"
        },
        {
          "type": "HAS_PART",
          "subject_type": "Model",
          "subject": "BZ-170",
          "object_type": "Part",
          "object": "P-2014"
        },
        {
          "type": "IS_A",
          "subject_type": "Part",
          "subject": "P-1020",
          "object_type": "Component",
          "object": "компрессор"
        },
        {
          "type": "IS_A",
          "subject_type": "Part",
          "subject": "P-2014",
          "object_type": "Component",
          "object": "вентилятор конденсатора"
        },
        {
          "type": "SUPPLIED_BY",
          "subject": "P-1020",
          "subject_type": "Part",
          "object": "КомпрессорМаш",
          "object_type": "Supplier"
        },
        {
          "type": "SUPPLIED_BY",
          "subject": "P-2014",
          "subject_type": "Part",
          "object": "ВентПром",
          "object_type": "Supplier"
        }
      ]
    }
    
    Текст: "Поставщик детали P-1012B – «ГидроТех»"
    Ответ:
    {
      "entities": [
        {"type": "Part", "name": "P-1012B"},
        {"type": "Supplier", "name": "ГидроТех"}
      ],
      "relations": [
        {
          "type": "SUPPLIED_BY",
          "subject": "P-1012B",
          "subject_type": "Part",
          "object": "ГидроТех",
          "object_type": "Supplier"
        }
      ]
    }
    
    Текст: "Регламент SP-205 применяется при ошибке E-99. Ошибка E-99 указывает на отказ узла «контроллер»"
    Ответ: 
    {
      "entities": [
        {"name": "SP-205", "type": "Procedure"},
        {"name": "E-99", "type": "ErrorCode"},
        {"name": "контроллер", "type": "Component"}
      ],
      "relations": [
        {
          "object": "контроллер",
          "object_type": "Component",
          "subject": "E-99",
          "subject_type": "ErrorCode",
          "type": "INDICATES"
        },
        {
          "object": "SP-205",
          "object_type": "Procedure",
          "subject": "E-99",
          "subject_type": "ErrorCode",
          "type": "FIXED_BY"
        }
      ]
    }
    """

    vocab = vocabulary()
    unique_codes = {
        code
        for raw in find_codes(chunk.text)
        if (code := normalize_code_extra(raw)) is not None
    }
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
        except LLMError as e:
            print(f"llm_error={e}")
            continue

        try:
            content = json.loads(resp.content)
        except json.JSONDecodeError:
            continue

        if "entities" not in content or "relations" not in content:
            continue

        raw_entities = content["entities"]
        raw_relations = content["relations"]
        if not isinstance(raw_entities, list) or not isinstance(raw_relations, list):
            continue

        unique_entities = set()
        unique_relations = set()
        final_relations = []

        for e in raw_entities:
            if not isinstance(e, dict):
                dropped += 1
                continue

            resolved = resolve_entity(e.get("type"), e.get("name"))
            if resolved is None:
                dropped += 1
                continue

            unique_entities.add(resolved)

        for r in raw_relations:
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

            if (subj_resolved, rel_type, obj_resolved) not in unique_relations:
                unique_relations.add((subj_resolved, rel_type, obj_resolved))

                rel = Relation(Entity(*subj_resolved), rel_type, Entity(*obj_resolved))
                if not rel.matches_schema():
                    dropped += 1
                    continue

                final_relations.append(rel)

        return ExtractionResult(
            chunk_id=chunk.chunk_id,
            doc_id=chunk.doc_id,
            entities=[Entity(*e) for e in unique_entities],
            relations=final_relations,
            dropped=dropped
        )

    return ExtractionResult(chunk_id=chunk.chunk_id, doc_id=chunk.doc_id, dropped=dropped)
