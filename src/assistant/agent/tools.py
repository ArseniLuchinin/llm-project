"""Инструменты агента. Требование FR-6."""

from collections.abc import Callable

from assistant.graphrag.search import global_search, local_search
from assistant.infra.erp_client import ErpClient, ErpNotFound, ErpRejected
from assistant.infra.grounding import code_type, normalize_code
from assistant.infra.llm import LLMClient
from assistant.infra.schemas import KnowledgeBase

# Склады учётной системы: тот же список, что в erp/seed.json и FakeErp контрактных тестов.
WAREHOUSES = ["Москва", "Санкт-Петербург", "Казань", "Новосибирск"]

_PART_ID = {
    "type": "string",
    "description": "Номер детали вида P-1042 или P-1042B. Не модель и не код ошибки.",
}
_WAREHOUSE = {
    "type": "string",
    "enum": WAREHOUSES,
    "description": "Склад, на котором проверяется наличие или оформляется заказ.",
}

# Описания пяти инструментов в формате function calling (FR-6). Описания попадают в промпт модели,
# поэтому говорят, когда инструмент вызывать и какие данные для него нужны.
TOOL_SCHEMAS: list[dict] = [
    {
        "type": "function",
        "function": {
            "name": "search_knowledge",
            "description": (
                "Ищет факты в графе знаний: замены деталей, коды ошибок, регламенты ремонта, отказы узлов. "
                "Вызывайте первым, если в вопросе нет номера детали или он неизвестен."
            ),
            "parameters": {
                "type": "object",
                "properties": {"question": {"type": "string", "description": "Вопрос к базе знаний."}},
                "required": ["question"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "summarize_topic",
            "description": (
                "Обзорный ответ по сообществам графа: типовые неисправности и частые отказы. "
                "Вызывайте, когда нужна сводка по теме, а не факт о конкретной сущности."
            ),
            "parameters": {
                "type": "object",
                "properties": {"question": {"type": "string", "description": "Тема обзора."}},
                "required": ["question"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "check_stock",
            "description": (
                "Остаток детали на складе и признак «снята с производства». "
                "Номер детали должен быть назван в вопросе или получен из search_knowledge."
            ),
            "parameters": {
                "type": "object",
                "properties": {"part_id": _PART_ID, "warehouse": _WAREHOUSE},
                "required": ["part_id", "warehouse"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "get_price",
            "description": (
                "Цена детали в рублях и признак «снята с производства». "
                "Номер детали должен быть назван в вопросе или получен из search_knowledge."
            ),
            "parameters": {
                "type": "object",
                "properties": {"part_id": _PART_ID},
                "required": ["part_id"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "create_service_order",
            "description": (
                "Создаёт заказ детали в учётной системе. Вызывайте, только когда пользователь назвал "
                "количество и подтвердил заказ; количество не придумывайте."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "part_id": _PART_ID,
                    "warehouse": _WAREHOUSE,
                    "qty": {"type": "integer", "minimum": 1, "description": "Количество штук из вопроса."},
                    "model": {
                        "type": "string",
                        "description": "Модель оборудования, для которой заказывается деталь, например AR-210.",
                    },
                },
                "required": ["part_id", "warehouse", "qty", "model"],
            },
        },
    },
]


class _BadArgument(ValueError):
    """Аргумент не прошёл проверку: инструмент не обращается ни к ERP, ни к модели."""


def _warehouse(raw: object) -> str:
    """Склад из закрытого списка; лишние пробелы значения не имеют."""
    name = raw.strip() if isinstance(raw, str) else ""
    if name not in WAREHOUSES:
        raise _BadArgument(f"неизвестный склад {raw!r}, допустимы: {', '.join(WAREHOUSES)}")
    return name


def _part(raw: object) -> str:
    """Канонический номер детали: P-1042, p 1042 или P-1042B. Модели и коды ошибок отклоняются."""
    code = normalize_code(raw) if isinstance(raw, str) else None
    if code is None or code_type(code) != "Part":
        raise _BadArgument(f"{raw!r} не номер детали: нужен код вида P-1042 или P-1042B")
    return code


def _qty(raw: object) -> int:
    """Положительное целое количество; bool и дробные числа количеством не считаются."""
    value = int(raw.strip()) if isinstance(raw, str) and raw.strip().isdecimal() else raw
    if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
        raise _BadArgument(f"количество {raw!r} должно быть целым числом больше нуля")
    return value


def _model(raw: object) -> str:
    """Название модели оборудования, для которой заказывается деталь."""
    if not isinstance(raw, str) or not raw.strip():
        raise _BadArgument("model должен быть названием модели оборудования, например AR-210")
    return raw.strip()


def _erp_call(method: Callable[..., dict], *args) -> dict:
    """Вызов ERP: ожидаемые отказы становятся {"error"}, ErpUnavailable передаётся дальше."""
    try:
        return method(*args)
    except (ErpNotFound, ErpRejected) as exc:
        return {"error": str(exc)}


def build_tools(kb: KnowledgeBase, llm: LLMClient, erp: ErpClient) -> dict[str, Callable[..., dict]]:
    """Словарь «имя инструмента -> функция». Каждая функция возвращает словарь.

    Функция проверяет свои аргументы до обращения к ERP: склад из списка, номер детали вида P-1042
    или P-1042B, количество больше нуля. При ошибке возвращает {"error": "..."} и ничего не выполняет.
    `ErpNotFound` и `ErpRejected` превращаются в {"error": "..."} с разным текстом, чтобы исходы
    оставались различимыми. `ErpUnavailable` передаётся дальше: повторы делает агент (FR-8).
    """

    def search_knowledge(question: str) -> dict:
        answer = local_search(question, kb, llm)
        return {
            "answer": answer.text,
            "facts": list(answer.facts),
            "sources": list(answer.sources),
            "refused": answer.refused,
        }

    def summarize_topic(question: str) -> dict:
        answer = global_search(question, kb, llm)
        return {"answer": answer.text, "sources": list(answer.sources), "refused": answer.refused}

    def check_stock(part_id: object, warehouse: object) -> dict:
        try:
            code, place = _part(part_id), _warehouse(warehouse)
        except _BadArgument as exc:
            return {"error": str(exc)}
        return _erp_call(erp.get_stock, code, place)

    def get_price(part_id: object) -> dict:
        try:
            code = _part(part_id)
        except _BadArgument as exc:
            return {"error": str(exc)}
        return _erp_call(erp.get_price, code)

    def create_service_order(part_id: object, warehouse: object, qty: object, model: object) -> dict:
        try:
            code, place, count, equipment = _part(part_id), _warehouse(warehouse), _qty(qty), _model(model)
        except _BadArgument as exc:
            return {"error": str(exc)}
        return _erp_call(erp.create_order, code, place, count, equipment)

    return {
        "search_knowledge": search_knowledge,
        "summarize_topic": summarize_topic,
        "check_stock": check_stock,
        "get_price": get_price,
        "create_service_order": create_service_order,
    }
