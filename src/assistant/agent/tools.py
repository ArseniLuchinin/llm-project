"""Инструменты агента. Требование FR-6."""

from collections.abc import Callable

from assistant.infra.erp_client import ErpClient
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


def build_tools(kb: KnowledgeBase, llm: LLMClient, erp: ErpClient) -> dict[str, Callable[..., dict]]:
    """Словарь «имя инструмента -> функция». Каждая функция возвращает словарь.

    Функция проверяет свои аргументы до обращения к ERP: склад из списка, номер детали вида P-1042
    или P-1042B, количество больше нуля. При ошибке возвращает {"error": "..."} и ничего не выполняет.
    `ErpNotFound` и `ErpRejected` превращаются в {"error": "..."} с разным текстом, чтобы исходы
    оставались различимыми. `ErpUnavailable` передаётся дальше: повторы делает агент (FR-8).
    """
    # TODO: пять функций поверх local_search, global_search и клиента ERP
    raise NotImplementedError("Реализуйте build_tools")
