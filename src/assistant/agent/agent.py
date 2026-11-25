"""Цикл агента. Требования FR-7, FR-8, FR-9 и FR-10."""

import json
import re
from collections.abc import Callable

from assistant.agent.tools import TOOL_SCHEMAS
from assistant.infra.erp_client import ErpUnavailable
from assistant.infra.grounding import find_codes, normalize_code
from assistant.infra.llm import LLMClient, ToolCall, tool_message
from assistant.infra.schemas import AgentResult, ToolStep

SYSTEM = (
    "Ты ассистент сервисной службы компании «ПолярТех». Отвечай на вопрос только по результатам инструментов "
    "и не придумывай номера деталей, цены и остатки. Коды моделей, деталей, ошибок и регламентов пиши точно "
    "так, как они записаны в результатах. Факты о заменах, кодах ошибок и регламентах ищи через "
    "search_knowledge, обзорные сведения — через summarize_topic. Заказ create_service_order вызывай, только "
    "если пользователь назвал количество и склад. Если данных для ответа нет, напиши: «в базе знаний не "
    "найдено»."
)

REFUSAL = "В базе знаний не найдено."
ERP_TOOLS = {"check_stock", "get_price", "create_service_order"}
ERP_ATTEMPTS = 3

# Число с единицей измерения: «2630 руб», «2 630 ₽», «3 шт».
_MEASURED = re.compile(r"(\d[\d\s\u00a0]*\d|\d)\s*(?:руб|₽|шт)", re.IGNORECASE)
# Отдельно стоящее число: не часть кода вида P-1042B или AR-210.
_NUMBER = re.compile(r"(?<![\w-])(\d+)(?![\w-])")


def _reject(call: ToolCall, arguments: dict, question: str, steps: list[ToolStep], tools: dict) -> str | None:
    """Проверка вызова до инструмента: имя, читаемость аргументов, деталь из вопроса, количество в заказе."""
    if call.name not in tools:
        return f"неизвестный инструмент {call.name!r}; доступны: {', '.join(sorted(tools))}"
    if "_raw" in arguments:
        return "аргументы не читаются как JSON; повторите вызов с корректными аргументами"
    if "part_id" in arguments:
        raw = arguments["part_id"]
        code = normalize_code(raw) if isinstance(raw, str) else None
        known = set(find_codes(question))
        for step in steps:
            if step.result is not None:
                known.update(find_codes(json.dumps(step.result, ensure_ascii=False)))
        if code is None or code not in known:
            return (
                f"номера детали {raw!r} нет ни в вопросе, ни в результатах предыдущих шагов. "
                "Сначала вызовите search_knowledge и возьмите номер детали оттуда."
            )
    if call.name == "create_service_order" and str(arguments.get("qty")).strip() not in _NUMBER.findall(question):
        return "пользователь не назвал количество; спросите, сколько штук заказать"
    return None


def _execute(tool: Callable[..., dict], arguments: dict) -> dict:
    """Вызов инструмента: TypeError становится ошибкой шага, ErpUnavailable повторяется до трёх раз."""
    failure: ErpUnavailable | None = None
    for _ in range(ERP_ATTEMPTS):
        try:
            return tool(**arguments)
        except ErpUnavailable as exc:
            failure = exc
        except TypeError as exc:
            return {"error": f"инструмент не принял такие аргументы: {exc}"}
    return {"error": f"данные ERP получить не удалось: {failure}"}


def _sources(steps: list[ToolStep]) -> list[str]:
    """Документы-источники из результатов поиска в порядке появления, без повторов."""
    found: list[str] = []
    for step in steps:
        for source in (step.result or {}).get("sources") or []:
            if source not in found:
                found.append(source)
    return found


def _grounded(question: str, steps: list[ToolStep]) -> str:
    """Текст, с которым сверяется ответ: вопрос и результаты выполненных шагов."""
    parts = [question]
    parts.extend(json.dumps(step.result, ensure_ascii=False) for step in steps if step.result is not None)
    return "\n".join(parts)


def _measured(text: str) -> list[str]:
    """Значения рядом с «руб», «₽» или «шт» без пробелов внутри числа."""
    return [re.sub(r"\s+", "", match.group(1)) for match in _MEASURED.finditer(text)]


def _ungrounded(answer: str, grounded: str) -> list[str]:
    """Коды и величины с единицами, которых нет ни в вопросе, ни в результатах инструментов."""
    flat = re.sub(r"\s+", "", grounded)
    missing = [code for code in find_codes(answer) if code not in flat]
    missing.extend(value for value in _measured(answer) if value not in flat)
    return missing


def _correction(problems: list[str]) -> str:
    return (
        "В ответе есть коды и величины, которых нет в вопросе и результатах инструментов: "
        f"{', '.join(problems)}. Перепиши ответ только по собранным фактам; если фактов нет, напиши: "
        "«в базе знаний не найдено»."
    )


def _refused(answer: str, steps: list[ToolStep]) -> bool:
    """Отказ, если ответ сам говорит об этом или все поиски отказались, а данных ERP нет."""
    if "не найдено" in answer.lower():
        return True
    searches = [step.result for step in steps if step.result is not None and "refused" in step.result]
    if not searches or not all(result["refused"] for result in searches):
        return False
    has_erp_data = any(
        step.tool in ERP_TOOLS and step.result is not None and "error" not in step.result for step in steps
    )
    return not has_erp_data


def _stopped(steps: list[ToolStep], reason: str) -> AgentResult:
    """Ответ при остановке цикла: только то, что уже собрано, без обращения к модели."""
    lines = [f"Работа прервана: {reason}."]
    if steps:
        lines.append("Успел узнать:")
    for step in steps:
        shown = f"ошибка: {step.error}" if step.error else json.dumps(step.result, ensure_ascii=False)
        lines.append(f"- {step.tool}: {shown}")
    return AgentResult("\n".join(lines), steps=steps, sources=_sources(steps))


def _pending(arguments: dict) -> str:
    return (
        f"Заказ детали {arguments.get('part_id')} в количестве {arguments.get('qty')} шт. на склад "
        f"{arguments.get('warehouse')} ждёт подтверждения: повторите запрос с подтверждением."
    )


def run_agent(
    question: str,
    llm: LLMClient,
    tools: dict[str, Callable[..., dict]],
    *,
    max_steps: int = 6,
    confirm: bool = False,
) -> AgentResult:
    """Ведёт диалог с моделью в цикле «рассуждение, вызов инструмента, наблюдение».

    Модель получает `TOOL_SCHEMAS` через `llm.chat(..., tools=...)`. Каждый вызов записывается
    в `steps` как `ToolStep`. Агент сам проверяет номер детали в аргументах (он должен встречаться
    в вопросе или в результатах прошлых инструментов), количество в заказе (должно быть названо
    пользователем) и повторные вызовы. При `ErpUnavailable` вызов повторяется до двух раз.
    Заказ создаётся только при `confirm=True`, иначе агент возвращает `needs_confirmation=True`
    и `pending_action`. Перед выдачей ответа коды, цены и остатки сверяются с вопросом
    и результатами инструментов. Число ответов модели ограничено, чтобы отклоняемые вызовы
    и кеш-хиты не зациклили диалог.
    """
    messages: list[dict] = [
        {"role": "system", "content": SYSTEM},
        {"role": "user", "content": question},
    ]
    steps: list[ToolStep] = []
    cache: dict[tuple[str, str], dict] = {}
    executed = 0
    rounds = 0
    limit = 2 * max_steps + 2
    answer = ""

    while rounds < limit:
        rounds += 1
        if executed >= max_steps:
            return _stopped(steps, f"исчерпан лимит вызовов инструментов ({max_steps})")
        response = llm.chat(messages, tools=TOOL_SCHEMAS)
        messages.append(response.as_message())
        if not response.tool_calls:
            answer = (response.content or "").strip()
            break
        for call in response.tool_calls:
            if executed >= max_steps:
                return _stopped(steps, f"исчерпан лимит вызовов инструментов ({max_steps})")
            arguments = dict(call.arguments)
            key = (call.name, json.dumps(arguments, ensure_ascii=False, sort_keys=True))
            problem = _reject(call, arguments, question, steps, tools)
            if problem is not None:
                steps.append(ToolStep(call.name, arguments, error=problem))
                messages.append(tool_message(call, {"error": problem}))
                continue
            if call.name == "create_service_order" and not confirm:
                return AgentResult(
                    _pending(arguments),
                    steps=steps,
                    sources=_sources(steps),
                    needs_confirmation=True,
                    pending_action={"tool": call.name, "args": arguments},
                )
            if key in cache:
                messages.append(tool_message(call, cache[key]))
                continue
            executed += 1
            result = _execute(tools[call.name], arguments)
            cache[key] = result
            error = str(result["error"]) if "error" in result else None
            steps.append(ToolStep(call.name, arguments, result=result, error=error))
            messages.append(tool_message(call, result))

    if rounds >= limit:
        return _stopped(steps, f"модель предложила больше {limit} ответов подряд")
    if not answer:
        return AgentResult(REFUSAL, steps=steps, sources=_sources(steps), refused=True)
    grounded = _grounded(question, steps)
    problems = _ungrounded(answer, grounded)
    if problems:
        messages.append({"role": "user", "content": _correction(problems)})
        answer = (llm.chat(messages).content or "").strip()
        if not answer or _ungrounded(answer, grounded):
            return AgentResult(REFUSAL, steps=steps, sources=_sources(steps), refused=True)
    return AgentResult(answer, steps=steps, sources=_sources(steps), refused=_refused(answer, steps))
