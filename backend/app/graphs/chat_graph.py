from __future__ import annotations

import asyncio
import json
import re
from functools import lru_cache
from typing import Any, Literal, TypedDict

from langchain.agents import create_agent
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_mcp_adapters.client import MultiServerMCPClient
from langchain_openai import ChatOpenAI
from langgraph.graph import END, START, StateGraph

from llm.llm import read_llm
from project_config import SETTINGS


RouteType = Literal["ticket", "rag", "roadmap", "other"]
ChatStatus = Literal["completed", "needs_confirmation"]

_TICKET_KEYWORDS = {
    "12306", "火车票", "高铁票", "车票", "余票", "车次", "改签", "退票", "购票", "抢票", "出发站", "到达站"
}
_ROADMAP_KEYWORDS = {
    "路线", "导航", "地图", "高德", "天气", "路况", "附近", "距离", "位置", "坐标", "打车", "公交", "地铁", "驾车",
    "景点", "景区", "游玩", "旅游", "旅行", "一日游", "两日游", "打卡", "行程", "攻略",
    "住宿", "住哪里", "酒店", "宾馆", "民宿", "旅店", "客栈", "青旅",
}
_RAG_KEYWORDS = {
    "铁路购票",
    "车站乘车",
    "地图导航",
    "跨城出行",
    "出行应急",
    "出行建议",
    "城市公共交通",
    "客服对话",
}
_CONFIRM_OBJECT_KEYWORDS = {
    "酒店", "宾馆", "民宿", "车票", "火车票", "高铁票", "机票", "门票", "船票", "票",
}
_HOTEL_OBJECT_KEYWORDS = {
    "酒店", "宾馆", "民宿",
}
_SPECIFIC_TICKET_OBJECT_KEYWORDS = {
    "车票", "火车票", "高铁票", "机票", "门票", "船票",
}
_SPECIFIC_TICKET_DETAIL_KEYWORDS = {
    "12306", "车次", "出发站", "到达站", "从", "到", "明天", "后天", "今天",
    "上午", "下午", "晚上", "几点", "一等座", "二等座", "头等舱", "经济舱",
}
_CONFIRM_ACTION_KEYWORDS = {
    "订", "预订", "预定", "预约", "订房", "订票", "购票", "买", "购买", "出票", "抢票", "下单",
}
_CONFIRM_REPLY_KEYWORDS = {
    "确认", "确定", "同意", "继续", "可以", "好的", "好", "行", "没问题", "yes", "ok", "okay", "sure",
}
_SELECTION_LOCK_KEYWORDS = {
    "我选这个", "就这个", "确定这个", "按这个来", "可以，就这个方案", "选这个", "这个方案", "定这个",
}
_FINAL_PLAN_CONFIRM_KEYWORDS = {
    "确认行程", "确认计划", "确认这个计划", "确认这份计划", "确认出行计划", "就按这个行程",
}
_TRAVEL_PLAN_KEYWORDS = {
    "旅行", "旅游", "行程", "攻略", "出游", "出行计划", "酒店", "民宿", "景点", "路线", "高铁", "火车票", "机票",
}


class ChatGraphState(TypedDict, total=False):
    question: str
    effective_question: str
    rewritten_question: str
    context_summary: str
    detected_needs: list[str]
    prior_routes: list[str]
    history: list[dict[str, Any]]
    top_k: int
    user_id: int
    route: RouteType
    status: ChatStatus
    confirmed: bool
    requires_confirmation: bool
    pending_confirmation: dict[str, Any]
    previous_task_summary: dict[str, Any]
    answer: str
    answer_source: str
    verification: dict[str, Any]
    final_summary: dict[str, Any]
    carry_context: bool
    current_plan_id: int | None
    plan_draft: dict[str, Any]
    locked_fields: dict[str, Any]
    candidate_options: list[dict[str, Any]]
    selection_action: str
    selection_target: dict[str, Any]
    ready_for_final_confirmation: bool
    final_confirmation_payload: dict[str, Any]
    plan_confirmation_completed: bool


def _build_llm(temperature: float = 0) -> ChatOpenAI:
    read_llm()
    return ChatOpenAI(model=SETTINGS.llm_model, temperature=temperature)


def _strip_think_tags(text: str) -> str:
    return re.sub(r"<think>.*?</think>", "", text, flags=re.DOTALL).strip()


def _normalize_history(history: list[dict[str, Any]] | None) -> list[dict[str, Any]]:
    normalized: list[dict[str, Any]] = []
    for item in history or []:
        role = str(item.get("role", "")).strip().lower() or "user"
        content = str(item.get("content", "")).strip()
        if not content:
            continue
        normalized_item: dict[str, Any] = {"role": role, "content": content}
        metadata = item.get("metadata")
        if isinstance(metadata, dict) and metadata:
            normalized_item["metadata"] = metadata
        normalized.append(normalized_item)
    return normalized


def _extract_pending_confirmation(history: list[dict[str, Any]] | None) -> dict[str, Any]:
    for item in reversed(history or []):
        if str(item.get("role", "")).strip().lower() != "assistant":
            continue
        metadata = item.get("metadata")
        if not isinstance(metadata, dict):
            continue
        pending = metadata.get("pending_confirmation")
        if isinstance(pending, dict) and pending:
            return pending
    return {}


def _extract_latest_final_summary(history: list[dict[str, Any]] | None) -> dict[str, Any]:
    for item in reversed(history or []):
        if str(item.get("role", "")).strip().lower() != "assistant":
            continue
        metadata = item.get("metadata")
        if not isinstance(metadata, dict):
            continue
        final_summary = metadata.get("final_summary")
        if isinstance(final_summary, dict) and final_summary:
            return final_summary
    return {}


def _serialize_previous_task_summary(summary: dict[str, Any]) -> str:
    if not isinstance(summary, dict) or not summary:
        return "无"
    payload = {
        "上轮问题": str(summary.get("rewritten_question", "")).strip() or str(summary.get("effective_question", "")).strip(),
        "上轮需求": summary.get("detected_needs", []),
        "上轮背景": str(summary.get("context_summary", "")).strip(),
        "上轮路由": str(summary.get("route", "")).strip(),
    }
    cleaned = {key: value for key, value in payload.items() if value not in ("", [], {}, None)}
    return json.dumps(cleaned, ensure_ascii=False)


def _is_context_dependent_query(text: str) -> bool:
    normalized = str(text or "").strip().lower()
    if not normalized:
        return False
    followup_keywords = {
        "继续", "接着", "然后", "顺便", "另外", "再", "再加", "补充", "修改", "改", "改成", "换", "换成",
        "调整", "删除", "删掉", "不要", "按这个", "照这个", "基于这个", "这个方案", "上面的", "刚才的", "之前那个",
        "在这个基础上", "延续上一个", "延续刚才", "沿用刚才", "基于上一个", "基于刚才",
    }
    explicit_followups = {
        "这个", "这个呢", "那个", "那个呢", "然后呢", "接下来呢", "继续吧", "继续", "再来一个",
    }
    if normalized in explicit_followups:
        return True
    return any(keyword in normalized for keyword in followup_keywords)


def _extract_topic_tokens(text: str) -> set[str]:
    normalized = str(text or "").lower()
    chinese_tokens = re.findall(r"[一-鿿]{2,}", normalized)
    english_tokens = re.findall(r"[a-z]{3,}", normalized)
    stopwords = {
        "帮我", "给我", "看看", "一下", "一个", "一份", "怎么", "如何", "关于", "相关", "推荐", "安排", "规划",
        "please", "help", "with", "about", "need", "want", "plan", "trip",
    }
    return {token for token in chinese_tokens + english_tokens if token not in stopwords}


def _should_carry_context(question: str, previous_summary: dict[str, Any]) -> bool:
    normalized = str(question or "").strip()
    if not normalized or not isinstance(previous_summary, dict) or not previous_summary:
        return False
    if _is_context_dependent_query(normalized):
        return True

    previous_question = str(previous_summary.get("rewritten_question", "")).strip() or str(previous_summary.get("effective_question", "")).strip()
    previous_needs = previous_summary.get("detected_needs", [])
    previous_text = "\n".join([previous_question] + [str(item).strip() for item in previous_needs if str(item).strip()])

    current_tokens = _extract_topic_tokens(normalized)
    previous_tokens = _extract_topic_tokens(previous_text)
    if not current_tokens or not previous_tokens:
        return False

    overlap = current_tokens & previous_tokens
    overlap_ratio = len(overlap) / max(len(current_tokens), 1)

    if overlap_ratio >= 0.6:
        return True
    if len(normalized) <= 10 and overlap_ratio >= 0.34:
        return True
    return False


def _build_continuation_question(question: str, previous_summary: dict[str, Any]) -> str:
    if not question or not isinstance(previous_summary, dict) or not previous_summary:
        return question
    previous_question = str(previous_summary.get("rewritten_question", "")).strip() or str(previous_summary.get("effective_question", "")).strip()
    previous_needs = previous_summary.get("detected_needs", [])
    previous_context = str(previous_summary.get("context_summary", "")).strip()
    if not previous_question and not previous_needs and not previous_context:
        return question
    parts = ["这是基于上一轮任务的继续补充，请结合之前已经确定的信息一起理解。"]
    if previous_question:
        parts.append(f"上一轮问题：{previous_question}")
    if previous_needs:
        parts.append(f"上一轮需求：{json.dumps(previous_needs, ensure_ascii=False)}")
    if previous_context:
        parts.append(f"上一轮背景：{previous_context}")
    parts.append(f"本轮补充或修改：{question}")
    return "\n".join(parts)


def _is_confirmation_reply(text: str) -> bool:
    normalized = text.strip().lower()
    if not normalized:
        return False
    return any(keyword in normalized for keyword in _CONFIRM_REPLY_KEYWORDS)


def _normalize_confirmation_text(text: str) -> str:
    return re.sub(r"\s+", " ", str(text).strip().lower())


def _build_confirmation_signature(
    *,
    route: str,
    rewritten_question: str,
    detected_needs: list[str] | None = None,
) -> str:
    normalized_needs = [
        _normalize_confirmation_text(item)
        for item in (detected_needs or [])
        if _normalize_confirmation_text(item)
    ]
    payload = {
        "route": _normalize_confirmation_text(route),
        "rewritten_question": _normalize_confirmation_text(rewritten_question),
        "detected_needs": normalized_needs,
    }
    if not payload["rewritten_question"] and not normalized_needs:
        return ""
    return json.dumps(payload, ensure_ascii=False, sort_keys=True)


def _extract_last_confirmed_signature(history: list[dict[str, Any]] | None) -> str:
    for item in reversed(history or []):
        if str(item.get("role", "")).strip().lower() != "assistant":
            continue
        metadata = item.get("metadata")
        if not isinstance(metadata, dict):
            continue
        final_summary = metadata.get("final_summary")
        if not isinstance(final_summary, dict):
            continue
        signature = str(final_summary.get("confirmation_signature", "")).strip()
        if signature:
            return signature
    return ""


def _resolve_effective_question(state: ChatGraphState) -> tuple[str, bool, dict[str, Any], dict[str, Any], bool]:
    question = str(state.get("question", "")).strip()
    history = state.get("history")
    pending_confirmation = _extract_pending_confirmation(history)
    previous_task_summary = _extract_latest_final_summary(history)
    if pending_confirmation and _is_confirmation_reply(question):
        original_question = str(pending_confirmation.get("original_question", "")).strip()
        if original_question:
            return original_question, True, pending_confirmation, previous_task_summary, True
    if previous_task_summary and _should_carry_context(question, previous_task_summary):
        return _build_continuation_question(question, previous_task_summary), False, pending_confirmation, previous_task_summary, True
    return question, False, pending_confirmation, {}, False


def _last_user_query(state: ChatGraphState) -> str:
    rewritten = str(state.get("rewritten_question", "")).strip()
    if rewritten:
        return rewritten
    effective = str(state.get("effective_question", "")).strip()
    if effective:
        return effective
    return str(state.get("question", "")).strip()


def _is_lodging_query(text: str) -> bool:
    normalized = text.strip().lower()
    if not normalized:
        return False
    lodging_keywords = {"住宿", "住哪里", "酒店", "宾馆", "民宿", "旅店", "客栈", "青旅", "旅馆"}
    return any(keyword.lower() in normalized for keyword in lodging_keywords)


def _is_dining_query(text: str) -> bool:
    normalized = text.strip().lower()
    if not normalized:
        return False
    dining_keywords = {"饭店", "餐馆", "餐厅", "美食", "吃饭", "就餐", "小吃", "火锅", "早餐", "午餐", "晚餐"}
    return any(keyword.lower() in normalized for keyword in dining_keywords)


def _is_scenic_query(text: str) -> bool:
    normalized = text.strip().lower()
    if not normalized:
        return False
    scenic_keywords = {"景点", "景区", "景色", "游玩", "打卡", "公园", "博物馆", "古镇", "海边", "寺庙"}
    return any(keyword.lower() in normalized for keyword in scenic_keywords)


def _extract_json(content: str) -> dict[str, Any]:
    text = _strip_think_tags(content).strip()
    if text.startswith("```"):
        lines = text.splitlines()
        if len(lines) >= 3:
            text = "\n".join(lines[1:-1]).strip()

    start = text.find("{")
    end = text.rfind("}")
    if start == -1 or end == -1 or end < start:
        raise ValueError("No JSON object found in response")
    return json.loads(text[start: end + 1])


def _safe_json_llm(*, system_prompt: str, user_prompt: str, history: list[dict[str, str]] | None) -> dict[str, Any]:
    llm = _build_llm()
    prompt = ChatPromptTemplate.from_messages(
        [
            ("system", system_prompt),
            MessagesPlaceholder(variable_name="chat_history"),
            ("user", user_prompt),
        ]
    )
    response = llm.invoke(
        prompt.format_messages(
            chat_history=_normalize_history(history),
        )
    )
    content = str(getattr(response, "content", response))
    return _extract_json(content)


def _combined_route_text(state: ChatGraphState) -> str:
    parts = [
        str(state.get("question", "")).strip(),
        str(state.get("effective_question", "")).strip(),
        str(state.get("rewritten_question", "")).strip(),
    ]
    return "\n".join(part for part in parts if part).lower()



def _contains_any(text: str, keywords: set[str] | list[str] | tuple[str, ...]) -> bool:
    return any(str(keyword).strip().lower() in text for keyword in keywords if str(keyword).strip())



def _deterministic_route(state: ChatGraphState) -> RouteType | None:
    text = _combined_route_text(state)
    if not text:
        return None

    ticket_keywords = set(_TICKET_KEYWORDS) | {
        "高铁",
        "动车",
        "火车票",
        "高铁票",
        "动车票",
        "车票",
        "车次",
        "余票",
        "抢票",
        "改签",
        "候补",
    }
    strong_roadmap_keywords = {
        "路线",
        "线路",
        "行程",
        "攻略",
        "规划",
        "游玩顺序",
        "排个一天路线",
        "排一个一天路线",
        "一日游路线",
        "两天一夜行程",
        "itinerary",
    }
    weak_roadmap_keywords = set(_ROADMAP_KEYWORDS) | {
        "安排",
        "一日游",
        "两天一夜",
        "半日游",
        "怎么玩",
    }
    explicit_lodging_keywords = {
        "住哪里",
        "住哪",
        "住哪个区",
        "住哪里比较方便",
        "哪个区域方便",
        "住宿建议",
        "酒店建议",
        "酒店推荐",
        "住宿推荐",
        "附近住宿",
        "住宿区域",
        "酒店区域",
    }
    other_keywords = {
        "预算",
        "多少钱",
        "费用",
        "花费",
        "老人",
        "带老人",
        "亲子",
        "带孩子",
        "轻松",
        "省心",
        "出行建议",
        "旅行建议",
    }

    has_ticket = _contains_any(text, ticket_keywords)
    has_strong_roadmap = _contains_any(text, strong_roadmap_keywords)
    has_weak_roadmap = _contains_any(text, weak_roadmap_keywords)
    has_explicit_lodging = _contains_any(text, explicit_lodging_keywords)
    has_broad_rag = _contains_any(text, _RAG_KEYWORDS)
    has_other = _contains_any(text, other_keywords)

    if has_ticket:
        return "ticket"
    if has_explicit_lodging and not has_strong_roadmap:
        return "rag"
    if has_other and not has_strong_roadmap and not has_explicit_lodging:
        return "other"
    if has_strong_roadmap:
        return "roadmap"
    if has_weak_roadmap and has_other and not has_explicit_lodging:
        return "other"
    if has_weak_roadmap:
        return "roadmap"
    if has_explicit_lodging or has_broad_rag:
        return "rag"
    if has_other:
        return "other"
    return None


def _rule_based_prior_routes(state: ChatGraphState) -> list[str]:
    text = _combined_route_text(state)

    hits: list[str] = []
    deterministic_route = _deterministic_route(state)
    if deterministic_route:
        hits.append(deterministic_route)
    if "ticket" not in hits and any(keyword.lower() in text for keyword in _TICKET_KEYWORDS):
        hits.append("ticket")
    if "roadmap" not in hits and any(keyword.lower() in text for keyword in _ROADMAP_KEYWORDS):
        hits.append("roadmap")
    if "rag" not in hits and any(keyword.lower() in text for keyword in _RAG_KEYWORDS):
        hits.append("rag")
    if not hits:
        hits.append("other")
    return hits


def _extract_needs(state: ChatGraphState) -> list[str]:
    question = _last_user_query(state)
    if not question:
        return []

    try:
        parsed = _safe_json_llm(
            system_prompt=(
                "你是需求提取助手。"
                "请根据用户当前问题、规则先验路由和历史上下文，提取用户明确表达的核心需求。"
                "避免凭空补充未提及的信息。"
                "只输出 JSON，对象中必须包含字段 extracted_needs，值为字符串数组。"
                "不要输出额外解释。"
            ),
            user_prompt=(
                f"规则先验：{json.dumps(state.get('prior_routes', []), ensure_ascii=False)}\n"
                f"上一轮任务摘要：{_serialize_previous_task_summary(state.get('previous_task_summary', {}))}\n"
                f"当前问题：{question}"
            ),
            history=state.get("history"),
        )
    except Exception:
        return [question]

    needs = parsed.get("extracted_needs", []) if isinstance(parsed, dict) else []
    normalized = [str(item).strip() for item in needs if str(item).strip()]
    return normalized or [question]


def _validate_needs(state: ChatGraphState, extracted_needs: list[str]) -> list[str]:
    question = _last_user_query(state)
    if not extracted_needs:
        return [question] if question else []

    try:
        parsed = _safe_json_llm(
            system_prompt=(
                "你是需求校验助手。"
                "请检查候选需求是否真实覆盖了用户问题，删除重复、歧义或明显不成立的项。"
                "保留用户明确提出、且后续回答需要覆盖的需求。"
                "只输出 JSON，对象中必须包含字段 validated_needs，值为字符串数组。"
                "不要输出额外解释。"
            ),
            user_prompt=(
                f"规则先验：{json.dumps(state.get('prior_routes', []), ensure_ascii=False)}\n"
                f"上一轮任务摘要：{_serialize_previous_task_summary(state.get('previous_task_summary', {}))}\n"
                f"当前问题：{question}\n"
                f"候选需求：{json.dumps(extracted_needs, ensure_ascii=False)}"
            ),
            history=state.get("history"),
        )
    except Exception:
        return extracted_needs

    needs = parsed.get("validated_needs", []) if isinstance(parsed, dict) else []
    normalized = [str(item).strip() for item in needs if str(item).strip()]
    return normalized or extracted_needs


def _fill_missing_needs(state: ChatGraphState, validated_needs: list[str]) -> list[str]:
    question = _last_user_query(state)
    if not question:
        return validated_needs

    try:
        parsed = _safe_json_llm(
            system_prompt=(
                "你是需求补全助手。"
                "请在不改变用户原意的前提下，补齐那些虽然用户没有逐字重复，但结合上下文显然属于同一任务的必要需求。"
                "如果没有可补充内容，就原样返回。"
                "只输出 JSON，对象中必须包含字段 completed_needs，值为字符串数组。"
                "不要输出额外解释。"
            ),
            user_prompt=(
                f"规则先验：{json.dumps(state.get('prior_routes', []), ensure_ascii=False)}\n"
                f"上一轮任务摘要：{_serialize_previous_task_summary(state.get('previous_task_summary', {}))}\n"
                f"当前问题：{question}\n"
                f"已校验需求：{json.dumps(validated_needs, ensure_ascii=False)}"
            ),
            history=state.get("history"),
        )
    except Exception:
        return validated_needs

    needs = parsed.get("completed_needs", []) if isinstance(parsed, dict) else []
    normalized = [str(item).strip() for item in needs if str(item).strip()]
    return normalized or validated_needs


def _summarize_context(state: ChatGraphState, detected_needs: list[str]) -> str:
    question = str(state.get("question", "")).strip()
    try:
        parsed = _safe_json_llm(
            system_prompt=(
                "你是上下文总结助手。"
                "请把和当前任务相关的历史背景压缩成简短摘要，便于后续改写问题和调度路由。"
                "只保留对本轮任务真正有帮助的信息。"
                "只输出 JSON，对象中必须包含字段 context_summary，值为字符串。"
                "不要输出额外解释。"
            ),
            user_prompt=(
                f"规则先验：{json.dumps(state.get('prior_routes', []), ensure_ascii=False)}\n"
                f"上一轮任务摘要：{_serialize_previous_task_summary(state.get('previous_task_summary', {}))}\n"
                f"当前问题：{question}\n"
                f"当前需求：{json.dumps(detected_needs, ensure_ascii=False)}"
            ),
            history=state.get("history"),
        )
    except Exception:
        parsed = {}

    summarized = str(parsed.get("context_summary", "")).strip() if isinstance(parsed, dict) else ""
    if summarized:
        return summarized
    previous_summary = state.get("previous_task_summary", {})
    return str(previous_summary.get("context_summary", "")).strip() if isinstance(previous_summary, dict) else ""


def _rewrite_question(state: ChatGraphState, detected_needs: list[str], context_summary: str) -> str:
    question = str(state.get("question", "")).strip()
    try:
        parsed = _safe_json_llm(
            system_prompt=(
                "你是问题改写助手。"
                "请基于用户原问题、需求清单和历史背景，把问题改写成更完整、更明确、适合后续路由和执行的版本。"
                "改写时要保留用户真实意图，不要凭空添加结论。"
                "rewritten_question 必须是一句可直接交给下游工具或助手处理的完整问题。"
                "只输出 JSON，对象中必须包含字段 rewritten_question，值为字符串。"
                "不要输出额外解释。"
            ),
            user_prompt=(
                f"规则先验：{json.dumps(state.get('prior_routes', []), ensure_ascii=False)}\n"
                f"上一轮任务摘要：{_serialize_previous_task_summary(state.get('previous_task_summary', {}))}\n"
                f"当前问题：{question}\n"
                f"当前需求：{json.dumps(detected_needs, ensure_ascii=False)}\n"
                f"历史背景：{context_summary or '无'}"
            ),
            history=state.get("history"),
        )
    except Exception:
        return question

    rewritten = str(parsed.get("rewritten_question", "")).strip() if isinstance(parsed, dict) else ""
    return rewritten or question


def _extract_latest_plan_state_from_history(history: list[dict[str, Any]] | None) -> dict[str, Any]:
    for item in reversed(history or []):
        if str(item.get("role", "")).strip().lower() != "assistant":
            continue
        metadata = item.get("metadata")
        if not isinstance(metadata, dict):
            continue
        final_summary = metadata.get("final_summary")
        if not isinstance(final_summary, dict):
            continue
        plan_state = final_summary.get("plan_state")
        if isinstance(plan_state, dict) and plan_state:
            return {
                "current_plan_id": plan_state.get("current_plan_id"),
                "plan_draft": dict(plan_state.get("plan_draft", {}) or {}),
                "locked_fields": dict(plan_state.get("locked_fields", {}) or {}),
                "candidate_options": list(plan_state.get("candidate_options", []) or []),
                "ready_for_final_confirmation": bool(plan_state.get("ready_for_final_confirmation")),
                "final_confirmation_payload": dict(plan_state.get("final_confirmation_payload", {}) or {}),
            }
    return {
        "current_plan_id": None,
        "plan_draft": {},
        "locked_fields": {},
        "candidate_options": [],
        "ready_for_final_confirmation": False,
        "final_confirmation_payload": {},
    }


def _is_travel_plan_query(state: ChatGraphState) -> bool:
    combined = "\n".join(
        [
            str(state.get("question", "")).strip(),
            str(state.get("effective_question", "")).strip(),
            str(state.get("rewritten_question", "")).strip(),
        ]
    ).lower()
    if any(keyword.lower() in combined for keyword in _TRAVEL_PLAN_KEYWORDS):
        return True
    prior_routes = {str(item).strip().lower() for item in state.get("prior_routes", [])}
    if prior_routes & {"ticket", "roadmap"}:
        return True
    previous = _extract_latest_plan_state_from_history(state.get("history"))
    return bool(previous.get("plan_draft") or previous.get("locked_fields") or previous.get("current_plan_id"))


FIELD_LABELS = {
    "origin": "\u51fa\u53d1\u5730",
    "destination": "\u76ee\u7684\u5730",
    "departure_date": "\u51fa\u53d1\u65e5\u671f",
    "return_date": "\u8fd4\u7a0b\u65e5\u671f",
    "travelers": "\u51fa\u884c\u4eba\u6570",
    "budget": "\u9884\u7b97",
    "title": "\u8ba1\u5212\u6807\u9898",
    "route": "\u8ba1\u5212\u7c7b\u578b",
    "plan_summary": "\u884c\u7a0b\u6458\u8981",
    "ticket_option": "\u5df2\u9501\u5b9a\u8f66\u7968",
    "hotel_option": "\u5df2\u9501\u5b9a\u4f4f\u5bbf",
    "plan_version": "\u5df2\u9501\u5b9a\u65b9\u6848",
    "route_option": "\u5df2\u9501\u5b9a\u8def\u7ebf",
    "scenic_option": "\u5df2\u9501\u5b9a\u666f\u70b9",
}


DEPENDENT_LOCKS = {
    "destination": ["route_option", "hotel_option", "scenic_option", "ticket_option", "plan_version"],
    "departure_date": ["route_option", "ticket_option"],
    "return_date": ["ticket_option"],
    "budget": ["hotel_option", "plan_version"],
    "travelers": ["hotel_option", "plan_version"],
}


def _extract_plan_updates(question: str) -> tuple[dict[str, Any], list[str], list[str]]:
    text = str(question or "").strip()
    lowered = text.lower()
    updates: dict[str, Any] = {}
    removed_fields: list[str] = []
    changed_fields: list[str] = []

    if not text:
        return updates, removed_fields, changed_fields

    city_patterns = {
        "origin": [
            "从([一-鿿A-Za-z]{2,20})出发",
            r"出发地[是为:]?\s*([一-鿿A-Za-z]{2,20})",
        ],
        "destination": [
            "去([一-鿿A-Za-z]{2,20})",
            "到([一-鿿A-Za-z]{2,20})",
            r"目的地[是为:]?\s*([一-鿿A-Za-z]{2,20})",
        ],
    }
    for field_name, patterns in city_patterns.items():
        for pattern in patterns:
            match = re.search(pattern, text)
            if match:
                updates[field_name] = match.group(1).strip()
                changed_fields.append(field_name)
                break

    date_matches = re.findall(r"(\d{4}-\d{1,2}-\d{1,2}|\d{1,2}月\d{1,2}日|今天|明天|后天)", text)
    if date_matches:
        updates["departure_date"] = date_matches[0]
        changed_fields.append("departure_date")
    if len(date_matches) >= 2:
        updates["return_date"] = date_matches[1]
        changed_fields.append("return_date")

    traveler_match = re.search(r"(\d+)\s*(人|位)", text)
    if traveler_match:
        updates["travelers"] = int(traveler_match.group(1))
        changed_fields.append("travelers")

    budget_match = re.search(r"(预算|人均|花费|控制在)\s*([0-9一二三四五六七八九十百千万]+\s*(元|块|w|万)?)", text, flags=re.IGNORECASE)
    if budget_match:
        updates["budget"] = budget_match.group(2).strip()
        changed_fields.append("budget")

    remove_terms = {
        "origin": ["不要出发地", "删除出发地", "去掉出发地", "取消出发地"],
        "destination": ["不要目的地", "删除目的地", "去掉目的地", "取消目的地"],
        "departure_date": ["不要出发日期", "删除出发日期"],
        "return_date": ["不要返程日期", "删除返程日期"],
        "travelers": ["不要出行人数", "删除出行人数"],
        "budget": ["不要预算", "删除预算", "去掉预算"],
    }
    for field_name, patterns in remove_terms.items():
        if any(term in text for term in patterns):
            removed_fields.append(field_name)

    changed_fields = [item for item in changed_fields if item not in removed_fields]
    return updates, removed_fields, changed_fields


def _extract_candidate_options_from_history(history: list[dict[str, Any]] | None) -> list[dict[str, Any]]:
    for item in reversed(history or []):
        if str(item.get("role", "")).strip().lower() != "assistant":
            continue
        metadata = item.get("metadata")
        if not isinstance(metadata, dict):
            continue
        final_summary = metadata.get("final_summary")
        if not isinstance(final_summary, dict):
            continue
        options = final_summary.get("candidate_options")
        if isinstance(options, list) and options:
            return [option for option in options if isinstance(option, dict)]
        plan_state = final_summary.get("plan_state")
        if isinstance(plan_state, dict):
            plan_options = plan_state.get("candidate_options")
            if isinstance(plan_options, list) and plan_options:
                return [option for option in plan_options if isinstance(option, dict)]
    return []


def _normalize_option_match_text(text: str) -> str:
    return re.sub(r"[^\w一-鿿]+", "", str(text or "").lower())


def _guess_candidate_field_from_text(text: str, route: str) -> str:
    lowered = str(text or "").lower()
    if any(keyword.lower() in lowered for keyword in _HOTEL_OBJECT_KEYWORDS) or any(token in lowered for token in ("hotel", "stay", "lodging")):
        return "hotel_option"
    if any(keyword.lower() in lowered for keyword in _SPECIFIC_TICKET_OBJECT_KEYWORDS) or any(token in lowered for token in ("ticket", "train", "flight")):
        return "ticket_option"
    if any(token in lowered for token in ("路线", "线路", "景点", "行程", "攻略", "route", "plan", "itinerary", "scenic")):
        return "plan_version"
    if route == "ticket":
        return "ticket_option"
    return "plan_version"


def _detect_selection_intent(question: str, history: list[dict[str, Any]]) -> tuple[str, dict[str, Any], list[dict[str, Any]]]:
    normalized = str(question or "").strip()
    candidate_options = _extract_candidate_options_from_history(history)
    if not normalized or not candidate_options:
        return "", {}, candidate_options

    normalized_match = _normalize_option_match_text(normalized)
    has_lock_keyword = any(keyword in normalized for keyword in _SELECTION_LOCK_KEYWORDS)
    has_index_keyword = bool(re.search(r"第\s*([一二三四五六七八九十123456789])|([1-9])|最后一个|上一个|前一个", normalized))
    if not has_lock_keyword and not has_index_keyword:
        for option in candidate_options:
            label = _normalize_option_match_text(option.get("label", ""))
            if label and (label in normalized_match or normalized_match in label):
                has_lock_keyword = True
                break
    if not has_lock_keyword and not has_index_keyword:
        return "", {}, candidate_options

    selected_index: int | None = None
    digit_match = re.search(r"([1-9])", normalized)
    if digit_match:
        selected_index = int(digit_match.group(1)) - 1
    elif "第一个" in normalized or "第一" in normalized:
        selected_index = 0
    elif "第二个" in normalized or "第二" in normalized:
        selected_index = 1
    elif "第三个" in normalized or "第三" in normalized:
        selected_index = 2
    elif "最后一个" in normalized:
        selected_index = len(candidate_options) - 1
    elif "上一个" in normalized or "前一个" in normalized:
        selected_index = max(len(candidate_options) - 2, 0)
    elif len(candidate_options) == 1:
        selected_index = 0

    if selected_index is None:
        for idx, option in enumerate(candidate_options):
            label = _normalize_option_match_text(option.get("label", ""))
            value = _normalize_option_match_text(option.get("value", ""))
            if (label and label in normalized_match) or (value and value in normalized_match):
                selected_index = idx
                break

    if selected_index is None or selected_index < 0 or selected_index >= len(candidate_options):
        return "lock_candidate_ambiguous", {}, candidate_options
    return "lock_candidate", candidate_options[selected_index], candidate_options


def _merge_plan_draft(
    base_draft: dict[str, Any],
    updates: dict[str, Any],
    removed_fields: list[str],
    locked_fields: dict[str, Any],
) -> dict[str, Any]:
    draft = dict(base_draft or {})
    for field_name in removed_fields:
        draft.pop(field_name, None)
    for field_name, value in (updates or {}).items():
        if field_name in locked_fields and locked_fields.get(field_name) == value:
            continue
        draft[field_name] = value
    return draft


def _apply_locked_fields(plan_draft: dict[str, Any], locked_fields: dict[str, Any]) -> dict[str, Any]:
    merged = dict(plan_draft or {})
    for field_name, value in (locked_fields or {}).items():
        merged[field_name] = value
    return merged


def _normalize_selection_field(route: str, selection_target: dict[str, Any], question: str) -> str:
    field_name = str(selection_target.get("field_name", "")).strip()
    if field_name:
        return field_name
    lowered = str(question or "").lower()
    if any(keyword in lowered for keyword in _HOTEL_OBJECT_KEYWORDS):
        return "hotel_option"
    if any(keyword in lowered for keyword in _SPECIFIC_TICKET_OBJECT_KEYWORDS):
        return "ticket_option"
    if route == "ticket":
        return "ticket_option"
    if route == "roadmap":
        return "plan_version"
    return "plan_version"


def _compute_plan_completeness(
    plan_draft: dict[str, Any],
    locked_fields: dict[str, Any],
    route: str,
) -> dict[str, Any]:
    required_base_fields = ["destination", "departure_date"]
    missing_fields = [field_name for field_name in required_base_fields if not str(plan_draft.get(field_name, "")).strip()]

    required_locked_fields: list[str] = []
    if route == "ticket":
        required_locked_fields.append("ticket_option")
    elif route == "roadmap" and any(keyword in str(plan_draft) for keyword in ("酒店", "住宿", "景点", "路线")):
        required_locked_fields.append("plan_version")

    missing_locked_fields = [field_name for field_name in required_locked_fields if field_name not in locked_fields]
    return {
        "is_complete": not missing_fields and not missing_locked_fields,
        "missing_fields": missing_fields,
        "missing_locked_fields": missing_locked_fields,
    }


def _build_final_confirmation_payload(
    plan_draft: dict[str, Any],
    locked_fields: dict[str, Any],
    route: str,
    answer: str = "",
) -> dict[str, Any]:
    payload = _apply_locked_fields(plan_draft, locked_fields)
    payload["route"] = route
    payload["locked_items"] = dict(locked_fields or {})
    payload["plan_summary"] = str(payload.get("plan_summary") or answer or "").strip()
    payload["title"] = str(payload.get("title") or payload.get("destination") or "\u51fa\u884c\u8ba1\u5212").strip()
    return payload


def _is_final_confirmation_reply(question: str, history: list[dict[str, Any]] | None) -> bool:
    normalized = str(question or "").strip()
    if not normalized:
        return False
    if any(keyword in normalized for keyword in _FINAL_PLAN_CONFIRM_KEYWORDS):
        return True
    if normalized not in {"确认", "确定", "好的", "可以"}:
        return False
    for item in reversed(history or []):
        if str(item.get("role", "")).strip().lower() != "assistant":
            continue
        metadata = item.get("metadata")
        if not isinstance(metadata, dict):
            continue
        final_summary = metadata.get("final_summary")
        if not isinstance(final_summary, dict):
            continue
        if isinstance(final_summary.get("plan_state"), dict) and final_summary["plan_state"].get("ready_for_final_confirmation"):
            return True
        break
    return False


def _build_plan_confirmation_text(final_payload: dict[str, Any]) -> str:
    locked_items = final_payload.get("locked_items", {}) if isinstance(final_payload, dict) else {}
    label_map = {
        "origin": "出发地",
        "destination": "目的地",
        "departure_date": "出发日期",
        "return_date": "返程日期",
        "travelers": "出行人数",
        "budget": "预算",
    }
    lines = ["我已经整理好这份最终出行确认单，请核对："]
    for field_name in ("origin", "destination", "departure_date", "return_date", "travelers", "budget"):
        value = final_payload.get(field_name)
        if value not in (None, "", [], {}):
            lines.append(f"- {label_map.get(field_name, field_name)}：{value}")
    if isinstance(locked_items, dict) and locked_items:
        lines.append("- 已锁定方案：")
        for key, value in locked_items.items():
            field_label = FIELD_LABELS.get(str(key), str(key))
            lines.append(f"  {field_label}?{value}")
    summary = str(final_payload.get("plan_summary", "")).strip()
    if summary:
        lines.append(f"- 行程摘要：{summary}")
    lines.append("回复“确认行程”后，我会保存到“我的出行计划”。")
    return "\n".join(lines)


def _build_plan_state_output(state: ChatGraphState) -> dict[str, Any]:
    return {
        "current_plan_id": state.get("current_plan_id"),
        "plan_draft": dict(state.get("plan_draft", {}) or {}),
        "locked_fields": dict(state.get("locked_fields", {}) or {}),
        "candidate_options": list(state.get("candidate_options", []) or []),
        "ready_for_final_confirmation": bool(state.get("ready_for_final_confirmation")),
        "final_confirmation_payload": dict(state.get("final_confirmation_payload", {}) or {}),
    }


def _extract_candidate_options_from_answer(answer: str, route: str) -> list[dict[str, Any]]:
    if not answer:
        return []

    options: list[dict[str, Any]] = []
    current_field = "ticket_option" if route == "ticket" else "plan_version"
    seen_labels: set[str] = set()
    heading_pattern = re.compile(r"^(酒店|住宿|民宿|宾馆|车票|火车票|高铁票|机票|路线|线路|景点|行程|方案|推荐方案)")
    bullet_pattern = re.compile(r"^([\-*]|\d+[\.\u3001]|[\u4e00\u4e8c\u4e09\u56db\u4e94\u516d\u4e03\u516b\u4e5d\u5341]+[\.\u3001])")
    bullet_cleanup_pattern = re.compile(r"^([\-*]|\d+[\.\u3001]|[\u4e00\u4e8c\u4e09\u56db\u4e94\u516d\u4e03\u516b\u4e5d\u5341]+[\.\u3001])\s*")

    for raw_line in answer.splitlines():
        stripped = raw_line.strip()
        if not stripped:
            continue

        heading_text = re.sub(r"^[#>*\-\s]+", "", stripped)
        heading_match = heading_pattern.match(heading_text)
        if heading_match:
            current_field = _guess_candidate_field_from_text(heading_match.group(1), route)
            continue

        if not bullet_pattern.match(stripped):
            continue

        cleaned = bullet_cleanup_pattern.sub("", stripped).strip()
        cleaned = re.sub(r"^\*\*(.+?)\*\*$", r"\1", cleaned)
        if len(cleaned) < 4:
            continue

        field_name = _guess_candidate_field_from_text(cleaned, route)
        if field_name == "plan_version":
            field_name = current_field

        label = cleaned[:120]
        if label in seen_labels:
            continue
        seen_labels.add(label)
        options.append(
            {
                "index": len(options) + 1,
                "field_name": field_name,
                "label": label,
                "value": cleaned,
                "source": "answer",
            }
        )
        if len(options) >= 8:
            break
    return options


def preprocess_query(state: ChatGraphState) -> dict[str, Any]:
    effective_question, confirmed, pending_confirmation, previous_task_summary, carry_context = _resolve_effective_question(state)
    fallback_question = effective_question
    prior_routes = _rule_based_prior_routes({**state, "question": effective_question})
    working_state: ChatGraphState = dict(state)
    working_state["question"] = effective_question
    working_state["effective_question"] = effective_question
    working_state["prior_routes"] = prior_routes
    working_state["previous_task_summary"] = previous_task_summary if carry_context else {}
    working_state["history"] = state.get("history") if carry_context else []
    extracted_needs = _extract_needs(working_state)
    validated_needs = _validate_needs(working_state, extracted_needs)
    completed_needs = _fill_missing_needs(working_state, validated_needs)
    normalized_needs = completed_needs or ([fallback_question] if fallback_question else [])
    normalized_summary = _summarize_context(working_state, normalized_needs)
    normalized_question = _rewrite_question(working_state, normalized_needs, normalized_summary)

    return {
        "effective_question": effective_question,
        "confirmed": confirmed,
        "pending_confirmation": pending_confirmation,
        "previous_task_summary": previous_task_summary if carry_context else {},
        "carry_context": carry_context,
        "status": "completed",
        "prior_routes": prior_routes,
        "detected_needs": normalized_needs,
        "context_summary": normalized_summary,
        "rewritten_question": normalized_question,
    }


async def sync_plan_state(state: ChatGraphState) -> dict[str, Any]:
    if not _is_travel_plan_query(state):
        return {
            "selection_action": "",
            "selection_target": {},
            "plan_confirmation_completed": False,
        }

    from backend.app.core.database import AsyncSessionLocal
    from backend.app.crued.travel_plan import confirm_plan, get_or_create_active_plan, lock_plan_field, unlock_plan_field, update_plan_draft

    route_hint = next((item for item in state.get("prior_routes", []) if item in {"ticket", "roadmap", "rag", "other"}), "other")
    question = _last_user_query(state)
    previous_locked_fields = dict(state.get("locked_fields", {}) or {})
    selection_action, selection_target, candidate_options = _detect_selection_intent(question, state.get("history", []))
    updates, removed_fields, changed_fields = _extract_plan_updates(question)

    locked_fields = dict(previous_locked_fields)
    for field_name in removed_fields + changed_fields:
        locked_fields.pop(field_name, None)
        for dependent_field in DEPENDENT_LOCKS.get(field_name, []):
            locked_fields.pop(dependent_field, None)

    if selection_action == "lock_candidate" and selection_target:
        selected_field = _normalize_selection_field(route_hint, selection_target, question)
        locked_fields[selected_field] = selection_target.get("value") or selection_target.get("label") or selection_target

    merged_draft = _merge_plan_draft(state.get("plan_draft", {}), updates, removed_fields, locked_fields)
    effective_draft = _apply_locked_fields(merged_draft, locked_fields)

    async with AsyncSessionLocal() as db:
        plan = await get_or_create_active_plan(db, int(state.get("user_id") or 0))
        await update_plan_draft(db, plan.id, merged_draft)

        previous_locked_names = set(previous_locked_fields)
        current_locked_names = set(locked_fields)
        for field_name in previous_locked_names - current_locked_names:
            await unlock_plan_field(db, plan.id, field_name)
        for field_name, field_value in locked_fields.items():
            if previous_locked_fields.get(field_name) != field_value:
                await lock_plan_field(db, plan.id, field_name, field_value, question)

        if _is_final_confirmation_reply(question, state.get("history")) and state.get("ready_for_final_confirmation"):
            final_payload = dict(state.get("final_confirmation_payload", {}) or {})
            if final_payload:
                confirmed_plan = await confirm_plan(db, plan.id, final_payload)
                return {
                    "current_plan_id": confirmed_plan.id,
                    "plan_draft": effective_draft,
                    "locked_fields": locked_fields,
                    "candidate_options": candidate_options or state.get("candidate_options", []),
                    "selection_action": selection_action,
                    "selection_target": selection_target,
                    "ready_for_final_confirmation": False,
                    "final_confirmation_payload": dict(confirmed_plan.final_confirmed_data or final_payload),
                    "plan_confirmation_completed": True,
                    "answer": f"出行计划已保存，计划 ID: {confirmed_plan.id}",
                    "answer_source": "travel_plan_confirm",
                    "status": "completed",
                    "verification": {
                        "is_complete": True,
                        "covered_needs": state.get("detected_needs", []),
                        "missing_needs": [],
                        "unsupported_needs": [],
                        "answer_source": "travel_plan_confirm",
                    },
                }

    completeness = _compute_plan_completeness(effective_draft, locked_fields, route_hint)
    return {
        "current_plan_id": plan.id,
        "plan_draft": effective_draft,
        "locked_fields": locked_fields,
        "candidate_options": candidate_options or state.get("candidate_options", []),
        "selection_action": selection_action,
        "selection_target": selection_target,
        "ready_for_final_confirmation": bool(completeness.get("is_complete") and locked_fields),
        "final_confirmation_payload": dict(state.get("final_confirmation_payload", {}) or {}),
        "plan_confirmation_completed": False,
    }


def _after_sync_plan_state(state: ChatGraphState) -> str:
    if bool(state.get("plan_confirmation_completed")):
        return "summarize_result"
    return "agent_manager"


async def travel_plan_finalize(state: ChatGraphState) -> dict[str, Any]:
    if not _is_travel_plan_query(state) or bool(state.get("plan_confirmation_completed")):
        return {}

    from backend.app.core.database import AsyncSessionLocal
    from backend.app.crued.travel_plan import mark_plan_ready_for_confirmation

    route_name = str(state.get("route", "")).strip() or next((item for item in state.get("prior_routes", []) if item in {"ticket", "roadmap", "rag", "other"}), "other")
    answer = str(state.get("answer", "")).strip()
    candidate_options = _extract_candidate_options_from_answer(answer, route_name) or list(state.get("candidate_options", []) or [])
    locked_fields = dict(state.get("locked_fields", {}) or {})
    effective_draft = _apply_locked_fields(state.get("plan_draft", {}), locked_fields)
    completeness = _compute_plan_completeness(effective_draft, locked_fields, route_name)
    ready = bool(completeness.get("is_complete") and locked_fields)
    final_payload: dict[str, Any] = {}

    if ready:
        final_payload = _build_final_confirmation_payload(effective_draft, locked_fields, route_name, answer)
        plan_id = int(state.get("current_plan_id") or 0)
        if plan_id:
            async with AsyncSessionLocal() as db:
                await mark_plan_ready_for_confirmation(db, plan_id, final_payload)
        if final_payload != dict(state.get("final_confirmation_payload", {}) or {}):
            confirmation_text = _build_plan_confirmation_text(final_payload)
            answer = f"{answer}\n\n{confirmation_text}" if answer else confirmation_text
    elif candidate_options:
        option_tip = "如需锁定某个候选方案，请直接回复“我选第1个方案”或点击下方选择按钮。"
        if option_tip not in answer:
            answer = f"{answer}\n\n{option_tip}" if answer else option_tip

    return {
        "answer": answer,
        "candidate_options": candidate_options,
        "ready_for_final_confirmation": ready,
        "final_confirmation_payload": final_payload or dict(state.get("final_confirmation_payload", {}) or {}),
    }


def _general_answer(state: ChatGraphState, system_prompt: str) -> str:
    llm = _build_llm()
    prompt = ChatPromptTemplate.from_messages(
        [
            ("system", system_prompt),
            MessagesPlaceholder(variable_name="chat_history"),
            (
                "user",
                "规则先验：{prior_routes}\n"
                "需求识别：{needs}\n"
                "历史背景：{context_summary}\n"
                "改写后的问题：{query}",
            ),
        ]
    )
    response = llm.invoke(
        prompt.format_messages(
            chat_history=_normalize_history(state.get("history")),
            prior_routes="；".join(state.get("prior_routes", [])),
            needs="；".join(state.get("detected_needs", [])),
            context_summary=str(state.get("context_summary", "")).strip() or "无",
            query=_last_user_query(state),
        )
    )
    return _strip_think_tags(str(getattr(response, "content", response)))


def _needs_confirmation(state: ChatGraphState) -> bool:
    if bool(state.get("confirmed")):
        return False

    query = _last_user_query(state).lower()
    if not query:
        return False

    has_action = any(keyword.lower() in query for keyword in _CONFIRM_ACTION_KEYWORDS)
    if not has_action:
        return False

    has_hotel_object = any(keyword.lower() in query for keyword in _HOTEL_OBJECT_KEYWORDS)
    if has_hotel_object:
        return True

    has_ticket_object = any(keyword.lower() in query for keyword in _SPECIFIC_TICKET_OBJECT_KEYWORDS)
    if not has_ticket_object:
        return False

    ticket_lookup_keywords = {
        "查",
        "查询",
        "看一下",
        "看下",
        "看看",
        "还有哪些",
        "有哪些",
        "有没有",
        "余票",
        "车次",
        "高铁票",
        "动车票",
        "火车票",
    }
    ticket_commit_keywords = {
        "预订",
        "订票",
        "买票",
        "购买",
        "下单",
        "订这个",
        "订这班",
        "就这班",
        "锁定",
        "改签",
        "候补",
        "抢票",
    }
    has_ticket_lookup_intent = any(keyword in query for keyword in ticket_lookup_keywords)
    has_ticket_commit_intent = any(keyword in query for keyword in ticket_commit_keywords)

    has_ticket_detail_keyword = any(keyword.lower() in query for keyword in _SPECIFIC_TICKET_DETAIL_KEYWORDS)
    has_ticket_datetime = bool(re.search(r"\d{1,2}\s*[:?]\s*\d{0,2}|\d{4}-\d{1,2}-\d{1,2}|\d{1,2}?\d{1,2}?", query))
    has_train_or_flight_no = bool(re.search(r"\b[a-z]{0,2}\d{2,4}\b", query))

    if has_ticket_commit_intent:
        return has_ticket_detail_keyword or has_ticket_datetime or has_train_or_flight_no
    if has_ticket_lookup_intent:
        return False
    return has_ticket_detail_keyword and has_train_or_flight_no


def confirmation_gate(state: ChatGraphState) -> dict[str, Any]:
    if bool(state.get("confirmed")):
        return {
            "requires_confirmation": False,
            "status": "completed",
            "pending_confirmation": {},
        }

    if not _needs_confirmation(state):
        return {
            "requires_confirmation": False,
            "status": "completed",
            "pending_confirmation": {},
        }

    rewritten_question = _last_user_query(state)
    current_signature = _build_confirmation_signature(
        route=str(state.get("route", "other")).strip(),
        rewritten_question=rewritten_question,
        detected_needs=state.get("detected_needs", []),
    )
    last_confirmed_signature = _extract_last_confirmed_signature(state.get("history"))
    if current_signature and current_signature == last_confirmed_signature:
        return {
            "requires_confirmation": False,
            "status": "completed",
            "pending_confirmation": {},
        }

    pending_confirmation = {
        "original_question": str(state.get("effective_question", state.get("question", ""))).strip(),
        "rewritten_question": rewritten_question,
        "route": str(state.get("route", "other")).strip(),
        "detected_needs": state.get("detected_needs", []),
        "confirmation_signature": current_signature,
    }
    answer = (
        f"我理解你的需求如下：\n{rewritten_question}\n\n"
        "这类请求可能会触发实际预订或购票操作。\n\n"
        "请先确认是否继续。\n"
        "回复“确认”后我再继续为你处理。"
    )
    return {
        "requires_confirmation": True,
        "status": "needs_confirmation",
        "pending_confirmation": pending_confirmation,
        "answer": answer,
        "answer_source": "confirmation_gate",
    }


def _after_confirmation_gate(state: ChatGraphState) -> str:
    if bool(state.get("requires_confirmation")):
        return "await_confirmation"
    return _route(state)


def await_confirmation(state: ChatGraphState) -> dict[str, Any]:
    return {
        "answer": str(state.get("answer", "")).strip(),
        "answer_source": str(state.get("answer_source", "confirmation_gate")).strip() or "confirmation_gate",
        "status": "needs_confirmation",
        "verification": {
            "is_complete": False,
            "covered_needs": [],
            "missing_needs": state.get("detected_needs", []),
            "unsupported_needs": [],
            "answer_source": "confirmation_gate",
        },
    }


def agent_manager(state: ChatGraphState) -> dict[str, RouteType]:
    deterministic_route = _deterministic_route(state)
    if deterministic_route is not None:
        return {"route": deterministic_route}

    llm = _build_llm()
    prompt = ChatPromptTemplate.from_messages(
        [
            (
                "system",
                "你是一个总调度助手。"
                "如果用户问题依赖知识库、文档、资料检索，返回 rag。"
                "如果用户问题和车票、火车票、12306 相关，返回 ticket。"
                "如果用户问题和地图、导航、路线、天气、地理位置相关，返回 roadmap。"
                "其他情况返回 other。"
                "请参考规则先验，但最终以改写后的完整问题为准。"
                "只允许输出 ticket、rag、roadmap、other 之一。",
            ),
            MessagesPlaceholder(variable_name="chat_history"),
            (
                "user",
                "规则先验：{prior_routes}\n"
                "需求识别：{needs}\n"
                "历史背景：{context_summary}\n"
                "改写后的问题：{query}",
            ),
        ]
    )
    response = llm.invoke(
        prompt.format_messages(
            chat_history=_normalize_history(state.get("history")),
            prior_routes="；".join(state.get("prior_routes", [])),
            needs="；".join(state.get("detected_needs", [])),
            context_summary=str(state.get("context_summary", "")).strip() or "无",
            query=_last_user_query(state),
        )
    )
    content = str(getattr(response, "content", response)).strip().lower()
    if content not in {"ticket", "rag", "roadmap", "other"}:
        content = "other"
    return {"route": content}  # type: ignore[return-value]


async def _run_mcp_agent(*, query: str, client_config: dict[str, dict[str, object]]) -> str:
    llm = _build_llm()
    client = MultiServerMCPClient(client_config)
    tools = await client.get_tools()
    agent = create_agent(model=llm, tools=tools)
    response = await agent.ainvoke({"messages": [{"role": "user", "content": query}]})
    messages = response.get("messages", [])
    if not messages:
        return ""
    last_message = messages[-1]
    content = getattr(last_message, "content", last_message)
    return _strip_think_tags(str(content))


def _build_ticket_query(state: ChatGraphState) -> str:
    query = _last_user_query(state)
    needs = state.get("detected_needs", [])
    context_summary = str(state.get("context_summary", "")).strip() or "无"
    return (
        "你是专业票务出行助手。请基于用户问题调用票务工具，并返回可直接用于决策的完整票务信息。\n"
        "你的目标不是只回答“有票/没票”，而是生成结构化票务结果。\n\n"
        "【必须优先提取并查询】\n"
        "1. 出发地\n"
        "2. 目的地\n"
        "3. 出行日期\n"
        "4. 出发时间范围（如上午/下午/晚上）\n"
        "5. 出行方式（高铁/火车/飞机/船等）\n"
        "6. 席别/舱位偏好\n"
        "7. 是否要购票、改签、退票、余票查询\n\n"
        "【输出要求】\n"
        "如果查到结果，必须尽量按下面结构输出，不要只给一句话总结：\n"
        "一、查询概况\n"
        "出发地：...\n"
        "目的地：...\n"
        "日期：...\n"
        "出行方式：...\n"
        "用户关注点：...\n\n"
        "二、推荐结果\n"
        "请按相关性和时间顺序列出 3 到 5 个最合适结果；每个结果尽量包含：\n"
        "- 车次/航班号\n"
        "- 出发站/机场\n"
        "- 到达站/机场\n"
        "- 出发时间\n"
        "- 到达时间\n"
        "- 历时\n"
        "- 席别/舱位\n"
        "- 票价\n"
        "- 余票/库存\n"
        "- 是否可预订/是否可改签\n"
        "- 推荐理由\n\n"
        "三、建议\n"
        "请明确说明最推荐哪一个，以及原因（时间更合适、价格更低、历时更短、余票更充足等）。\n\n"
        "四、缺失信息\n"
        "如果用户信息不足，必须明确列出缺少哪些关键信息，而不是笼统说“信息不足”。\n"
        "例如：缺少出发日期、缺少出发地、缺少具体车次、缺少席别偏好。\n\n"
        "【重要约束】\n"
        "1. 如果查到多个结果，不要省略关键字段。\n"
        "2. 如果工具没有返回某项，就明确写“未提供”，不要编造。\n"
        "3. 如果用户问的是购票/改签/退票，请同时说明当前能否继续执行，以及执行前还需要用户确认什么。\n"
        "4. 输出要像真实票务助手，结构清晰，便于用户直接做决定。\n\n"
        f"用户当前问题：{query}\n"
        f"识别到的需求：{json.dumps(needs, ensure_ascii=False)}\n"
        f"历史上下文摘要：{context_summary}"
    )


def _format_ticket_answer(state: ChatGraphState, raw_answer: str) -> str:
    answer = raw_answer.strip()
    if not answer:
        return answer

    try:
        llm = _build_llm()
        prompt = ChatPromptTemplate.from_messages(
            [
                (
                    "system",
                    "你是票务结果整理助手。"
                    "请把原始票务查询结果整理成清晰、可直接决策的票务摘要。"
                    "不要只保留概括，尽量保留具体车次/航班、时间、票价、余票、席别等细节。"
                    "如果原始结果里有多个班次，请按相关性和时间顺序整理。"
                    "如果有字段缺失，请明确写“未提供”，不要编造。"
                    "建议尽量按下面结构输出：\n"
                    "一、查询概况\n"
                    "二、推荐结果\n"
                    "三、最优建议\n"
                    "四、缺失信息或下一步需要确认的信息"
                ),
                MessagesPlaceholder(variable_name="chat_history"),
                (
                    "user",
                    "改写后的问题：{query}\n"
                    "识别到的需求：{needs}\n"
                    "历史背景：{context_summary}\n"
                    "原始票务结果：\n{raw_answer}"
                ),
            ]
        )
        response = llm.invoke(
            prompt.format_messages(
                chat_history=_normalize_history(state.get("history")),
                query=_last_user_query(state),
                needs=json.dumps(state.get("detected_needs", []), ensure_ascii=False),
                context_summary=str(state.get("context_summary", "")).strip() or "无",
                raw_answer=answer,
            )
        )
        formatted = _strip_think_tags(str(getattr(response, "content", response))).strip()
        return formatted or answer
    except Exception:
        return answer


def _build_roadmap_query(state: ChatGraphState) -> str:
    query = _last_user_query(state)
    needs = state.get("detected_needs", [])
    context_summary = str(state.get("context_summary", "")).strip() or "无"
    prior_routes = state.get("prior_routes", [])

    return (
        "你是高德地图旅游规划助手。请根据用户问题调用地图相关工具，生成一份尽量接近真实旅游攻略的结果，而不是只返回零散地点名称。\n\n"
        "【任务目标】\n"
        "请优先完成以下几类任务中的一个或多个：\n"
        "1. 查路线：给出从起点到终点的清晰路线方案\n"
        "2. 做攻略：围绕景点、酒店、餐饮生成可执行游玩方案\n"
        "3. 找周边：根据用户位置或目标地点推荐附近酒店、景点、餐厅\n"
        "4. 做串联：当问题同时涉及酒店、景点、餐饮、多站点时，必须生成一条完整动线\n\n"
        "【信息提取要求】\n"
        "请尽量识别并利用这些信息：\n"
        "- 出发地 / 当前位置\n"
        "- 目的地\n"
        "- 城市 / 景区\n"
        "- 时间（今天、明天、周末、一日游、两日游等）\n"
        "- 出行方式（步行、打车、公交、地铁、驾车）\n"
        "- 用户偏好（省时、省钱、少换乘、适合拍照、适合亲子、适合情侣等）\n"
        "- 是否涉及住宿、景点、餐饮、打卡点、购物点\n\n"
        "【住宿类问题强制要求】\n"
        "如果用户问题包含住宿、酒店、宾馆、民宿、住哪里：\n"
        "1. 必须优先返回住宿类地点，不要默认返回餐馆或景点\n"
        "2. 必须说明推荐入住片区或推荐酒店列表\n"
        "3. 每个住宿推荐至少包含：名称、位置、与核心景点/车站/商圈的距离或到达方式、推荐理由\n"
        "4. 如果工具能提供评分、价格、设施、房型，可补充；拿不到就写“未提供”\n\n"
        "【景点/餐饮/住宿配图强制要求】\n"
        "如果结果中出现酒店、景点、景区、餐厅、饭馆、打卡点：\n"
        "1. 必须尽量返回对应图片资源\n"
        "2. 图片字段统一写成：图片：URL1；URL2；URL3\n"
        "3. 如果工具能返回封面图、照片链接、静态地图图、POI图片，都要保留\n"
        "4. 如果没有拿到图片，也必须显式写：图片：暂无\n"
        "5. 不允许省略图片字段\n\n"
        "【旅游攻略强制要求】\n"
        "如果用户提到景点、景区、游玩、旅游、打卡、攻略、一日游、两日游：\n"
        "不要只返回景点列表，必须生成接近真实旅游攻略的方案，至少包括：\n"
        "1. 行程主题或适合人群\n"
        "2. 推荐游玩顺序\n"
        "3. 每一段交通方式\n"
        "4. 每一段预计耗时或距离\n"
        "5. 每个景点建议停留时长\n"
        "6. 中途适合吃饭或休息的位置\n"
        "7. 晚上住哪里或行程终点建议\n"
        "8. 必要的注意事项（避开高峰、预约、携带物品、天气影响等）\n\n"
        "【多站点串联强制要求】\n"
        "如果用户同时提到酒店、景点、餐饮或多个站点，必须输出串联动线。\n"
        "不要分别推荐，要按先后顺序写清楚：\n"
        "起点 -> 第1站 -> 第2站 -> 用餐点 -> 第3站 -> 住宿地/终点\n"
        "并说明每段交通方式、时间、推荐原因。\n\n"
        "【输出风格要求】\n"
        "请尽量结构化输出，像真实旅游攻略而不是地图检索摘要。\n"
        "优先包含以下模块：\n"
        "一、行程概览\n"
        "二、推荐路线/攻略正文\n"
        "三、推荐酒店/景点/餐厅清单\n"
        "四、交通建议\n"
        "五、注意事项\n\n"
        "【重要约束】\n"
        "1. 信息不足时，先基于现有信息给出尽量可执行的默认方案，再指出缺少什么。\n"
        "2. 不要只回答“建议去某某景点”。\n"
        "3. 不要只输出泛泛推荐，要尽量形成可落地的路线或攻略。\n"
        "4. 对每个 POI，尽可能补充图片字段。\n\n"
        f"用户当前问题：{query}\n"
        f"识别到的需求：{json.dumps(needs, ensure_ascii=False)}\n"
        f"规则先验路由：{json.dumps(prior_routes, ensure_ascii=False)}\n"
        f"历史上下文摘要：{context_summary}"
    )


def _extract_image_urls(text: str) -> list[str]:
    if not text:
        return []

    urls: list[str] = []

    markdown_matches = re.findall(
        r"!\[[^\]]*\]\((https?://[^)\s]+)\)",
        text,
        flags=re.IGNORECASE,
    )

    direct_matches = re.findall(
        r"https?://[^\s<>\])，。；;,]+",
        text,
        flags=re.IGNORECASE,
    )

    image_field_matches = re.findall(
        r"图片[:：]\s*(.+)",
        text,
        flags=re.IGNORECASE,
    )

    image_suffixes = (".png", ".jpg", ".jpeg", ".gif", ".webp", ".bmp", ".svg")

    candidates: list[str] = []
    candidates.extend(markdown_matches)
    candidates.extend(direct_matches)

    for field in image_field_matches:
        parts = re.split(r"[；;、,\s]+", field.strip())
        for part in parts:
            if part.startswith("http://") or part.startswith("https://"):
                candidates.append(part)

    for url in candidates:
        normalized = url.rstrip(").,;]），。；")
        lowered = normalized.lower()
        if lowered.endswith(image_suffixes) or any(
            token in lowered for token in ("image", "img", "photo", "picture", "staticmap", "snapshot")
        ):
            if normalized not in urls:
                urls.append(normalized)

    return urls


def _build_image_items(text: str) -> list[dict[str, str]]:
    if not text:
        return []

    image_urls = _extract_image_urls(text)
    if not image_urls:
        return []

    lines = [line.strip() for line in text.splitlines()]
    items: list[dict[str, str]] = []

    def infer_anchor_text(url: str) -> str:
        for index, line in enumerate(lines):
            if url not in line:
                continue
            for offset in range(index - 1, max(-1, index - 4), -1):
                candidate = lines[offset].lstrip("-•1234567890. ").strip()
                if candidate and "http" not in candidate and not candidate.startswith("图片"):
                    return candidate[:80]
        return ""

    for index, url in enumerate(image_urls):
        anchor_text = infer_anchor_text(url)
        title = anchor_text or f"图片 {index + 1}"
        items.append(
            {
                "image_url": url,
                "title": title,
                "category": "",
                "anchor_text": anchor_text,
            }
        )
    return items

def _inline_image_markdown(text: str, max_images_per_block: int = 3) -> str:
    """
    将回答中的“图片：URL1；URL2 ...”转换为 Markdown 图片展示。
    """
    if not text:
        return text

    def repl(match: re.Match[str]) -> str:
        raw = match.group(1).strip()
        if not raw or raw == "暂无":
            return "图片：暂无"

        parts = re.split(r"[；;、,\s]+", raw)
        urls: list[str] = []
        for part in parts:
            part = part.strip()
            if part.startswith("http://") or part.startswith("https://"):
                urls.append(part)

        if not urls:
            return f"图片：{raw}"

        urls = urls[:max_images_per_block]
        markdown_imgs = "\n".join(f"![]({url})" for url in urls)
        return f"图片：\n{markdown_imgs}"

    return re.sub(r"图片[:：]\s*(.+)", repl, text, flags=re.IGNORECASE)


def _inline_loose_image_urls(text: str, max_total_images: int = 6) -> str:
    """
    将正文中裸露的图片 URL 转成 Markdown 图片。
    只处理明显是图片资源的链接。
    """
    if not text:
        return text

    image_suffixes = (".png", ".jpg", ".jpeg", ".gif", ".webp", ".bmp", ".svg")
    replaced_count = 0

    def repl(match: re.Match[str]) -> str:
        nonlocal replaced_count
        url = match.group(0).rstrip(").,;]）。，；")
        lowered = url.lower()

        if replaced_count >= max_total_images:
            return url

        if lowered.endswith(image_suffixes) or any(
            token in lowered for token in ("image", "img", "photo", "picture", "staticmap", "snapshot")
        ):
            replaced_count += 1
            return f"\n![]({url})\n"
        return url

    return re.sub(r"https?://[^\s<>\])，。；;,]+", repl, text, flags=re.IGNORECASE)


def _render_images_in_answer(text: str) -> str:
    if not text:
        return text
    rendered = _inline_image_markdown(text)
    rendered = _inline_loose_image_urls(rendered)
    return rendered


def _format_roadmap_answer(state: ChatGraphState, raw_answer: str) -> str:
    answer = raw_answer.strip()
    if not answer:
        return answer

    try:
        llm = _build_llm()
        prompt = ChatPromptTemplate.from_messages(
            [
                (
                    "system",
                    "你是旅游攻略整理助手。"
                    "请把地图工具返回的原始结果整理成一份更像真实旅游攻略的成品答案。"
                    "不是简单概括，而是让用户看完就知道怎么走、怎么玩、住哪里、吃什么。"
                    "请严格遵守以下要求：\n\n"
                    "【总体要求】\n"
                    "1. 输出必须结构清晰，尽量像成型旅游攻略\n"
                    "2. 如果问题涉及路线，就要有分段路线说明\n"
                    "3. 如果涉及游玩，就要有行程安排\n"
                    "4. 如果涉及住宿/景点/餐饮，每个地点尽量保留图片字段\n"
                    "5. 不要丢失原始结果中的可用细节\n\n"
                    "【推荐输出结构】\n"
                    "一、行程概览\n"
                    "- 用 2 到 4 句话概括这条路线/这份攻略适合谁、总耗时大概多久、核心亮点是什么\n\n"
                    "二、详细行程\n"
                    "请按“第1段、第2段、第3段……”输出，每段尽量写清：\n"
                    "- 起点\n"
                    "- 终点\n"
                    "- 推荐交通方式\n"
                    "- 预计耗时/距离\n"
                    "- 到达后建议做什么\n"
                    "- 建议停留多久\n\n"
                    "三、推荐地点清单\n"
                    "如果有酒店、景点、餐厅，请分别列出。每个地点尽量包含：\n"
                    "- 名称\n"
                    "- 位置\n"
                    "- 推荐理由\n"
                    "- 图片：URL1；URL2；URL3 或 图片：暂无\n\n"
                    "四、交通与住宿建议\n"
                    "如果存在住宿安排，明确说明建议住在哪个区域或哪家酒店，以及原因。\n\n"
                    "五、注意事项\n"
                    "给出 2 到 5 条真实可执行建议，如避开高峰、预约、天气、步行强度、适合打车/地铁等。\n\n"
                    "【重要约束】\n"
                    "1. 不能只做笼统概括\n"
                    "2. 不能把多站点拆成孤立推荐，必须串起来\n"
                    "3. 不能省略图片字段；没有图也要写“图片：暂无”\n"
                    "4. 尽量让答案像真正旅游攻略，而不是工具原始回显"
                ),
                MessagesPlaceholder(variable_name="chat_history"),
                (
                    "user",
                    "改写后的问题：{query}\n"
                    "识别到的需求：{needs}\n"
                    "历史背景：{context_summary}\n"
                    "原始地图结果：\n{raw_answer}"
                ),
            ]
        )
        response = llm.invoke(
            prompt.format_messages(
                chat_history=_normalize_history(state.get("history")),
                query=_last_user_query(state),
                needs=json.dumps(state.get("detected_needs", []), ensure_ascii=False),
                context_summary=str(state.get("context_summary", "")).strip() or "无",
                raw_answer=answer,
            )
        )
        formatted = _strip_think_tags(str(getattr(response, "content", response))).strip()
        return formatted or answer
    except Exception:
        return answer


async def ticket(state: ChatGraphState) -> dict[str, Any]:
    raw_answer = await _run_mcp_agent(
        query=_build_ticket_query(state),
        client_config={
            "12306-mcp": {
                "transport": "stdio",
                "command": "npx",
                "args": ["-y", "12306-mcp"],
            }
        },
    )
    answer = _format_ticket_answer(state, raw_answer)
    answer = _render_images_in_answer(answer)
    return {"answer": answer, "answer_source": "ticket_mcp"}


async def roadmap(state: ChatGraphState) -> dict[str, Any]:
    raw_answer = await _run_mcp_agent(
        query=_build_roadmap_query(state),
        client_config={
            "amap-maps-streamableHTTP": {
                "transport": "streamable_http",
                "url": "https://mcp.amap.com/mcp?key=1f8c43d66527b0fdf3c98ded711f86b7",
            }
        },
    )
    formatted_answer = _format_roadmap_answer(state, raw_answer)
    display_answer = _render_images_in_answer(formatted_answer or raw_answer)
    return {
        "answer": display_answer,
        "answer_source": "roadmap_mcp",
    }


def rag(state: ChatGraphState) -> dict[str, Any]:
    from backend.app.crued.chat import get_chat_answer

    question = _last_user_query(state)
    top_k = int(state.get("top_k") or SETTINGS.final_top_k)
    user_id = int(state.get("user_id") or 0)
    answer = get_chat_answer(question=question, top_k=top_k, user_id=user_id)
    answer = _render_images_in_answer(answer)
    return {"answer": answer, "answer_source": "rag_service"}


def other(state: ChatGraphState) -> dict[str, Any]:
    answer = _general_answer(
        state,
        "你是旅行场景下的综合出行建议助手。"
        "当用户的问题不适合票务查询、规则型知识检索或路线规划时，"
        "请给出清晰、实用、贴近真实出行决策的建议。"
        "优先结合预算、人群特征、出行时长、轻松程度、便利性等因素，"
        "输出可执行的建议，而不是空泛表述。",
    )
    answer = _render_images_in_answer(answer)
    return {"answer": answer, "answer_source": "general_llm"}


def verify_answer(state: ChatGraphState) -> dict[str, Any]:
    if str(state.get("status", "completed")).strip().lower() == "needs_confirmation":
        return {
            "verification": {
                "is_complete": False,
                "covered_needs": [],
                "missing_needs": state.get("detected_needs", []),
                "unsupported_needs": [],
                "answer_source": "confirmation_gate",
            }
        }

    answer = str(state.get("answer", "")).strip()
    detected_needs = state.get("detected_needs", [])
    fallback = {
        "is_complete": bool(answer),
        "covered_needs": detected_needs if answer else [],
        "missing_needs": [] if answer else detected_needs,
        "unsupported_needs": [],
        "answer_source": str(state.get("answer_source", "")).strip(),
    }

    try:
        parsed = _safe_json_llm(
            system_prompt=(
                "你是答案校验助手。"
                "请根据用户原问题、改写后的问题、需求清单、答案内容和答案来源，判断答案是否覆盖了用户问题的全部需求。"
                "只做校验，不要重写答案。"
                "只输出 JSON，对象中必须包含字段："
                "is_complete(boolean)、covered_needs(array)、missing_needs(array)、unsupported_needs(array)、answer_source(string)。"
                "如果某个需求虽然提到了但没有真正回答，要放进 missing_needs。"
                "不要输出额外解释。"
            ),
            user_prompt=(
                f"原问题：{str(state.get('question', '')).strip()}\n"
                f"改写后问题：{_last_user_query(state)}\n"
                f"规则先验：{json.dumps(state.get('prior_routes', []), ensure_ascii=False)}\n"
                f"需求清单：{json.dumps(detected_needs, ensure_ascii=False)}\n"
                f"答案来源：{str(state.get('answer_source', '')).strip()}\n"
                f"答案内容：{answer}"
            ),
            history=state.get("history"),
        )
    except Exception:
        parsed = fallback

    is_complete = bool(parsed.get("is_complete", fallback["is_complete"])) if isinstance(parsed, dict) else fallback["is_complete"]
    covered_needs = parsed.get("covered_needs", fallback["covered_needs"]) if isinstance(parsed, dict) else fallback["covered_needs"]
    missing_needs = parsed.get("missing_needs", fallback["missing_needs"]) if isinstance(parsed, dict) else fallback["missing_needs"]
    unsupported_needs = parsed.get("unsupported_needs", fallback["unsupported_needs"]) if isinstance(parsed, dict) else fallback["unsupported_needs"]
    answer_source = str(parsed.get("answer_source", fallback["answer_source"])) if isinstance(parsed, dict) else fallback["answer_source"]

    verification = {
        "is_complete": is_complete,
        "covered_needs": [str(item).strip() for item in covered_needs if str(item).strip()],
        "missing_needs": [str(item).strip() for item in missing_needs if str(item).strip()],
        "unsupported_needs": [str(item).strip() for item in unsupported_needs if str(item).strip()],
        "answer_source": answer_source,
    }
    return {"verification": verification}


def summarize_result(state: ChatGraphState) -> dict[str, Any]:
    final_answer = str(state.get("answer", "")).strip()
    rewritten_question = _last_user_query(state)
    is_confirmed = bool(state.get("confirmed"))
    confirmation_signature = ""
    if is_confirmed:
        confirmation_signature = _build_confirmation_signature(
            route=str(state.get("route", "")).strip(),
            rewritten_question=rewritten_question,
            detected_needs=state.get("detected_needs", []),
        )
    final_summary = {
        "original_question": str(state.get("question", "")).strip(),
        "effective_question": str(state.get("effective_question", "")).strip(),
        "rewritten_question": rewritten_question,
        "prior_routes": state.get("prior_routes", []),
        "detected_needs": state.get("detected_needs", []),
        "context_summary": str(state.get("context_summary", "")).strip(),
        "route": str(state.get("route", "")).strip(),
        "status": str(state.get("status", "completed")).strip() or "completed",
        "pending_confirmation": state.get("pending_confirmation", {}),
        "confirmed": is_confirmed,
        "confirmation_signature": confirmation_signature,
        "answer_source": str(state.get("answer_source", "")).strip(),
        "verification": state.get("verification", {}),
        "image_urls": _extract_image_urls(final_answer),
        "image_items": _build_image_items(final_answer),
        "final_answer": final_answer,
    }
    return {"final_summary": final_summary}


def _route(state: ChatGraphState) -> str:
    route = str(state.get("route", "other")).strip().lower()
    if route in {"ticket", "rag", "roadmap", "other"}:
        return route
    return "other"


@lru_cache(maxsize=1)
def build_chat_graph():
    builder: StateGraph[ChatGraphState] = StateGraph(ChatGraphState)
    builder.add_node("preprocess_query", preprocess_query)
    builder.add_node("agent_manager", agent_manager)
    builder.add_node("confirmation_gate", confirmation_gate)
    builder.add_node("await_confirmation", await_confirmation)
    builder.add_node("ticket", ticket)
    builder.add_node("rag", rag)
    builder.add_node("roadmap", roadmap)
    builder.add_node("other", other)
    builder.add_node("verify_answer", verify_answer)
    builder.add_node("summarize_result", summarize_result)
    builder.add_edge(START, "preprocess_query")
    builder.add_edge("preprocess_query", "agent_manager")
    builder.add_edge("agent_manager", "confirmation_gate")
    builder.add_conditional_edges(
        "confirmation_gate",
        _after_confirmation_gate,
        ["await_confirmation", "ticket", "rag", "roadmap", "other", END],
    )
    builder.add_edge("await_confirmation", "summarize_result")
    builder.add_edge("ticket", "verify_answer")
    builder.add_edge("rag", "verify_answer")
    builder.add_edge("roadmap", "verify_answer")
    builder.add_edge("other", "verify_answer")
    builder.add_edge("verify_answer", "summarize_result")
    builder.add_edge("summarize_result", END)
    return builder.compile()


graph = build_chat_graph()


async def run_chat_graph(
    question: str,
    top_k: int,
    user_id: int,
    history: list[dict[str, Any]] | None = None,
) -> dict[str, Any]:
    result = await graph.ainvoke(
        {
            "question": question,
            "history": _normalize_history(history),
            "top_k": top_k,
            "user_id": user_id,
        }
    )
    pending_confirmation = result.get("pending_confirmation")
    return {
        "answer": str(result.get("answer", "")).strip(),
        "status": str(result.get("status", "completed")).strip() or "completed",
        "pending_confirmation": pending_confirmation if isinstance(pending_confirmation, dict) and pending_confirmation else None,
        "final_summary": result.get("final_summary", {}),
    }
