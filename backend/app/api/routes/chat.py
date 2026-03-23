from fastapi import APIRouter, Header, HTTPException, status

from backend.app.core.database import AsyncSessionLocal
from backend.app.crued.chat import get_agent_chat_answer
from backend.app.crued.user import verify_token
from backend.app.schemas.chat import ChatMessage, ChatRequest, ChatResponse
from backend.app.utils.user import parse_bearer_token


router = APIRouter()


@router.post("/completion", response_model=ChatResponse)
async def chat_completion(
    payload: ChatRequest,
    authorization: str | None = Header(default=None, alias="Authorization"),
) -> ChatResponse:
    token = parse_bearer_token(authorization)
    async with AsyncSessionLocal() as db:
        user_id = await verify_token(token, db)

    try:
        history_payload = [message.model_dump() for message in payload.history]
        result = await get_agent_chat_answer(
            payload.question,
            payload.top_k,
            user_id,
            history_payload,
        )
    except Exception as exc:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"LLM generation failed: {exc}",
        ) from exc

    reply = str(result.get("answer", "")).strip()
    status_value = str(result.get("status", "completed")).strip().lower()
    status_name = "needs_confirmation" if status_value == "needs_confirmation" else "completed"
    pending_confirmation = result.get("pending_confirmation")
    final_summary = result.get("final_summary")
    assistant_metadata: dict[str, object] = {}
    if isinstance(pending_confirmation, dict) and pending_confirmation:
        assistant_metadata["pending_confirmation"] = pending_confirmation
    if isinstance(final_summary, dict) and final_summary:
        assistant_metadata["final_summary"] = final_summary

    history = payload.history + [
        ChatMessage(role="user", content=payload.question),
        ChatMessage(role="assistant", content=reply, metadata=assistant_metadata),
    ]
    return ChatResponse(
        answer=reply,
        history=history,
        status=status_name,
        pending_confirmation=pending_confirmation if isinstance(pending_confirmation, dict) else None,
        final_summary=final_summary if isinstance(final_summary, dict) else None,
    )
