from fastapi import APIRouter, HTTPException, Depends, status, Header
from typing import List, Optional, Dict, Any
from datetime import datetime
import time

from loguru import logger
from ..services.openai_client import get_openai_client
from ..services.aihub_processor import get_teen_empathy_processor
from ..services.conversation_service import get_conversation_service
from ..core.vector_store import get_vector_store  # 아직 사용 예정이므로 유지
from ..models.function_models import (
    TeenChatRequest, TeenChatResponse, ReActStep,
    EmotionType, RelationshipType, EmpathyStrategy,
    ChatMessage, EmotionAnalysisRequest,
)

router = APIRouter()


class EnhancedTrueRAGTeenChatbot:
    """향상된 True RAG 챗봇 (세션 영구 저장 지원)"""

    def __init__(self):
        self.emotions = [e.value for e in EmotionType]
        self.relationships = [r.value for r in RelationshipType]
        self.strategies = [s.value for s in EmpathyStrategy]

    # ------------------------------------------------------------------
    # 1. 감정ㆍ관계 맥락 감지
    # ------------------------------------------------------------------
    async def detect_emotion_and_context(
        self, message: str, openai_client
    ) -> tuple[EmotionType, Optional[RelationshipType]]:
        try:
            result = await openai_client.analyze_emotion_and_context(
                text=message, additional_context=None
            )
            return result.primary_emotion, result.relationship_context
        except Exception as e:
            logger.error(f"감정/맥락 감지 실패: {e}")
            # 실패 시 기본값
            return EmotionType.ANXIETY, None

    # ------------------------------------------------------------------
    # 2. 세션 관리
    # ------------------------------------------------------------------
    async def get_or_create_session(
        self, session_id: str | None, conversation_service
    ) -> str:
        return await conversation_service.get_or_create_session(session_id)

    # ------------------------------------------------------------------
    # 3. 대화 컨텍스트 확보
    # ------------------------------------------------------------------
    async def get_enhanced_conversation_context(
        self,
        session_id: str,
        current_message: str,
        conversation_service,
        fallback_history: List[ChatMessage | Dict] | None = None,
    ) -> Dict[str, Any]:
        try:
            db_context = await conversation_service.get_enhanced_context(
                session_id, current_message
            )

            # DB에 없으면 클라이언트 히스토리 사용
            if not db_context["conversation_history"] and fallback_history:
                logger.info("📱 DB 컨텍스트 없음 → 클라이언트 히스토리 사용")
                formatted = []
                for msg in fallback_history[-10:]:
                    if hasattr(msg, "role"):
                        formatted.append(
                            {
                                "role": msg.role.value if hasattr(msg.role, "value") else str(msg.role),
                                "content": msg.content,
                                "timestamp": getattr(msg, "timestamp", datetime.now().isoformat()),
                                "emotion": getattr(msg, "emotion", None),
                                "priority": 2,
                            }
                        )
                    elif isinstance(msg, dict):
                        formatted.append(
                            {
                                "role": msg.get("role", "user"),
                                "content": msg.get("content", ""),
                                "timestamp": msg.get("timestamp", datetime.now().isoformat()),
                                "emotion": msg.get("emotion"),
                                "priority": 2,
                            }
                        )
                db_context["conversation_history"] = formatted
                db_context["context_source"] = "client_fallback"
            else:
                db_context["context_source"] = "database"

            logger.info(
                f"📚 컨텍스트 소스: {db_context['context_source']}, "
                f"{len(db_context['conversation_history'])}개 메시지"
            )
            return db_context
        except Exception as e:
            logger.error(f"❌ 컨텍스트 조회 실패: {e}")
            return {
                "conversation_history": fallback_history[-8:] if fallback_history else [],
                "session_stats": {},
                "context_summary": {},
                "context_source": "fallback_only",
            }

    # ------------------------------------------------------------------
    # 4. 세션 요약
    # ------------------------------------------------------------------
    def _summarize_session_context(self, history: List[Dict] | None) -> str:
        if not history:
            return ""

        recent_user_msgs = [
            (m.get("content", "") if isinstance(m, dict) else getattr(m, "content", ""))
            for m in history[-6:]
            if (m.get("role", "") if isinstance(m, dict) else getattr(m, "role", ""))
            == "user"
        ]
        if not recent_user_msgs:
            return ""
        short = [msg[:40] + "..." if len(msg) > 40 else msg for msg in recent_user_msgs[-3:]]
        return "이전 대화 맥락: " + " → ".join(short)

    # ------------------------------------------------------------------
    # 5. 응답 생성 (세션 인식 + True RAG)
    # ------------------------------------------------------------------
    async def _generate_with_session_awareness(
        self,
        user_message: str,
        similar_contexts: List[Dict],
        history: List[Dict],
        emotion: str,
        relationship: str,
        openai_client,
    ) -> Dict[str, Any]:
        context_summary = self._summarize_session_context(history)

        # 5‑1. RAG 검색 결과 이용
        if similar_contexts:
            best = similar_contexts[0]
            expert_response = best["system_response"]
            prompt = f"""
다음 전문가 상담사의 응답을 현재 청소년 상황에 맞게 적응시켜주세요.

전문가 응답: "{expert_response}"
현재 상황: "{user_message}" (감정: {emotion}, 관계: {relationship})
{context_summary}

🔥 요구사항:
1. 맥락 일관성 유지  2. 청소년 친화 표현  3. 구체적 조언

청소년 맞춤 응답:
"""
            try:
                res = await openai_client.create_completion(
                    messages=[{"role": "user", "content": prompt}],
                    temperature=0.4,
                    max_tokens=400,
                )
                return {
                    "response": res.content.strip(),
                    "retrieval_used": True,
                    "source": "single_expert_with_context",
                    "similarity_score": best.get("similarity_score", 0),
                    "adaptation_level": "context_aware",
                    "context_influence": bool(context_summary),
                }
            except Exception as e:
                logger.error(f"전문가 응답 적응 실패: {e}")
                # 폴백: 전문가 응답 최소 변환
                return await self._basic_expert_adaptation(expert_response, relationship, openai_client)

        # 5‑2. 검색 결과가 없을 때 – 컨텍스트 인식 폴백
        return await self._context_aware_fallback(
            user_message, history, emotion, openai_client
        )

    # ------------------------------------------------------------------
    # 6. 외부 API에 노출되는 상위 래퍼
    # ------------------------------------------------------------------
    async def generate_response_with_full_context(
        self,
        user_message: str,
        session_context: Dict[str, Any],
        emotion: str,
        relationship: str,
        similar_contexts: List[Dict],
        openai_client,
    ) -> Dict[str, Any]:
        history = session_context["conversation_history"]
        resp = await self._generate_with_session_awareness(
            user_message,
            similar_contexts,
            history,
            emotion,
            relationship,
            openai_client,
        )

        resp["context_metadata"] = {
            "db_context_used": len(history),
            "context_source": session_context["context_source"],
            "session_stats": session_context.get("session_stats", {}),
            "web_debug_info": {
                "search_results_count": len(similar_contexts),
                "rag_active": resp["retrieval_used"],
            },
        }
        return resp

    # ------------------------------------------------------------------
    # 7. 기본/폴백 변환 로직  (생략 부분 동일)
    # ------------------------------------------------------------------
    async def _context_aware_fallback(
        self,
        user_message: str,
        history: List[Dict],
        emotion: str,
        openai_client,
    ) -> Dict[str, Any]:
        context_summary = self._summarize_session_context(history)
        prompt = f"""
청소년과의 이전 대화를 고려하여 공감적인 응답을 해주세요.

현재 메시지: "{user_message}" (감정: {emotion})
{context_summary}

120자 내외, 친근한 톤:
"""
        try:
            res = await openai_client.create_completion(
                messages=[{"role": "user", "content": prompt}],
                temperature=0.6,
                max_tokens=300,
            )
            return {
                "response": res.content.strip(),
                "retrieval_used": False,
                "source": "context_aware_fallback",
                "adaptation_level": "session_context",
                "context_influence": bool(context_summary),
            }
        except Exception as e:
            logger.error(f"폴백 실패: {e}")
            return {
                "response": f"{emotion}한 마음이 충분히 이해돼. 천천히 해결해보자! 💙",
                "retrieval_used": False,
                "source": "emergency_fallback",
            }

    async def _basic_expert_adaptation(self, expert_response: str, relationship: str, openai_client):
        converted = expert_response
        for old, new in {
            "자기야": "너",
            "자기가": "네가",
            "하세요": "해",
            "어떠세요": "어때",
            "해보세요": "해봐",
            "직장": "학교",
            "동료": "친구",
            "상사": "선생님",
            "업무": "공부",
        }.items():
            converted = converted.replace(old, new)
        return {
            "response": converted,
            "retrieval_used": True,
            "source": "basic_adaptation",
            "adaptation_level": "minimal",
        }


# ----------------------------------------------------------------------------
# 인스턴스 & 라우트 정의
# ----------------------------------------------------------------------------

enhanced_chatbot = EnhancedTrueRAGTeenChatbot()


# ------------------------------------------------------------------
#  A. 메인 엔드포인트  /teen-chat-enhanced
# ------------------------------------------------------------------
@router.post("/teen-chat-enhanced", response_model=TeenChatResponse)
async def teen_chat_enhanced(
    request: TeenChatRequest,
    session_id: Optional[str] = Header(None, alias="X-Session-ID"),
    openai_client=Depends(get_openai_client),
    processor=Depends(get_teen_empathy_processor),
    conversation_service=Depends(get_conversation_service),
):
    try:
        t0 = time.time()
        logger.info(f"🚀 요청: {request.message[:50]}…")

        # 1. 세션 확보
        sess_id = await enhanced_chatbot.get_or_create_session(
            session_id, conversation_service
        )

        # 2. 감정·관계 감지
        emotion, relationship = await enhanced_chatbot.detect_emotion_and_context(
            request.message, openai_client
        )
        emotion_str = emotion.value if hasattr(emotion, "value") else str(emotion)
        relationship_str = (
            relationship.value
            if relationship and isinstance(relationship, RelationshipType)
            else (relationship or "친구")
        )

        # 3. 컨텍스트 수집
        sess_ctx = await enhanced_chatbot.get_enhanced_conversation_context(
            sess_id,
            request.message,
            conversation_service,
            fallback_history=getattr(request, "conversation_history", []),
        )

        # 3.5. RAG 검색
        similar_ctx = await processor.search_similar_contexts(
            query=request.message,
            emotion=emotion_str,
            relationship=relationship_str,
            top_k=5,
        )

        # 4. 응답 생성
        resp_data = await enhanced_chatbot.generate_response_with_full_context(
            request.message,
            sess_ctx,
            emotion_str,
            relationship_str,
            similar_ctx,
            openai_client,
        )

        # 5. 공감 전략
        empathy_strategy = processor.get_empathy_strategy(emotion_str)

        # 6. 신뢰도 계산
        confidence = min(
            0.95,
            0.8
            + min(0.1, len(sess_ctx["conversation_history"]) * 0.01)
            + resp_data.get("similarity_score", 0) * 0.1,
        )

        # 7. DB 저장
        meta = {
            "emotion": emotion_str,
            "empathy_strategy": empathy_strategy,
            "context_used": True,
            "similarity_score": confidence,
            "assistant": {
                "source": resp_data["source"],
                "adaptation_level": resp_data.get("adaptation_level", "unknown"),
                "context_influence": resp_data.get("context_influence", False),
            },
        }
        turn = await conversation_service.save_conversation_turn(
            sess_id, request.message, resp_data["response"], meta
        )

        # 8. 응답 객체
        latency = (time.time() - t0) * 1000
        sanitized = (
            resp_data["response"]
            .replace("자기야", "너")
            .replace("자기가", "네가")
            .replace("하세요", "해")
            .replace("어떠세요", "어때")
        )
        return TeenChatResponse(
            response=sanitized,
            detected_emotion=emotion,
            empathy_strategy=[EmpathyStrategy(s) for s in empathy_strategy],
            similar_contexts=similar_ctx[:3],
            react_steps=None,
            confidence_score=confidence,
            response_metadata={
                "processing_time_ms": latency,
                "rag_method": "enhanced_true_rag_with_context",
                "session_id": sess_id,
                "retrieval_used": resp_data["retrieval_used"],
                "source": resp_data["source"],
                "adaptation_level": resp_data.get("adaptation_level", "unknown"),
                "context_metadata": resp_data.get("context_metadata", {}),
                "conversation_saved": True,
                "db_message_ids": {
                    "user": turn["user_message_id"],
                    "assistant": turn["assistant_message_id"],
                },
            },
        )
    except Exception as e:
        logger.error(f"❌ 처리 실패: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"채팅 처리 중 오류: {e}",
        )


# ------------------------------------------------------------------
#  B. 레거시 엔드포인트  /teen-chat  (호환성)
# ------------------------------------------------------------------
@router.post("/teen-chat", response_model=TeenChatResponse)
async def teen_chat_legacy(
    request: TeenChatRequest,
    session_id: Optional[str] = Header(None, alias="X-Session-ID"),
    openai_client=Depends(get_openai_client),
    processor=Depends(get_teen_empathy_processor),
    conversation_service=Depends(get_conversation_service),
):
    # teen_chat_enhanced와 동일 로직 재사용
    return await teen_chat_enhanced(
        request=request,
        session_id=session_id,
        openai_client=openai_client,
        processor=processor,
        conversation_service=conversation_service,
    )


# ------------------------------------------------------------------
#  C. 유틸 엔드포인트  /empathy-strategies
# ------------------------------------------------------------------
@router.get("/empathy-strategies")
async def get_empathy_strategies():
    processor = await get_teen_empathy_processor()
    return {
        "empathy_strategies": {
            EmotionType.JOY.value: processor.get_empathy_strategy(EmotionType.JOY.value),
            EmotionType.CONFUSION.value: processor.get_empathy_strategy(EmotionType.CONFUSION.value),
            EmotionType.ANGER.value: processor.get_empathy_strategy(EmotionType.ANGER.value),
            EmotionType.ANXIETY.value: processor.get_empathy_strategy(EmotionType.ANXIETY.value),
            EmotionType.HURT.value: processor.get_empathy_strategy(EmotionType.HURT.value),
            EmotionType.SADNESS.value: processor.get_empathy_strategy(EmotionType.SADNESS.value),
        },
        "system_type": "enhanced_expert_counseling",
        "data_source": "ai_hub_with_persistent_context",
    }
