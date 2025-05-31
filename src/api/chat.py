"""
청소년 공감형 True RAG 채팅 API
AI Hub 검색 결과를 직접 활용하는 진짜 RAG 시스템
"""

from fastapi import APIRouter, HTTPException, Depends, status
from typing import List, Optional, Dict, Any
from datetime import datetime
import json
import time
import os
import numpy as np

from loguru import logger
from ..services.openai_client import get_openai_client
from ..services.aihub_processor import get_teen_empathy_processor
from ..core.vector_store import get_vector_store
from ..models.function_models import (
    TeenChatRequest, TeenChatResponse, ReActStep,
    EmotionType, RelationshipType, EmpathyStrategy,
    ChatMessage, EmotionAnalysisRequest
)


router = APIRouter()


class TrueRAGTeenChatbot:
    """진짜 RAG: 검색된 AI Hub 응답을 직접 활용하는 챗봇"""

    def __init__(self):
        self.emotions = [e.value for e in EmotionType]
        self.relationships = [r.value for r in RelationshipType]
        self.strategies = [s.value for s in EmpathyStrategy]

    async def detect_emotion_and_context(self, message: str, openai_client) -> tuple[EmotionType, Optional[RelationshipType]]:
        """메시지에서 감정과 관계 맥락 감지"""
        try:
            analysis_request = EmotionAnalysisRequest(text=message)
            result = await openai_client.analyze_emotion_and_context(
                text=message,
                additional_context=None
            )
            return result.primary_emotion, result.relationship_context
        except Exception as e:
            logger.error(f"감정/맥락 감지 실패: {e}")
            return EmotionType.ANXIETY, None

    async def search_similar_contexts(self, message: str, emotion: str, relationship: str,
                                    processor) -> List[Dict[str, Any]]:
        """유사한 대화 맥락 검색"""
        try:
            logger.info(f"🔍 유사 맥락 검색 - 메시지: {message[:30]}..., 감정: {emotion}, 관계: {relationship}")

            results = await processor.search_similar_contexts(
                query=message,
                emotion=emotion,
                relationship=relationship,
                top_k=5
            )

            logger.info(f"✅ 검색 완료 - {len(results)}개 결과")
            return results

        except Exception as e:
            logger.error(f"❌ 유사 맥락 검색 실패: {e}")
            return []

    async def generate_response_from_search_results(self,
                                                  user_message: str,
                                                  similar_contexts: List[Dict[str, Any]],
                                                  emotion: str,
                                                  relationship: str,
                                                  openai_client) -> Dict[str, Any]:
        """🔥 핵심: 검색된 AI Hub 응답을 직접 활용해서 답변 생성"""

        if not similar_contexts or len(similar_contexts) == 0:
            return {
                "response": await self._fallback_response(user_message, emotion),
                "retrieval_used": False,
                "source": "fallback",
                "adaptation_level": "none"
            }

        # 1️⃣ 검색된 응답들 품질 평가 및 분류
        high_quality_results = []
        medium_quality_results = []

        for ctx in similar_contexts:
            system_response = ctx.get('system_response', '')
            similarity_score = ctx.get('similarity_score', 0)

            if not system_response:
                continue

            if similarity_score >= 0.3:  # 매우 유사
                high_quality_results.append(ctx)
            elif similarity_score >= 0.15:   # 적당히 유사
                medium_quality_results.append(ctx)

        logger.info(f"📊 검색 품질 - 고품질: {len(high_quality_results)}개, 중품질: {len(medium_quality_results)}개")

        # 2️⃣ 품질에 따른 응답 생성 전략 선택
        if high_quality_results:
            # 🔥 고품질 결과: 직접 활용 (최소 변환)
            return await self._use_high_quality_results(
                high_quality_results, user_message, emotion, relationship, openai_client
            )
        elif medium_quality_results:
            # 🔥 중품질 결과: 적응적 활용
            return await self._adapt_medium_quality_results(
                medium_quality_results, user_message, emotion, relationship, openai_client
            )
        else:
            # 낮은 품질: 조합 활용
            return await self._combine_low_quality_results(
                similar_contexts, user_message, emotion, relationship, openai_client
            )

    async def _use_high_quality_results(self,
                                      high_quality_results: List[Dict],
                                      user_message: str, emotion: str, relationship: str,
                                      openai_client) -> Dict[str, Any]:
        """고품질 검색 결과 직접 활용 (최소 변환)"""

        # 가장 유사한 결과 선택
        best_result = max(high_quality_results, key=lambda x: x.get('similarity_score', 0))
        original_response = best_result['system_response']

        logger.info(f"🎯 고품질 매칭 - 유사도: {best_result.get('similarity_score', 0):.3f}")
        logger.info(f"   원본: {original_response[:100]}...")

        # 🔥 핵심: 성인 대화를 청소년 맥락으로 최소 변환
        teen_response = await self._minimal_adult_to_teen_conversion(
            original_response, relationship, openai_client
        )

        return {
            "response": teen_response,
            "retrieval_used": True,
            "source": "high_quality_direct",
            "adaptation_level": "minimal",
            "original_response": original_response,
            "similarity_score": best_result.get('similarity_score', 0),
            "empathy_strategy": best_result.get('empathy_label', '위로')
        }

    async def _adapt_medium_quality_results(self,
                                          medium_quality_results: List[Dict],
                                          user_message: str, emotion: str, relationship: str,
                                          openai_client) -> Dict[str, Any]:
        """중품질 결과 적응적 활용"""

        if len(medium_quality_results) == 1:
            # 단일 결과 적응
            result = medium_quality_results[0]
            original_response = result['system_response']

            adapted_response = await self._moderate_adaptation(
                original_response, user_message, emotion, relationship, openai_client
            )

            return {
                "response": adapted_response,
                "retrieval_used": True,
                "source": "medium_quality_adapted",
                "adaptation_level": "moderate",
                "original_response": original_response,
                "similarity_score": result.get('similarity_score', 0)
            }
        else:
            # 다중 결과 조합
            return await self._combine_multiple_responses(
                medium_quality_results, user_message, emotion, relationship, openai_client
            )

    async def _combine_multiple_responses(self,
                                        retrieved_responses: List[Dict],
                                        user_message: str, emotion: str, relationship: str,
                                        openai_client) -> Dict[str, Any]:
        """🔥 핵심: 여러 검색 결과를 조합해서 최적 응답 생성"""

        logger.info(f"🧩 다중 결과 조합 - {len(retrieved_responses)}개 응답")

        # 검색된 실제 응답들을 GPT에게 제공하여 조합
        combination_prompt = f"""
다음은 AI Hub에서 검색된 실제 전문 상담사의 응답들입니다. 이들을 참고하여 현재 청소년 상황에 최적화된 응답을 만들어주세요.

현재 상황: "{user_message}"
- 감정: {emotion}  
- 관계: {relationship}

검색된 전문가 응답들:
"""

        for i, resp_data in enumerate(retrieved_responses[:3], 1):  # 상위 3개만
            original_situation = resp_data.get('user_utterance', '')
            expert_response = resp_data.get('system_response', '')
            similarity = resp_data.get('similarity_score', 0)
            empathy_type = resp_data.get('empathy_label', '')

            combination_prompt += f"""
{i}. 유사도: {similarity:.2f} | 공감전략: {empathy_type}
   상황: "{original_situation}"
   전문가 응답: "{expert_response}"

"""

        combination_prompt += f"""
위 전문가 응답들의 핵심 접근법과 표현을 활용하여 다음 요구사항에 맞는 응답을 생성하세요:

1. 전문가 응답들의 공감 방식과 해결 접근법을 적극 활용
2. 청소년(13-19세)에게 맞는 친근하고 따뜻한 표현으로 변환
3. 구체적이고 실행 가능한 조언 포함
4. 150자 내외로 간결하게
5. 원본 응답들의 장점을 자연스럽게 조합

청소년 맞춤 응답:
"""

        try:
            response = await openai_client.create_completion(
                messages=[{"role": "user", "content": combination_prompt}],
                temperature=0.4,  # 조합에 약간의 창의성 허용
                max_tokens=300
            )

            combined_response = response.content.strip()

            return {
                "response": combined_response,
                "retrieval_used": True,
                "source": "multi_expert_combination",
                "adaptation_level": "combination",
                "num_sources": len(retrieved_responses),
                "avg_similarity": sum(r.get('similarity_score', 0) for r in retrieved_responses) / len(retrieved_responses),
                "source_strategies": [r.get('empathy_label', '') for r in retrieved_responses]
            }

        except Exception as e:
            logger.error(f"다중 응답 조합 실패: {e}")
            # 폴백: 가장 좋은 단일 응답 사용
            best_result = max(retrieved_responses, key=lambda x: x.get('similarity_score', 0))
            return await self._use_single_response_as_fallback(best_result, relationship, openai_client)

    async def _minimal_adult_to_teen_conversion(self,
                                              original_response: str,
                                              relationship: str,
                                              openai_client) -> str:
        """성인 응답을 청소년 맥락으로 최소 변환"""

        # 🔥 간단한 규칙 기반 변환 (빠르고 정확)
        teen_response = original_response

        # 기본 용어 변환
        conversion_map = {
            # 장소/환경
            "직장": "학교", "회사": "학교", "사무실": "교실",
            "업무": "공부", "일": "공부", "근무": "수업",

            # 관계
            "동료": "친구", "상사": "선생님", "부하직원": "후배",
            "거래처": "다른 학교", "고객": "친구",

            # 활동
            "회의": "수업", "프로젝트": "과제", "출장": "현장학습",
            "야근": "야자", "휴가": "방학", "퇴사": "전학",

            # 존댓말 → 반말 (친구 관계인 경우)
            "하세요": "해", "하시죠": "하자", "어떠세요": "어때",
            "해보세요": "해봐", "-습니다": "해", "-요": "야",
            "그렇군요": "그렇구나", "이해합니다": "이해해",
        }

        for adult_term, teen_term in conversion_map.items():
            teen_response = teen_response.replace(adult_term, teen_term)

        # 관계별 톤 조정
        if relationship in ["친구", "동급생"]:
            # 더 친근한 반말로
            teen_response = teen_response.replace("당신", "너")
            teen_response = teen_response.replace("귀하", "너")

        return teen_response

    async def _moderate_adaptation(self,
                                 original_response: str, user_message: str,
                                 emotion: str, relationship: str,
                                 openai_client) -> str:
        """중간 정도 적응 (구조 유지, 내용 조정)"""

        adaptation_prompt = f"""
다음 전문 상담사 응답을 현재 청소년 상황에 맞게 적응시켜주세요.
원본의 공감 방식과 해결 접근법은 유지하되, 청소년에게 맞는 표현으로 바꿔주세요.

원본 상담사 응답: "{original_response}"
현재 청소년 상황: "{user_message}" (감정: {emotion}, 관계: {relationship})

적응 요구사항:
1. 원본의 공감 방식과 해결 접근법 완전 유지
2. 13-19세 청소년에게 맞는 친근한 표현으로 변환
3. 원본보다 길어지지 말 것
4. 구체적이고 실행 가능한 조언 유지
5. 따뜻하고 지지적인 톤 유지

청소년 맞춤 응답:
"""

        try:
            response = await openai_client.create_completion(
                messages=[{"role": "user", "content": adaptation_prompt}],
                temperature=0.3,  # 낮은 창의성 - 원본 유지 중심
                max_tokens=250
            )

            return response.content.strip()

        except Exception as e:
            logger.error(f"중간 적응 실패: {e}")
            # 폴백: 최소 변환만
            return await self._minimal_adult_to_teen_conversion(original_response, relationship, openai_client)

    async def _use_single_response_as_fallback(self, result_data: Dict, relationship: str, openai_client) -> Dict[str, Any]:
        """단일 응답 폴백 사용"""
        original_response = result_data.get('system_response', '')
        adapted_response = await self._minimal_adult_to_teen_conversion(original_response, relationship, openai_client)

        return {
            "response": adapted_response,
            "retrieval_used": True,
            "source": "single_fallback",
            "adaptation_level": "minimal",
            "similarity_score": result_data.get('similarity_score', 0)
        }

    async def _fallback_response(self, user_message: str, emotion: str) -> str:
        """검색 결과가 없을 때 폴백 응답"""
        emotion_responses = {
            "기쁨": "정말 기쁜 일이구나! 함께 기뻐해줄게 😊 더 자세히 이야기해줄래?",
            "분노": "정말 화가 날 만한 상황이네. 네 마음 충분히 이해해. 어떤 일이 있었는지 말해줄래?",
            "슬픔": "많이 슬프겠다. 그런 기분일 때는 혼자 있기보다 누군가와 이야기하는 게 도움이 돼. 무슨 일인지 들어볼게.",
            "불안": "걱정이 많이 되는구나. 그런 마음 정말 이해해. 함께 좋은 방법을 찾아보자.",
            "상처": "마음이 많이 아플 것 같아. 힘들 때는 누군가와 이야기하는 게 중요해. 무엇이든 편하게 말해줘.",
            "당황": "갑작스러운 상황이라 당황스럽겠어. 천천히 정리해보자. 어떤 일이 있었는지 말해줄래?"
        }

        base_response = emotion_responses.get(emotion, "힘든 상황이구나. 네 마음을 이해해.")
        return f"{base_response}\n\n무엇이든 편하게 이야기해줘. 함께 해결방법을 찾아보자! 💪"


# 전역 챗봇 인스턴스
true_rag_chatbot = TrueRAGTeenChatbot()


@router.post("/teen-chat", response_model=TeenChatResponse)
async def teen_empathy_chat_with_true_rag(
    request: TeenChatRequest,
    openai_client = Depends(get_openai_client),
    processor = Depends(get_teen_empathy_processor)
):
    """
    🔥 마음이: 전문 상담사 조언을 활용하는 청소년 상담 채팅
    """
    try:
        start_time = time.time()
        logger.info(f"🧠 True RAG 채팅 요청: {request.message[:50]}...")
        logger.info(f"🚨 CRITICAL DEBUG: conversation_history = {request.conversation_history}")

        # 1. 감정 및 관계 맥락 감지
        emotion, relationship = await true_rag_chatbot.detect_emotion_and_context(
            request.message, openai_client
        )

        emotion_str = emotion.value if isinstance(emotion, EmotionType) else emotion
        relationship_str = relationship.value if relationship and isinstance(relationship, RelationshipType) else (relationship or "친구")

        logger.info(f"📊 감지된 맥락 - 감정: {emotion_str}, 관계: {relationship_str}")

        # 2. 유사한 대화 맥락 검색 (AI Hub 데이터에서)
        # 🔥 대화 맥락을 포함한 확장 쿼리 생성
        enhanced_query = request.message
        context_info = ""
        
        logger.info(f"🐛 DEBUG: conversation_history 길이: {len(getattr(request, 'conversation_history', []))}")
        logger.info(f"🐛 DEBUG: conversation_history 내용: {getattr(request, 'conversation_history', [])}")
        
        if request.conversation_history:
            # 최근 대화에서 핵심 키워드 추출
            recent_user_messages = [
                msg.content for msg in request.conversation_history[-4:] 
                if msg.role == "user"
            ]
            
            if recent_user_messages:
                # 이전 대화 맥락을 쿼리에 포함
                context_keywords = " ".join(recent_user_messages[-2:])  # 최근 2개
                enhanced_query = f"{context_keywords} {request.message}"
                context_info = f"이전 대화: {' → '.join(recent_user_messages)}"
                
                logger.info(f"📜 대화 맥락 포함 쿼리: '{enhanced_query[:100]}...'")
        
        similar_contexts = await true_rag_chatbot.search_similar_contexts(
            enhanced_query, emotion_str, relationship_str, processor
        )

        # 3. 🔥 핵심: 검색된 AI Hub 응답을 직접 활용해서 답변 생성
        response_data = await true_rag_chatbot.generate_response_from_search_results(
            request.message, similar_contexts, emotion_str, relationship_str, openai_client
        )

        # 4. 공감 전략 결정
        empathy_strategy = processor.get_empathy_strategy(emotion_str)

        # 5. 신뢰도 점수 계산
        confidence_score = min(0.95, max(0.8, 0.7 + len(similar_contexts) * 0.05))
        if response_data.get("similarity_score"):
            confidence_score = max(confidence_score, response_data["similarity_score"])

        # 6. 디버깅 정보 수집
        processing_time = (time.time() - start_time) * 1000
        debug_info = None

        if request.include_reasoning:
            debug_info = await _collect_true_rag_debug_info(
                request.message, emotion_str, relationship_str,
                similar_contexts, response_data, processing_time
            )

        # 7. 응답 구성
        # 🔥 최종 후처리 변환 (마지막 안전장치)
        final_response = response_data["response"]
        final_response = final_response.replace('자기야', '너')
        final_response = final_response.replace('자기가', '네가')  
        final_response = final_response.replace('자기도', '너도')
        final_response = final_response.replace('자기를', '너를')
        final_response = final_response.replace('자기의', '네')
        final_response = final_response.replace('무슨 공부이', '무슨 일이')
        final_response = final_response.replace('어떤 공부이', '어떤 일이')
        final_response = final_response.replace('공부이 있었', '일이 있었')
        final_response = final_response.replace('하세요', '해')
        final_response = final_response.replace('어떠세요', '어때')
        
        chat_response = TeenChatResponse(
            response=final_response,
            detected_emotion=emotion if isinstance(emotion, EmotionType) else EmotionType(emotion_str),
            empathy_strategy=[EmpathyStrategy(s) for s in empathy_strategy],
            similar_contexts=similar_contexts,
            react_steps=None,  # True RAG에서는 ReAct 단계 대신 검색-활용 과정
            confidence_score=confidence_score,
            response_metadata={
                "processing_time_ms": processing_time,
                "rag_method": "true_rag",
                "retrieval_used": response_data["retrieval_used"],
                "source": response_data["source"],
                "adaptation_level": response_data.get("adaptation_level", "unknown"),
                "similarity_score": response_data.get("similarity_score", 0),
                "num_sources": response_data.get("num_sources", len(similar_contexts)),
                "context_matches": len(similar_contexts),
                "relationship_context": relationship_str,
                "debug_info": debug_info
            }
        )

        logger.info(f"✅ True RAG 응답 완료 - 방식: {response_data['source']}, 신뢰도: {confidence_score:.2f}")
        return chat_response

    except Exception as e:
        logger.error(f"❌ True RAG 채팅 실패: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"True RAG 채팅 처리 중 오류가 발생했습니다: {str(e)}"
        )


async def _collect_true_rag_debug_info(message: str, emotion: str, relationship: str,
                                      similar_contexts: List[Dict], response_data: Dict,
                                      processing_time: float) -> Dict[str, Any]:
    """True RAG 전용 디버깅 정보 수집"""

    debug_info = {
        "timestamp": datetime.now().isoformat(),
        "total_processing_time_ms": processing_time,
        "rag_pipeline_steps": []
    }

    # 1. 검색 과정
    debug_info["rag_pipeline_steps"].append({
        "step": "1. AI Hub Vector Search",
        "details": {
            "query": message,
            "emotion_filter": emotion,
            "relationship_filter": relationship,
            "results_found": len(similar_contexts),
            "top_similarities": [ctx.get('similarity_score', 0) for ctx in similar_contexts[:3]]
        }
    })

    # 2. 검색 결과 품질 분석
    high_quality = [ctx for ctx in similar_contexts if ctx.get('similarity_score', 0) >= 0.85]
    medium_quality = [ctx for ctx in similar_contexts if 0.7 <= ctx.get('similarity_score', 0) < 0.85]

    debug_info["rag_pipeline_steps"].append({
        "step": "2. Search Quality Analysis",
        "details": {
            "high_quality_matches": len(high_quality),
            "medium_quality_matches": len(medium_quality),
            "selected_strategy": response_data.get("source", "unknown"),
            "adaptation_level": response_data.get("adaptation_level", "unknown")
        }
    })

    # 3. 원본 AI Hub 응답들
    if similar_contexts:
        original_responses = []
        for i, ctx in enumerate(similar_contexts[:3]):
            original_responses.append({
                "rank": i + 1,
                "similarity": ctx.get('similarity_score', 0),
                "original_situation": ctx.get('user_utterance', ''),
                "expert_response": ctx.get('system_response', ''),
                "empathy_strategy": ctx.get('empathy_label', '')
            })

        debug_info["rag_pipeline_steps"].append({
            "step": "3. Retrieved Expert Responses",
            "details": {
                "original_responses": original_responses
            }
        })

    # 4. 최종 응답 생성 과정
    debug_info["rag_pipeline_steps"].append({
        "step": "4. Response Generation",
        "details": {
            "generation_method": response_data.get("source", "unknown"),
            "original_response": response_data.get("original_response", ""),
            "final_response": response_data["response"],
            "transformation_applied": response_data.get("adaptation_level", "unknown")
        }
    })

    return debug_info


# ======================================
# 🔄 기존 API 엔드포인트들 유지 (호환성)
# ======================================

@router.get("/empathy-strategies")
async def get_empathy_strategies():
    """📋 감정별 공감 전략 조회"""
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
        "system_type": "expert_counseling",
        "data_source": "ai_hub_direct_utilization"
    }


@router.post("/analyze-emotion")
async def analyze_emotion(
    message: str,
    context: Optional[str] = None,
    openai_client = Depends(get_openai_client)
):
    """🎭 메시지 감정 분석"""
    try:
        analysis_result = await openai_client.analyze_emotion_and_context(
            text=message,
            additional_context=context
        )

        return {
            "message": message,
            "analysis": {
                "primary_emotion": analysis_result.primary_emotion.value,
                "emotion_confidence": analysis_result.emotion_confidence,
                "relationship_context": analysis_result.relationship_context.value if analysis_result.relationship_context else None,
                "recommended_strategies": [s.value for s in analysis_result.recommended_strategies],
                "analysis_details": analysis_result.analysis_details
            },
            "system_type": "expert_counseling",
            "analysis_time": datetime.now().isoformat()
        }

    except Exception as e:
        logger.error(f"감정 분석 실패: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"감정 분석 실패: {str(e)}"
        )


@router.post("/search-context")
async def search_similar_context(
    request: Dict[str, Any],
    processor = Depends(get_teen_empathy_processor)
):
    """🔍 유사 대화 맥락 검색 (True RAG용) - JSON Body 방식"""
    try:
        query = request.get("query", "")
        emotion = request.get("emotion")
        relationship = request.get("relationship") 
        top_k = request.get("top_k", 5)
        
        if not query:
            raise HTTPException(status_code=400, detail="query는 필수입니다")
        
        results = await processor.search_similar_contexts(
            query=query,
            emotion=emotion,
            relationship=relationship,
            top_k=top_k
        )

        return {
            "query": query,
            "filters": {
                "emotion": emotion,
                "relationship": relationship
            },
            "results": results,
            "total_found": len(results),
            "search_metadata": {
                "data_source": "professional_counseling",
                "system_type": "expert_counseling",
                "search_time": datetime.now().isoformat()
            }
        }

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"유사 맥락 검색 실패: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"유사 맥락 검색 실패: {str(e)}"
        )


# 나머지 기존 엔드포인트들은 동일하게 유지...
# (process-aihub-data, create-sample-data 등)