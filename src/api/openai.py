"""
OpenAI API 라우터
GPT-4 관련 엔드포인트들 (채팅, 감정분석 등)
"""

from fastapi import APIRouter, HTTPException, Depends, status
from typing import List, Dict, Any
from loguru import logger

from ..services.openai_client import get_openai_client
from ..models.function_models import (
    OpenAICompletionRequest, OpenAICompletionResponse,
    EmotionAnalysisRequest, EmotionAnalysisResponse,
    ChatMessage, SystemHealthCheck
)


router = APIRouter()


@router.post("/completion", response_model=OpenAICompletionResponse)
async def create_completion(
    request: OpenAICompletionRequest,
    openai_client = Depends(get_openai_client)
):
    """
    🤖 GPT-4 채팅 완성 생성

    - 일반적인 GPT-4 채팅 완성
    - 사용자 정의 모델, 온도, 토큰 수 설정 가능
    - 스트리밍 지원 (선택적)
    """
    try:
        logger.info(f"GPT-4 완성 요청 - 모델: {request.model}, 메시지 수: {len(request.messages)}")

        # ChatMessage를 dict로 변환
        messages = [
            {"role": msg.role.value, "content": msg.content}
            for msg in request.messages
        ]

        response = await openai_client.create_completion(
            messages=messages,
            model=request.model,
            temperature=request.temperature,
            max_tokens=request.max_tokens,
            top_p=request.top_p
        )

        return response

    except Exception as e:
        logger.error(f"GPT-4 완성 생성 실패: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"GPT-4 완성 생성 중 오류가 발생했습니다: {str(e)}"
        )


@router.post("/teen-empathy", response_model=str)
async def create_teen_empathy_response(
    user_message: str,
    conversation_history: List[ChatMessage] = None,
    context_info: str = None,
    openai_client = Depends(get_openai_client)
):
    """
    💙 청소년 공감형 응답 생성

    - 청소년 전용 공감 시스템 프롬프트 적용
    - 대화 히스토리 및 맥락 정보 활용
    - 따뜻하고 지지적인 응답 생성
    """
    try:
        logger.info(f"청소년 공감 응답 요청: '{user_message[:50]}...'")

        response = await openai_client.create_teen_empathy_response(
            user_message=user_message,
            conversation_history=conversation_history,
            context_info=context_info
        )

        return response

    except Exception as e:
        logger.error(f"청소년 공감 응답 생성 실패: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"청소년 공감 응답 생성 중 오류가 발생했습니다: {str(e)}"
        )


@router.post("/analyze-emotion", response_model=EmotionAnalysisResponse)
async def analyze_emotion(
    request: EmotionAnalysisRequest,
    openai_client = Depends(get_openai_client)
):
    """
    🎭 감정 및 맥락 분석

    - 텍스트에서 주요 감정 추출
    - 관계 맥락 파악 (부모님, 친구, 형제자매 등)
    - 적절한 공감 전략 추천
    """
    try:
        logger.info(f"감정 분석 요청: '{request.text[:50]}...'")

        response = await openai_client.analyze_emotion_and_context(
            text=request.text,
            additional_context=request.context
        )

        return response

    except Exception as e:
        logger.error(f"감정 분석 실패: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"감정 분석 중 오류가 발생했습니다: {str(e)}"
        )


@router.post("/react-response")
async def generate_react_response(
    user_message: str,
    similar_contexts: List[Dict[str, Any]] = None,
    emotion: str = None,
    relationship: str = None,
    openai_client = Depends(get_openai_client)
):
    """
    🧠 ReAct 패턴 응답 생성

    - Thought → Action → Observation → Response
    - 단계별 추론 과정 포함
    - 유사 맥락 정보 활용
    """
    try:
        logger.info(f"ReAct 응답 요청: '{user_message[:50]}...'")

        response_text, react_steps = await openai_client.generate_react_response(
            user_message=user_message,
            similar_contexts=similar_contexts or [],
            emotion=emotion,
            relationship=relationship
        )

        return {
            "response": response_text,
            "react_steps": react_steps,
            "metadata": {
                "emotion": emotion,
                "relationship": relationship,
                "context_count": len(similar_contexts) if similar_contexts else 0
            }
        }

    except Exception as e:
        logger.error(f"ReAct 응답 생성 실패: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"ReAct 응답 생성 중 오류가 발생했습니다: {str(e)}"
        )


@router.get("/models")
async def list_available_models():
    """
    📋 사용 가능한 모델 목록

    - 지원하는 OpenAI 모델들
    - 각 모델의 특징 및 사용 권장사항
    """
    return {
        "available_models": [
            {
                "name": "gpt-4",
                "description": "가장 강력한 모델, 복잡한 추론에 최적",
                "recommended_for": ["청소년 공감 상담", "복잡한 맥락 이해"],
                "max_tokens": 8192,
                "cost": "높음"
            },
            {
                "name": "gpt-4-turbo",
                "description": "빠르고 효율적인 GPT-4 버전",
                "recommended_for": ["실시간 채팅", "일반적인 상담"],
                "max_tokens": 128000,
                "cost": "중간"
            },
            {
                "name": "gpt-3.5-turbo",
                "description": "빠르고 경제적인 모델",
                "recommended_for": ["간단한 질문", "테스트용"],
                "max_tokens": 4096,
                "cost": "낮음"
            }
        ],
        "current_default": "gpt-4",
        "recommendation": "청소년 공감형 상담에는 gpt-4를 권장합니다"
    }


@router.get("/health", response_model=SystemHealthCheck)
async def openai_health_check(openai_client = Depends(get_openai_client)):
    """
    💊 OpenAI 서비스 헬스 체크

    - API 연결 상태 확인
    - 응답 시간 측정
    - 서비스 가용성 점검
    """
    try:
        import time
        start_time = time.time()

        # 간단한 테스트 요청으로 연결 확인
        test_response = await openai_client.create_completion(
            messages=[{"role": "user", "content": "Hello"}],
            max_tokens=5,
            temperature=0
        )

        response_time_ms = (time.time() - start_time) * 1000

        return SystemHealthCheck(
            status="healthy",
            services={
                "openai_api": True,
                "gpt4_model": True,
                "embedding_generation": True
            },
            response_time_ms=response_time_ms,
            version="1.0.0"
        )

    except Exception as e:
        logger.error(f"OpenAI 헬스 체크 실패: {e}")
        return SystemHealthCheck(
            status="unhealthy",
            services={
                "openai_api": False,
                "gpt4_model": False,
                "embedding_generation": False
            },
            response_time_ms=0.0,
            version="1.0.0"
        )


@router.get("/usage-stats")
async def get_usage_stats():
    """
    📊 OpenAI API 사용 통계

    - 토큰 사용량 추정
    - 비용 관련 정보
    """
    return {
        "current_session": {
            "requests_made": "실시간 추적 필요",
            "tokens_used": "실시간 추적 필요",
            "estimated_cost": "실시간 추적 필요"
        },
        "cost_info": {
            "gpt-4": {
                "input_per_1k_tokens": "$0.03",
                "output_per_1k_tokens": "$0.06"
            },
            "gpt-4-turbo": {
                "input_per_1k_tokens": "$0.01",
                "output_per_1k_tokens": "$0.03"
            },
            "gpt-3.5-turbo": {
                "input_per_1k_tokens": "$0.0015",
                "output_per_1k_tokens": "$0.002"
            }
        },
        "optimization_tips": [
            "적절한 max_tokens 설정으로 비용 절약",
            "간단한 작업은 gpt-3.5-turbo 사용",
            "시스템 프롬프트 최적화로 토큰 절약",
            "불필요한 대화 히스토리 제거"
        ]
    }


@router.post("/test-empathy")
async def test_empathy_response(
    test_message: str = "친구가 나를 무시하는 것 같아서 기분이 나빠",
    openai_client = Depends(get_openai_client)
):
    """
    🧪 공감형 응답 테스트

    - 청소년 공감형 시스템의 응답 품질 테스트
    - 다양한 테스트 케이스 제공
    """
    try:
        # 감정 분석
        emotion_result = await openai_client.analyze_emotion_and_context(test_message)

        # 공감형 응답 생성
        empathy_response = await openai_client.create_teen_empathy_response(test_message)

        # ReAct 응답 생성
        react_response, react_steps = await openai_client.generate_react_response(
            user_message=test_message,
            emotion=emotion_result.primary_emotion.value,
            relationship=emotion_result.relationship_context.value if emotion_result.relationship_context else None
        )

        return {
            "test_input": test_message,
            "emotion_analysis": {
                "primary_emotion": emotion_result.primary_emotion.value,
                "confidence": emotion_result.emotion_confidence,
                "relationship": emotion_result.relationship_context.value if emotion_result.relationship_context else None,
                "strategies": [s.value for s in emotion_result.recommended_strategies]
            },
            "empathy_response": empathy_response,
            "react_response": {
                "response": react_response,
                "steps": react_steps
            },
            "test_info": {
                "response_quality": "수동 평가 필요",
                "empathy_level": "수동 평가 필요",
                "actionability": "수동 평가 필요"
            }
        }

    except Exception as e:
        logger.error(f"공감 응답 테스트 실패: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"테스트 실행 중 오류가 발생했습니다: {str(e)}"
        )