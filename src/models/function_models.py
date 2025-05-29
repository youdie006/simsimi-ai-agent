"""
OpenAI 및 기타 기능 관련 데이터 모델들
GPT-4 API 호출, 채팅, 감정 분석 등의 모델들
"""

from pydantic import BaseModel, Field
from typing import List, Optional, Dict, Any, Literal
from datetime import datetime
from enum import Enum


class ChatRole(str, Enum):
    """채팅 역할 열거형"""
    SYSTEM = "system"
    USER = "user"
    ASSISTANT = "assistant"


class ChatMessage(BaseModel):
    """채팅 메시지 모델"""
    role: ChatRole = Field(..., description="메시지 역할")
    content: str = Field(..., description="메시지 내용", min_length=1)
    timestamp: Optional[str] = Field(default_factory=lambda: datetime.now().isoformat(), description="메시지 시간")

    class Config:
        json_schema_extra = {
            "example": {
                "role": "user",
                "content": "친구가 나를 무시하는 것 같아서 기분이 나빠",
                "timestamp": "2024-01-01T12:00:00"
            }
        }


class OpenAICompletionRequest(BaseModel):
    """OpenAI 완성 요청 모델"""
    messages: List[ChatMessage] = Field(..., description="대화 메시지 목록")
    model: str = Field(default="gpt-4", description="사용할 모델")
    temperature: float = Field(default=0.7, description="응답 창의성", ge=0, le=2)
    max_tokens: int = Field(default=500, description="최대 토큰 수", ge=1, le=4000)
    top_p: float = Field(default=1.0, description="확률 임계값", ge=0, le=1)
    stream: bool = Field(default=False, description="스트리밍 여부")

    class Config:
        json_schema_extra = {
            "example": {
                "messages": [
                    {
                        "role": "system",
                        "content": "당신은 청소년 상담사입니다."
                    },
                    {
                        "role": "user",
                        "content": "친구와 싸웠어요"
                    }
                ],
                "model": "gpt-4",
                "temperature": 0.7,
                "max_tokens": 300
            }
        }


class OpenAICompletionResponse(BaseModel):
    """OpenAI 완성 응답 모델"""
    content: str = Field(..., description="생성된 응답 내용")
    model: str = Field(..., description="사용된 모델")
    tokens_used: int = Field(..., description="사용된 토큰 수")
    processing_time_ms: float = Field(..., description="처리 시간 (밀리초)")
    finish_reason: str = Field(..., description="완료 이유")

    class Config:
        json_schema_extra = {
            "example": {
                "content": "친구와의 갈등은 정말 힘들겠다. 어떤 일로 싸우게 됐는지 말해줄래?",
                "model": "gpt-4",
                "tokens_used": 45,
                "processing_time_ms": 1200.5,
                "finish_reason": "stop"
            }
        }


class EmotionType(str, Enum):
    """감정 유형 열거형"""
    JOY = "기쁨"
    CONFUSION = "당황"
    ANGER = "분노"
    ANXIETY = "불안"
    HURT = "상처"
    SADNESS = "슬픔"


class RelationshipType(str, Enum):
    """관계 유형 열거형"""
    PARENT = "부모님"
    FRIEND = "친구"
    SIBLING = "형제자매"
    CRUSH = "좋아하는 사람"
    CLASSMATE = "동급생"


class EmpathyStrategy(str, Enum):
    """공감 전략 열거형"""
    ENCOURAGE = "격려"
    AGREE = "동조"
    COMFORT = "위로"
    ADVISE = "조언"


class EmotionAnalysisRequest(BaseModel):
    """감정 분석 요청 모델"""
    text: str = Field(..., description="분석할 텍스트", min_length=1, max_length=1000)
    context: Optional[str] = Field(default=None, description="추가 맥락 정보")

    class Config:
        json_schema_extra = {
            "example": {
                "text": "친구가 나만 빼고 놀러 가서 서운해",
                "context": "학교 친구들과의 관계"
            }
        }


class EmotionAnalysisResponse(BaseModel):
    """감정 분석 응답 모델"""
    primary_emotion: EmotionType = Field(..., description="주요 감정")
    emotion_confidence: float = Field(..., description="감정 신뢰도", ge=0, le=1)
    relationship_context: Optional[RelationshipType] = Field(default=None, description="관계 맥락")
    recommended_strategies: List[EmpathyStrategy] = Field(..., description="추천 공감 전략")
    analysis_details: Dict[str, Any] = Field(default={}, description="분석 상세 정보")

    class Config:
        json_schema_extra = {
            "example": {
                "primary_emotion": "상처",
                "emotion_confidence": 0.85,
                "relationship_context": "친구",
                "recommended_strategies": ["위로", "격려"],
                "analysis_details": {
                    "keywords": ["빼고", "서운해"],
                    "intensity": "medium"
                }
            }
        }


class TeenChatRequest(BaseModel):
    """청소년 채팅 요청 모델"""
    message: str = Field(..., description="사용자 메시지", min_length=1, max_length=1000)
    emotion: Optional[EmotionType] = Field(default=None, description="감정 상태")
    relationship_context: Optional[RelationshipType] = Field(default=None, description="관계 맥락")
    conversation_history: List[ChatMessage] = Field(default=[], description="대화 히스토리")
    include_reasoning: bool = Field(default=False, description="추론 과정 포함 여부")
    max_context_length: int = Field(default=5, description="참조할 이전 대화 수", ge=1, le=10)

    class Config:
        json_schema_extra = {
            "example": {
                "message": "부모님이 성적 때문에 계속 잔소리하셔서 스트레스 받아",
                "emotion": "분노",
                "relationship_context": "부모님",
                "conversation_history": [
                    {
                        "role": "user",
                        "content": "요즘 공부가 너무 힘들어"
                    }
                ],
                "include_reasoning": True
            }
        }


class ReActStep(BaseModel):
    """ReAct 추론 단계 모델"""
    step_type: Literal["thought", "action", "observation"] = Field(..., description="단계 유형")
    content: str = Field(..., description="단계 내용")
    timestamp: str = Field(default_factory=lambda: datetime.now().isoformat(), description="단계 시간")

    class Config:
        json_schema_extra = {
            "example": {
                "step_type": "thought",
                "content": "사용자가 부모님의 성적 압박으로 스트레스를 받고 있다. 분노 감정이 주요하다.",
                "timestamp": "2024-01-01T12:00:00"
            }
        }


class TeenChatResponse(BaseModel):
    """청소년 채팅 응답 모델"""
    response: str = Field(..., description="공감형 응답")
    detected_emotion: EmotionType = Field(..., description="감지된 감정")
    empathy_strategy: List[EmpathyStrategy] = Field(..., description="적용된 공감 전략")
    similar_contexts: List[Dict[str, Any]] = Field(default=[], description="유사한 대화 맥락")
    react_steps: Optional[List[ReActStep]] = Field(default=None, description="ReAct 추론 과정")
    confidence_score: float = Field(..., description="응답 신뢰도", ge=0, le=1)
    response_metadata: Dict[str, Any] = Field(default={}, description="응답 메타데이터")

    class Config:
        json_schema_extra = {
            "example": {
                "response": "부모님의 성적 압박이 정말 스트레스가 많이 될 것 같아. 네 마음 충분히 이해해. 부모님과 솔직하게 대화해보는 건 어떨까?",
                "detected_emotion": "분노",
                "empathy_strategy": ["위로", "조언"],
                "similar_contexts": [
                    {
                        "user_utterance": "부모님이 성적 때문에 잔소리해요",
                        "system_response": "부모님의 기대가 부담스러우시겠어요",
                        "similarity_score": 0.92
                    }
                ],
                "confidence_score": 0.88,
                "response_metadata": {
                    "processing_time": "2024-01-01T12:00:00",
                    "context_matches": 3
                }
            }
        }


class SystemHealthCheck(BaseModel):
    """시스템 헬스 체크 모델"""
    status: Literal["healthy", "degraded", "unhealthy"] = Field(..., description="시스템 상태")
    services: Dict[str, bool] = Field(..., description="서비스별 상태")
    response_time_ms: float = Field(..., description="응답 시간 (밀리초)")
    timestamp: str = Field(default_factory=lambda: datetime.now().isoformat(), description="체크 시간")
    version: str = Field(..., description="시스템 버전")

    class Config:
        json_schema_extra = {
            "example": {
                "status": "healthy",
                "services": {
                    "openai": True,
                    "chromadb": True,
                    "embedding_model": True
                },
                "response_time_ms": 25.5,
                "timestamp": "2024-01-01T12:00:00",
                "version": "2.0.0"
            }
        }


class ErrorResponse(BaseModel):
    """에러 응답 모델"""
    error: str = Field(..., description="에러 유형")
    message: str = Field(..., description="에러 메시지")
    details: Optional[Dict[str, Any]] = Field(default=None, description="에러 상세 정보")
    timestamp: str = Field(default_factory=lambda: datetime.now().isoformat(), description="에러 발생 시간")
    request_id: Optional[str] = Field(default=None, description="요청 ID")

    class Config:
        json_schema_extra = {
            "example": {
                "error": "ValidationError",
                "message": "입력 데이터가 올바르지 않습니다",
                "details": {
                    "field": "message",
                    "issue": "필수 필드가 누락되었습니다"
                },
                "timestamp": "2024-01-01T12:00:00",
                "request_id": "req_123456"
            }
        }