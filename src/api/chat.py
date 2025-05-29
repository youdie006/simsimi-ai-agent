"""
청소년 공감형 ReAct 패턴 채팅 API
AI Hub 데이터 기반 맥락 인식 + GPT-4 공감 응답 + 벡터 검색
"""

from fastapi import APIRouter, HTTPException, Depends, status
from typing import List, Optional, Dict, Any
from datetime import datetime
import json
import time

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


class TeenEmpathyChatbot:
    """청소년 공감형 채팅봇 (ReAct 패턴)"""

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
            results = await processor.search_similar_contexts(
                query=message,
                emotion=emotion,
                relationship=relationship,
                top_k=3
            )
            return results
        except Exception as e:
            logger.error(f"유사 맥락 검색 실패: {e}")
            return []

    async def generate_react_response(self, request: TeenChatRequest,
                                   emotion: str, relationship: str,
                                   similar_contexts: List[Dict[str, Any]],
                                   openai_client) -> tuple[str, List[ReActStep]]:
        """ReAct 패턴으로 공감형 응답 생성"""

        react_steps = []

        try:
            # ReAct 패턴으로 응답 생성
            response_text, raw_steps = await openai_client.generate_react_response(
                user_message=request.message,
                similar_contexts=similar_contexts,
                emotion=emotion,
                relationship=relationship
            )

            # ReActStep 객체로 변환
            if request.include_reasoning and raw_steps:
                for step in raw_steps:
                    react_steps.append(ReActStep(
                        step_type=step.get("step_type", "thought"),
                        content=step.get("content", ""),
                        timestamp=datetime.now().isoformat()
                    ))

            return response_text, react_steps

        except Exception as e:
            logger.error(f"ReAct 응답 생성 실패: {e}")
            # 폴백 응답
            fallback_response = self._generate_fallback_response(emotion, relationship)
            return fallback_response, []

    def _generate_fallback_response(self, emotion: str, relationship: str) -> str:
        """폴백 응답 생성"""
        empathy_phrases = {
            EmotionType.JOY.value: "정말 기쁜 일이구나! 함께 기뻐해줄게 😊",
            EmotionType.CONFUSION.value: "갑작스러운 상황이라 당황스럽겠다. 천천히 생각해보자.",
            EmotionType.ANGER.value: "정말 화가 날 만한 상황이네. 네 마음 충분히 이해해.",
            EmotionType.ANXIETY.value: "걱정이 많이 되는구나. 괜찮아, 함께 해결해보자.",
            EmotionType.HURT.value: "마음이 많이 아플 것 같아. 힘들 때는 누군가와 이야기하는 게 도움이 돼.",
            EmotionType.SADNESS.value: "정말 슬프겠다. 울고 싶을 때는 울어도 괜찮아."
        }

        base_response = empathy_phrases.get(emotion, "힘든 상황이구나. 네 마음을 이해해.")
        return f"{base_response}\n\n무엇이든 편하게 이야기해줘. 함께 좋은 방법을 찾아보자! 💪"


# 전역 챗봇 인스턴스
chatbot = TeenEmpathyChatbot()


@router.post("/teen-chat", response_model=TeenChatResponse)
async def teen_empathy_chat(
    request: TeenChatRequest,
    openai_client = Depends(get_openai_client),
    processor = Depends(get_teen_empathy_processor)
):
    """
    🧠 청소년 공감형 채팅 API (ReAct 패턴)

    - AI Hub 데이터 기반 맥락 인식
    - GPT-4 기반 공감형 응답 생성
    - 감정별 맞춤 전략 적용
    - ReAct 패턴 단계별 추론 (선택적)
    """
    try:
        start_time = time.time()
        logger.info(f"청소년 채팅 요청: {request.message[:50]}...")

        # 1. 감정 및 관계 맥락 감지
        if request.emotion and request.relationship_context:
            emotion = request.emotion
            relationship = request.relationship_context
        else:
            emotion, relationship = await chatbot.detect_emotion_and_context(
                request.message, openai_client
            )

        emotion_str = emotion.value if isinstance(emotion, EmotionType) else emotion
        relationship_str = relationship.value if relationship and isinstance(relationship, RelationshipType) else (relationship or "친구")

        logger.info(f"감지된 맥락 - 감정: {emotion_str}, 관계: {relationship_str}")

        # 2. 유사한 대화 맥락 검색
        similar_contexts = await chatbot.search_similar_contexts(
            request.message, emotion_str, relationship_str, processor
        )

        # 3. 공감 전략 결정
        empathy_strategy = processor.get_empathy_strategy(emotion_str)

        # 4. ReAct 패턴으로 응답 생성
        response_text, react_steps = await chatbot.generate_react_response(
            request, emotion_str, relationship_str, similar_contexts, openai_client
        )

        # 5. 신뢰도 점수 계산 (간단한 휴리스틱)
        confidence_score = min(0.95, 0.6 + len(similar_contexts) * 0.1)
        if emotion != EmotionType.ANXIETY:  # 감정이 명확할 때 신뢰도 높임
            confidence_score += 0.1

        # 6. 응답 구성
        processing_time = (time.time() - start_time) * 1000

        chat_response = TeenChatResponse(
            response=response_text,
            detected_emotion=emotion if isinstance(emotion, EmotionType) else EmotionType(emotion_str),
            empathy_strategy=[EmpathyStrategy(s) for s in empathy_strategy],
            similar_contexts=similar_contexts,
            react_steps=react_steps if request.include_reasoning else None,
            confidence_score=confidence_score,
            response_metadata={
                "processing_time_ms": processing_time,
                "context_matches": len(similar_contexts),
                "relationship_context": relationship_str,
                "vector_search_performed": True,
                "emotion_detection_method": "gpt4_analysis"
            }
        )

        logger.info(f"응답 생성 완료 - 신뢰도: {confidence_score:.2f}, 처리시간: {processing_time:.1f}ms")
        return chat_response

    except Exception as e:
        logger.error(f"청소년 채팅 처리 실패: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"채팅 처리 중 오류가 발생했습니다: {str(e)}"
        )


@router.get("/empathy-strategies")
async def get_empathy_strategies():
    """
    📋 감정별 공감 전략 조회

    - 6가지 감정별 추천 공감 전략
    - 청소년 상담에 효과적인 접근법
    """
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
        "available_emotions": chatbot.emotions,
        "available_relationships": chatbot.relationships,
        "available_strategies": chatbot.strategies,
        "strategy_descriptions": {
            EmpathyStrategy.ENCOURAGE.value: "응원하고 동기부여하기",
            EmpathyStrategy.AGREE.value: "함께 공감하고 이해하기",
            EmpathyStrategy.COMFORT.value: "따뜻하게 달래주기",
            EmpathyStrategy.ADVISE.value: "구체적 해결방안 제시하기"
        }
    }


@router.post("/analyze-emotion")
async def analyze_emotion(
    message: str,
    context: Optional[str] = None,
    openai_client = Depends(get_openai_client)
):
    """
    🎭 메시지 감정 분석

    - 텍스트에서 주요 감정 추출
    - 관계 맥락 파악
    - 공감 전략 추천
    """
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
    query: str,
    emotion: Optional[str] = None,
    relationship: Optional[str] = None,
    top_k: int = 5,
    processor = Depends(get_teen_empathy_processor)
):
    """
    🔍 유사 대화 맥락 검색

    - AI Hub 데이터에서 유사한 상황 찾기
    - 감정, 관계별 필터링
    - 과거 성공적인 대화 사례 제공
    """
    try:
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
                "data_source": "aihub",
                "search_time": datetime.now().isoformat()
            }
        }

    except Exception as e:
        logger.error(f"유사 맥락 검색 실패: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"유사 맥락 검색 실패: {str(e)}"
        )


@router.post("/process-aihub-data")
async def process_aihub_data(
    file_paths: List[str],
    processor = Depends(get_teen_empathy_processor)
):
    """
    📊 AI Hub 데이터 처리 및 인덱싱

    - AI Hub JSON 파일들을 청소년 맥락으로 변환
    - ChromaDB에 벡터 인덱싱
    - 처리 통계 반환
    """
    try:
        logger.info(f"AI Hub 데이터 처리 시작: {len(file_paths)}개 파일")

        stats = await processor.process_and_index_data(file_paths)

        return {
            "success": True,
            "processing_stats": {
                "total_sessions": stats.total_sessions,
                "teen_converted": stats.teen_converted,
                "emotion_distribution": stats.emotion_distribution,
                "relationship_distribution": stats.relationship_distribution,
                "empathy_distribution": stats.empathy_distribution,
                "processing_time": stats.processing_time
            },
            "next_steps": [
                "데이터 인덱싱 완료",
                "청소년 채팅 API 사용 가능",
                "/api/v1/chat/teen-chat 엔드포인트로 테스트 가능"
            ]
        }

    except Exception as e:
        logger.error(f"AI Hub 데이터 처리 실패: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"AI Hub 데이터 처리 실패: {str(e)}"
        )


@router.post("/process-all-aihub-data")
async def process_all_aihub_data(processor = Depends(get_teen_empathy_processor)):
    """
    📊 모든 AI Hub 데이터 일괄 처리 (JSON 파일들)

    - data/aihub/ 폴더의 모든 JSON 파일 자동 검색
    - 31,821개 세션 전체 처리 및 인덱싱
    """
    try:
        import glob

        # JSON 파일들 찾기 (모든 하위 폴더 포함)
        json_files = []

        # 직접 data/aihub/ 폴더의 JSON 파일들
        json_files.extend(glob.glob("/app/data/aihub/*.json"))

        # training 및 validation 폴더가 있다면 포함
        json_files.extend(glob.glob("/app/data/aihub/training/*.json"))
        json_files.extend(glob.glob("/app/data/aihub/validation/*.json"))

        # 모든 하위 폴더의 JSON 파일들
        json_files.extend(glob.glob("/app/data/aihub/**/*.json", recursive=True))

        # 중복 제거
        json_files = list(set(json_files))

        if not json_files:
            return {
                "success": False,
                "message": "AI Hub JSON 파일을 찾을 수 없습니다",
                "searched_paths": [
                    "/app/data/aihub/*.json",
                    "/app/data/aihub/training/*.json",
                    "/app/data/aihub/validation/*.json",
                    "/app/data/aihub/**/*.json"
                ],
                "instructions": [
                    "1. data/aihub/ 폴더에 AI Hub JSON 파일들을 복사",
                    "2. 파일명 예시: Empathy_기쁨_직장동료_14.json",
                    "3. 이 API를 다시 호출하여 자동 처리"
                ],
                "current_files": await _list_aihub_files()
            }

        logger.info(f"발견된 JSON 파일들: {len(json_files)}개")

        # 파일별 통계
        file_stats = {}
        for file_path in json_files[:5]:  # 처음 5개 파일만 미리보기
            try:
                filename = os.path.basename(file_path)
                # 파일명에서 감정, 관계 추출
                if 'Empathy_' in filename:
                    parts = filename.replace('Empathy_', '').replace('.json', '').split('_')
                    emotion = parts[0] if len(parts) > 0 else 'Unknown'
                    relation = parts[1] if len(parts) > 1 else 'Unknown'
                    file_stats[filename] = f"{emotion} + {relation}"
            except:
                file_stats[filename] = "파싱 실패"

        # 전체 데이터 처리
        stats = await processor.process_and_index_data(json_files)

        return {
            "success": True,
            "message": f"AI Hub 데이터 전체 처리 완료: {stats.total_sessions}개 세션",
            "file_summary": {
                "total_json_files": len(json_files),
                "sample_files": file_stats,
                "file_examples": [os.path.basename(f) for f in json_files[:3]]
            },
            "processing_stats": {
                "total_sessions": stats.total_sessions,
                "teen_converted": stats.teen_converted,
                "emotion_distribution": stats.emotion_distribution,
                "relationship_distribution": stats.relationship_distribution,
                "empathy_distribution": stats.empathy_distribution,
                "processing_time": stats.processing_time
            },
            "data_ready": True,
            "next_steps": [
                "✅ 모든 AI Hub JSON 데이터 인덱싱 완료",
                "✅ 청소년 채팅 API 사용 가능",
                "✅ 웹 인터페이스에서 채팅 테스트 가능",
                "🌐 http://localhost:8000 접속하여 채팅 시작"
            ]
        }

    except Exception as e:
        logger.error(f"전체 AI Hub 데이터 처리 실패: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"AI Hub 데이터 처리 실패: {str(e)}"
        )


async def _list_aihub_files():
    """data/aihub 폴더의 현재 파일들 나열"""
    try:
        import glob
        import os

        all_files = glob.glob("/app/data/aihub/**/*", recursive=True)
        files_info = []

        for file_path in all_files:
            if os.path.isfile(file_path):
                filename = os.path.basename(file_path)
                size = os.path.getsize(file_path)
                files_info.append({
                    "name": filename,
                    "size_mb": round(size / (1024*1024), 2),
                    "type": "JSON" if filename.endswith('.json') else "기타"
                })

        return files_info[:10]  # 처음 10개만
    except:
        return []


@router.get("/aihub-data-status")
async def get_aihub_data_status():
    """
    📋 AI Hub 데이터 상태 확인 (JSON 파일들)
    """
    try:
        import glob
        import os

        # JSON 파일 수 확인
        json_files = []
        json_files.extend(glob.glob("/app/data/aihub/*.json"))
        json_files.extend(glob.glob("/app/data/aihub/**/*.json", recursive=True))
        json_files = list(set(json_files))

        # 처리된 데이터 확인
        stats_file = "/app/data/teen_empathy_stats.json"
        processing_completed = os.path.exists(stats_file)

        processed_stats = {}
        if processing_completed:
            try:
                with open(stats_file, 'r', encoding='utf-8') as f:
                    processed_stats = json.load(f)
            except:
                pass

        # 파일별 미리보기
        file_preview = {}
        for file_path in json_files[:5]:
            try:
                filename = os.path.basename(file_path)
                size_mb = round(os.path.getsize(file_path) / (1024*1024), 2)

                # 파일명에서 정보 추출
                if 'Empathy_' in filename:
                    parts = filename.replace('Empathy_', '').replace('.json', '').split('_')
                    emotion = parts[0] if len(parts) > 0 else '?'
                    relation = parts[1] if len(parts) > 1 else '?'
                    file_preview[filename] = f"{emotion}+{relation} ({size_mb}MB)"
                else:
                    file_preview[filename] = f"({size_mb}MB)"
            except:
                continue

        return {
            "data_files": {
                "total_json_files": len(json_files),
                "file_preview": file_preview,
                "folder_structure": "data/aihub/ (모든 하위 폴더 포함)"
            },
            "processing_status": {
                "completed": processing_completed,
                "total_processed_sessions": processed_stats.get("total_sessions", 0),
                "teen_converted_count": processed_stats.get("teen_converted", 0),
                "processing_time": processed_stats.get("processing_time", "Unknown")
            },
            "current_files": await _list_aihub_files(),
            "recommendations": [
                "AI Hub에서 다운로드한 JSON 파일들을 data/aihub/ 폴더에 복사하세요",
                "파일명 예시: Empathy_기쁨_직장동료_14.json",
                "모든 파일이 준비되면 POST /process-all-aihub-data 호출하세요"
            ] if not processing_completed else [
                "✅ 데이터 처리 완료! 웹 채팅 인터페이스를 사용해보세요",
                "🌐 http://localhost:8000 접속하여 청소년과 대화해보세요"
            ]
        }

    except Exception as e:
        logger.error(f"데이터 상태 확인 실패: {e}")
        return {
            "error": str(e),
            "message": "데이터 상태 확인 중 오류가 발생했습니다"
        }
    """
    🎯 샘플 데이터 생성 및 처리
    
    - 테스트용 샘플 AI Hub 데이터 생성
    - 청소년 맥락 변환 데모
    - 벡터 인덱싱까지 완료
    """
    try:
        # 샘플 데이터 생성
        sample_data = await processor.create_sample_data()

        # 임시 파일로 저장
        import tempfile
        with tempfile.NamedTemporaryFile(mode='w', suffix='.json', delete=False, encoding='utf-8') as f:
            json.dump(sample_data, f, ensure_ascii=False, indent=2)
            temp_file_path = f.name

        # 데이터 처리
        stats = await processor.process_and_index_data([temp_file_path])

        # 임시 파일 삭제
        import os
        os.unlink(temp_file_path)

        return {
            "success": True,
            "message": "샘플 데이터 생성 및 처리 완료",
            "sample_data": sample_data,
            "processing_stats": {
                "total_sessions": stats.total_sessions,
                "teen_converted": stats.teen_converted,
                "processing_time": stats.processing_time
            },
            "ready_for_testing": True
        }

    except Exception as e:
        logger.error(f"샘플 데이터 생성 실패: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"샘플 데이터 생성 실패: {str(e)}"
        )


@router.get("/chat-demo")
async def get_chat_demo():
    """
    💬 채팅 API 데모 및 사용법

    - 다양한 테스트 케이스 제공
    - API 사용 예시
    - 성능 최적화 팁
    """
    return {
        "demo_conversations": [
            {
                "scenario": "친구 관계 갈등",
                "example_request": {
                    "message": "친구가 나만 빼고 놀러 가서 서운해",
                    "include_reasoning": True
                },
                "expected_emotion": "상처",
                "expected_strategies": ["위로", "격려"]
            },
            {
                "scenario": "부모님과의 갈등",
                "example_request": {
                    "message": "부모님이 성적 때문에 계속 잔소리하셔서 스트레스 받아",
                    "emotion": "분노",
                    "relationship_context": "부모님"
                },
                "expected_strategies": ["위로", "조언"]
            },
            {
                "scenario": "좋은 소식 공유",
                "example_request": {
                    "message": "시험을 정말 잘 봐서 기분이 너무 좋아!",
                    "emotion": "기쁨"
                },
                "expected_strategies": ["격려", "동조"]
            }
        ],
        "api_usage_tips": [
            "구체적인 상황을 포함해서 메시지를 작성하면 더 정확한 분석이 가능합니다",
            "include_reasoning=true로 설정하면 AI의 사고 과정을 볼 수 있습니다",
            "emotion과 relationship_context를 미리 제공하면 처리 속도가 빨라집니다",
            "conversation_history를 포함하면 더 맥락에 맞는 응답을 받을 수 있습니다"
        ],
        "response_quality_factors": [
            "유사한 맥락의 AI Hub 데이터 존재 여부",
            "감정 감지 정확도",
            "관계 맥락 파악 정확성",
            "GPT-4 모델의 응답 품질"
        ]
    }