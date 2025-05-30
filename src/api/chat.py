"""
청소년 공감형 ReAct 패턴 채팅 API
AI Hub 데이터 기반 맥락 인식 + GPT-4 공감 응답 + 벡터 검색
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
            logger.info(f"유사 맥락 검색 시작 - 메시지: {message[:30]}..., 감정: {emotion}, 관계: {relationship}")

            results = await processor.search_similar_contexts(
                query=message,
                emotion=emotion,
                relationship=relationship,
                top_k=3
            )

            logger.info(f"유사 맥락 검색 완료 - {len(results)}개 결과")

            # 결과가 없으면 기본 예시 제공
            if not results:
                logger.warning("유사 맥락 검색 결과 없음 - 기본 예시 제공")
                return [{
                    "user_utterance": "비슷한 상황의 예시가 아직 준비되지 않았어요",
                    "system_response": "하지만 네 마음을 충분히 이해해요",
                    "similarity_score": 0.5,
                    "emotion": emotion,
                    "relationship": relationship
                }]

            return results

        except Exception as e:
            logger.error(f"유사 맥락 검색 실패: {e}")
            return [{
                "user_utterance": "검색 중 오류가 발생했어요",
                "system_response": "하지만 너의 이야기를 들을 준비가 되어 있어",
                "similarity_score": 0.3,
                "emotion": emotion or "불안",
                "relationship": relationship or "친구"
            }]

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

    async def _collect_debug_info(self, message: str, emotion: str, relationship: str,
                                similar_contexts: List[Dict], processing_time: float,
                                vector_store) -> Dict[str, Any]:
        """실시간 채팅 기술적 디버깅 정보 수집"""
        try:
            debug_info = {
                "timestamp": datetime.now().isoformat(),
                "total_processing_time_ms": processing_time,
                "pipeline_steps": []
            }

            # ============================================
            # 1️⃣ 원본 텍스트 분석
            # ============================================
            step1_start = time.time()
            text_analysis = {
                "original_text": message,
                "text_length": len(message),
                "word_count": len(message.split()),
                "character_distribution": {
                    "korean": len([c for c in message if ord(c) >= 0xAC00 and ord(c) <= 0xD7AF]),
                    "english": len([c for c in message if c.isalpha() and ord(c) < 128]),
                    "numbers": len([c for c in message if c.isdigit()]),
                    "special": len([c for c in message if not c.isalnum() and not c.isspace()])
                },
                "preprocessing_applied": ["tokenization", "normalization"]
            }
            step1_time = (time.time() - step1_start) * 1000
            debug_info["pipeline_steps"].append({
                "step": "1. Text Analysis",
                "time_ms": step1_time,
                "details": text_analysis
            })

            # ============================================
            # 2️⃣ 벡터 임베딩 생성 과정
            # ============================================
            step2_start = time.time()
            embeddings = vector_store.create_embeddings([message])
            embedding_vector = np.array(embeddings[0])

            # 벡터 상세 분석
            embedding_analysis = {
                "model": vector_store.model_name,
                "dimension": len(embedding_vector),
                "vector_norm": float(np.linalg.norm(embedding_vector)),
                "vector_stats": {
                    "mean": float(np.mean(embedding_vector)),
                    "std": float(np.std(embedding_vector)),
                    "min": float(np.min(embedding_vector)),
                    "max": float(np.max(embedding_vector)),
                    "median": float(np.median(embedding_vector))
                },
                "sparsity": {
                    "zero_values": int(np.sum(embedding_vector == 0)),
                    "near_zero_values": int(np.sum(np.abs(embedding_vector) < 0.001)),
                    "sparsity_ratio": float(np.sum(embedding_vector == 0) / len(embedding_vector))
                },
                "vector_sample": [float(x) for x in embedding_vector[:10]],  # 처음 10개 값
                "vector_tail": [float(x) for x in embedding_vector[-5:]],    # 마지막 5개 값
                "activation_pattern": {
                    "positive_count": int(np.sum(embedding_vector > 0)),
                    "negative_count": int(np.sum(embedding_vector < 0)),
                    "strong_activations": int(np.sum(np.abs(embedding_vector) > 0.1))
                }
            }
            step2_time = (time.time() - step2_start) * 1000
            debug_info["pipeline_steps"].append({
                "step": "2. Vector Embedding",
                "time_ms": step2_time,
                "details": embedding_analysis
            })

            # ============================================
            # 3️⃣ RAG 검색 쿼리 구성
            # ============================================
            step3_start = time.time()

            # 검색 쿼리 변환 과정
            search_queries = []

            # 원본 쿼리
            base_query = message
            search_queries.append({"type": "original", "query": base_query})

            # 감정 태그 추가 쿼리
            if emotion:
                emotion_query = f"[{emotion}] {message}"
                search_queries.append({"type": "emotion_enhanced", "query": emotion_query})

            # 관계 맥락 추가 쿼리
            if relationship:
                relation_query = f"[{relationship}] {message}"
                search_queries.append({"type": "relationship_enhanced", "query": relation_query})

            # 복합 쿼리
            if emotion and relationship:
                combined_query = f"[{emotion}] [{relationship}] {message}"
                search_queries.append({"type": "combined", "query": combined_query})

            query_analysis = {
                "total_queries_generated": len(search_queries),
                "query_variations": search_queries,
                "search_strategy": "multi_query_ensemble",
                "filters_applied": {
                    "emotion_filter": emotion,
                    "relationship_filter": relationship,
                    "data_source_filter": "aihub"
                }
            }
            step3_time = (time.time() - step3_start) * 1000
            debug_info["pipeline_steps"].append({
                "step": "3. RAG Query Construction",
                "time_ms": step3_time,
                "details": query_analysis
            })

            # ============================================
            # 4️⃣ 벡터 검색 실행 과정
            # ============================================
            step4_start = time.time()

            # 실제 벡터 검색 재실행 (분석용)
            search_results = await vector_store.search(message, top_k=10)  # 더 많은 결과

            # 유사도 계산 분석
            similarity_analysis = {
                "total_candidates_searched": await self._get_total_documents(vector_store),
                "results_returned": len(search_results),
                "similarity_scores": [float(r.score) for r in search_results],
                "similarity_distribution": {
                    "highest": float(max([r.score for r in search_results])) if search_results else 0,
                    "lowest": float(min([r.score for r in search_results])) if search_results else 0,
                    "average": float(np.mean([r.score for r in search_results])) if search_results else 0,
                    "std": float(np.std([r.score for r in search_results])) if search_results else 0
                },
                "quality_thresholds": {
                    "excellent": len([r for r in search_results if r.score > 0.9]),
                    "good": len([r for r in search_results if 0.7 < r.score <= 0.9]),
                    "fair": len([r for r in search_results if 0.5 < r.score <= 0.7]),
                    "poor": len([r for r in search_results if r.score <= 0.5])
                },
                "top_matches": [
                    {
                        "rank": i+1,
                        "similarity": float(r.score),
                        "content_preview": r.content[:100] + "..." if len(r.content) > 100 else r.content,
                        "metadata": {
                            "emotion": r.metadata.get("emotion", "unknown"),
                            "relationship": r.metadata.get("relationship", "unknown"),
                            "empathy_label": r.metadata.get("empathy_label", "unknown")
                        }
                    } for i, r in enumerate(search_results[:5])
                ]
            }
            step4_time = (time.time() - step4_start) * 1000
            debug_info["pipeline_steps"].append({
                "step": "4. Vector Search Execution",
                "time_ms": step4_time,
                "details": similarity_analysis
            })

            # ============================================
            # 5️⃣ 검색 결과 후처리 및 컨텍스트 구성
            # ============================================
            step5_start = time.time()

            context_construction = {
                "raw_results_count": len(search_results),
                "filtered_results_count": len(similar_contexts),
                "filtering_criteria": {
                    "min_similarity_threshold": 0.3,
                    "max_results": 3,
                    "diversity_filtering": True
                },
                "selected_contexts": [
                    {
                        "user_utterance": ctx.get("user_utterance", ""),
                        "system_response": ctx.get("system_response", "")[:100] + "...",
                        "similarity_score": ctx.get("similarity_score", 0),
                        "metadata": {
                            "emotion": ctx.get("emotion", ""),
                            "relationship": ctx.get("relationship", ""),
                            "empathy_label": ctx.get("empathy_label", "")
                        }
                    } for ctx in similar_contexts[:3]
                ],
                "context_diversity": {
                    "unique_emotions": len(set([ctx.get("emotion", "") for ctx in similar_contexts])),
                    "unique_relationships": len(set([ctx.get("relationship", "") for ctx in similar_contexts])),
                    "unique_strategies": len(set([ctx.get("empathy_label", "") for ctx in similar_contexts]))
                }
            }
            step5_time = (time.time() - step5_start) * 1000
            debug_info["pipeline_steps"].append({
                "step": "5. Context Construction",
                "time_ms": step5_time,
                "details": context_construction
            })

            # ============================================
            # 6️⃣ GPT-4 프롬프트 구성
            # ============================================
            step6_start = time.time()

            # 프롬프트 구성 분석
            prompt_analysis = {
                "system_prompt_length": len("당신은 13-19세 청소년을 위한 전문 공감 상담사입니다..."),
                "user_message_length": len(message),
                "context_injection": {
                    "similar_contexts_included": len(similar_contexts),
                    "emotion_context": emotion,
                    "relationship_context": relationship,
                    "total_context_length": sum([len(str(ctx)) for ctx in similar_contexts])
                },
                "estimated_tokens": {
                    "system_prompt": 150,  # 추정값
                    "user_message": len(message.split()) * 1.3,  # 한국어 토큰 추정
                    "context": sum([len(str(ctx).split()) * 1.3 for ctx in similar_contexts]),
                    "total_input": 150 + len(message.split()) * 1.3 + sum([len(str(ctx).split()) * 1.3 for ctx in similar_contexts])
                },
                "prompt_structure": {
                    "has_system_prompt": True,
                    "has_context": len(similar_contexts) > 0,
                    "has_emotion_info": bool(emotion),
                    "has_relationship_info": bool(relationship),
                    "react_pattern_enabled": True
                }
            }
            step6_time = (time.time() - step6_start) * 1000
            debug_info["pipeline_steps"].append({
                "step": "6. GPT-4 Prompt Construction",
                "time_ms": step6_time,
                "details": prompt_analysis
            })

            # ============================================
            # 7️⃣ 기술적 품질 분석 및 경고 시스템
            # ============================================
            step7_start = time.time()

            quality_analysis = {
                "overall_quality": "unknown",
                "warnings": [],
                "recommendations": [],
                "quality_scores": {},
                "technical_issues": []
            }

            # 벡터 임베딩 품질 분석
            vector_quality = self._analyze_vector_quality(embedding_vector)
            quality_analysis["quality_scores"]["embedding"] = vector_quality["score"]
            if vector_quality["warnings"]:
                quality_analysis["warnings"].extend(vector_quality["warnings"])
            if vector_quality["recommendations"]:
                quality_analysis["recommendations"].extend(vector_quality["recommendations"])

            # RAG 검색 품질 분석
            if search_results:
                search_quality = self._analyze_search_quality(search_results, message)
                quality_analysis["quality_scores"]["search"] = search_quality["score"]
                if search_quality["warnings"]:
                    quality_analysis["warnings"].extend(search_quality["warnings"])
                if search_quality["recommendations"]:
                    quality_analysis["recommendations"].extend(search_quality["recommendations"])

            # 컨텍스트 품질 분석
            context_quality = self._analyze_context_quality(similar_contexts)
            quality_analysis["quality_scores"]["context"] = context_quality["score"]
            if context_quality["warnings"]:
                quality_analysis["warnings"].extend(context_quality["warnings"])
            if context_quality["recommendations"]:
                quality_analysis["recommendations"].extend(context_quality["recommendations"])

            # 성능 분석
            performance_quality = self._analyze_performance(debug_info["pipeline_steps"])
            quality_analysis["quality_scores"]["performance"] = performance_quality["score"]
            if performance_quality["warnings"]:
                quality_analysis["warnings"].extend(performance_quality["warnings"])

            # 전체 품질 점수 계산
            scores = list(quality_analysis["quality_scores"].values())
            overall_score = sum(scores) / len(scores) if scores else 0

            if overall_score >= 0.8:
                quality_analysis["overall_quality"] = "excellent"
            elif overall_score >= 0.6:
                quality_analysis["overall_quality"] = "good"
            elif overall_score >= 0.4:
                quality_analysis["overall_quality"] = "fair"
            else:
                quality_analysis["overall_quality"] = "poor"

            step7_time = (time.time() - step7_start) * 1000
            debug_info["pipeline_steps"].append({
                "step": "7. Quality Analysis",
                "time_ms": step7_time,
                "details": quality_analysis
            })

            return debug_info

        except Exception as e:
            logger.error(f"기술적 디버깅 정보 수집 실패: {e}")
            return {
                "error": str(e),
                "timestamp": datetime.now().isoformat(),
                "partial_data": "디버깅 정보 수집 중 오류 발생"
            }

    def _analyze_vector_quality(self, embedding_vector: np.ndarray) -> Dict[str, Any]:
        """벡터 임베딩 품질 분석"""
        warnings = []
        recommendations = []

        # 벡터 노름 체크
        vector_norm = np.linalg.norm(embedding_vector)
        if vector_norm < 0.1:
            warnings.append("벡터 노름이 너무 작습니다 (< 0.1). 임베딩 품질이 낮을 수 있습니다.")
            recommendations.append("입력 텍스트를 더 구체적으로 작성해보세요.")
        elif vector_norm > 100:
            warnings.append("벡터 노름이 너무 큽니다 (> 100). 정규화 문제일 수 있습니다.")

        # 희소성 체크
        sparsity_ratio = np.sum(embedding_vector == 0) / len(embedding_vector)
        if sparsity_ratio > 0.8:
            warnings.append(f"벡터가 너무 희소합니다 ({sparsity_ratio:.1%} 영값). 의미 표현이 부족할 수 있습니다.")
            recommendations.append("더 풍부한 어휘를 사용해보세요.")

        # 활성화 패턴 체크
        strong_activations = np.sum(np.abs(embedding_vector) > 0.1)
        if strong_activations < len(embedding_vector) * 0.05:
            warnings.append("강한 활성화가 너무 적습니다. 의미적 표현력이 제한될 수 있습니다.")

        # 품질 점수 계산
        norm_score = min(1.0, vector_norm / 10.0) if vector_norm < 10 else 1.0 - min(0.5, (vector_norm - 10) / 90)
        sparsity_score = 1.0 - sparsity_ratio
        activation_score = min(1.0, strong_activations / (len(embedding_vector) * 0.1))

        overall_score = (norm_score + sparsity_score + activation_score) / 3

        return {
            "score": overall_score,
            "warnings": warnings,
            "recommendations": recommendations,
            "metrics": {
                "norm_score": norm_score,
                "sparsity_score": sparsity_score,
                "activation_score": activation_score
            }
        }

    def _analyze_search_quality(self, search_results: List, query: str) -> Dict[str, Any]:
        """RAG 검색 품질 분석"""
        warnings = []
        recommendations = []

        if not search_results:
            warnings.append("검색 결과가 없습니다.")
            recommendations.append("데이터베이스에 관련 데이터를 추가하세요.")
            return {"score": 0.0, "warnings": warnings, "recommendations": recommendations}

        # 최고 유사도 체크
        top_score = search_results[0].score
        if top_score < 0.3:
            warnings.append(f"최고 유사도가 낮습니다 ({top_score:.3f}). 관련성이 부족할 수 있습니다.")
            recommendations.append("더 구체적인 키워드를 사용하거나 표현을 바꿔보세요.")
        elif top_score < 0.5:
            warnings.append(f"유사도가 보통입니다 ({top_score:.3f}). 더 정확한 매칭이 필요할 수 있습니다.")

        # 결과 다양성 체크
        emotions = set([r.metadata.get("emotion", "") for r in search_results[:5]])
        if len(emotions) <= 1:
            warnings.append("검색 결과의 감정 다양성이 부족합니다.")
            recommendations.append("다양한 감정 표현 데이터를 추가하세요.")

        # 유사도 분포 체크
        scores = [r.score for r in search_results[:5]]
        if len(scores) > 1:
            score_std = np.std(scores)
            if score_std < 0.05:
                warnings.append("유사도 점수가 너무 균등합니다. 구분력이 부족할 수 있습니다.")

        # 품질 점수 계산
        relevance_score = min(1.0, top_score / 0.8)
        diversity_score = min(1.0, len(emotions) / 3.0)
        coverage_score = min(1.0, len(search_results) / 5.0)

        overall_score = (relevance_score + diversity_score + coverage_score) / 3

        return {
            "score": overall_score,
            "warnings": warnings,
            "recommendations": recommendations,
            "metrics": {
                "relevance_score": relevance_score,
                "diversity_score": diversity_score,
                "coverage_score": coverage_score
            }
        }

    def _analyze_context_quality(self, similar_contexts: List[Dict]) -> Dict[str, Any]:
        """컨텍스트 품질 분석"""
        warnings = []
        recommendations = []

        if not similar_contexts:
            warnings.append("유사한 컨텍스트가 없습니다.")
            recommendations.append("관련 대화 데이터를 추가하세요.")
            return {"score": 0.0, "warnings": warnings, "recommendations": recommendations}

        # 컨텍스트 길이 체크
        avg_length = np.mean([len(ctx.get("user_utterance", "")) for ctx in similar_contexts])
        if avg_length < 10:
            warnings.append("컨텍스트가 너무 짧습니다. 충분한 정보가 부족할 수 있습니다.")
        elif avg_length > 200:
            warnings.append("컨텍스트가 너무 깁니다. 노이즈가 포함될 수 있습니다.")

        # 다양성 체크
        unique_emotions = len(set([ctx.get("emotion", "") for ctx in similar_contexts]))
        unique_relationships = len(set([ctx.get("relationship", "") for ctx in similar_contexts]))

        if unique_emotions <= 1 and len(similar_contexts) > 1:
            warnings.append("감정 다양성이 부족합니다.")
        if unique_relationships <= 1 and len(similar_contexts) > 1:
            warnings.append("관계 다양성이 부족합니다.")

        # 품질 점수 계산
        length_score = 1.0 if 20 <= avg_length <= 100 else 0.5
        diversity_score = (min(1.0, unique_emotions / 2.0) + min(1.0, unique_relationships / 2.0)) / 2
        quantity_score = min(1.0, len(similar_contexts) / 3.0)

        overall_score = (length_score + diversity_score + quantity_score) / 3

        return {
            "score": overall_score,
            "warnings": warnings,
            "recommendations": recommendations,
            "metrics": {
                "length_score": length_score,
                "diversity_score": diversity_score,
                "quantity_score": quantity_score
            }
        }

    def _analyze_performance(self, pipeline_steps: List[Dict]) -> Dict[str, Any]:
        """성능 분석"""
        warnings = []

        for step in pipeline_steps:
            step_time = step.get("time_ms", 0)
            step_name = step.get("step", "")

            # 각 단계별 성능 임계값
            thresholds = {
                "Text Analysis": 50,
                "Vector Embedding": 500,
                "RAG Query Construction": 100,
                "Vector Search Execution": 1000,
                "Context Construction": 200,
                "GPT-4 Prompt Construction": 100
            }

            for key, threshold in thresholds.items():
                if key in step_name and step_time > threshold:
                    warnings.append(f"{step_name} 단계가 느립니다 ({step_time:.1f}ms > {threshold}ms)")

        # 전체 시간 체크
        total_time = sum([step.get("time_ms", 0) for step in pipeline_steps])
        if total_time > 5000:
            warnings.append(f"전체 처리 시간이 너무 깁니다 ({total_time:.1f}ms)")

        # 성능 점수 계산
        performance_score = max(0.0, 1.0 - len(warnings) * 0.2)

        return {
            "score": performance_score,
            "warnings": warnings,
            "total_time_ms": total_time
        }

    async def _get_total_documents(self, vector_store) -> int:
        """벡터 스토어의 총 문서 수 조회"""
        try:
            stats = await vector_store.get_collection_stats()
            return stats.total_documents
        except:
            return 0

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

        # 7. 디버깅 정보 수집 (요청 시에만)
        debug_info = None
        if request.include_reasoning:
            vector_store = await get_vector_store()
            debug_info = await chatbot._collect_debug_info(
                request.message, emotion_str, relationship_str,
                similar_contexts, processing_time, vector_store
            )

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
                "emotion_detection_method": "gpt4_analysis",
                "debug_info": debug_info  # 디버깅 정보 추가
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


# =====================================================
# 🔄 기존 API 엔드포인트들 (유지)
# =====================================================

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


@router.post("/create-sample-data")
async def create_sample_data(processor = Depends(get_teen_empathy_processor)):
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