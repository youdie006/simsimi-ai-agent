"""
AI Hub 공감형 대화 데이터 처리기 - 간단 버전
"""

import json
import os
import asyncio
from typing import Dict, List, Optional
from datetime import datetime
from loguru import logger


class FixedTeenEmpathyDataProcessor:
    """간단한 청소년 공감 데이터 처리기"""

    def __init__(self, data_dir: str = "./data"):
        self.data_dir = data_dir

    async def search_similar_contexts(self, query: str, emotion: str = None,
                                    relationship: str = None, top_k: int = 5) -> List[Dict]:
        """유사한 대화 맥락 검색"""
        try:
            from ..core.vector_store import get_vector_store
            vector_store = await get_vector_store()
            
            # 검색 쿼리 구성
            search_query = query
            if emotion:
                search_query = f"[{emotion}] {search_query}"
            if relationship:
                search_query = f"[{relationship}] {search_query}"
            
            logger.info(f"🔍 검색 쿼리: '{search_query}'")
            
            # 벡터 검색 실행
            results = await vector_store.search(
                query=search_query,
                top_k=top_k * 2,
                filter_metadata={"data_source": "aihub_unified"}
            )
            
            # 결과 포맷팅 (매우 낮은 임계값)
            formatted_results = []
            for result in results:
                if result.score >= 0.01:  # 임계값 0.01
                    formatted_results.append({
                        "user_utterance": result.metadata.get("user_utterance", ""),
                        "system_response": result.metadata.get("system_response", ""),
                        "emotion": result.metadata.get("emotion", ""),
                        "relationship": result.metadata.get("relationship", ""),
                        "empathy_label": result.metadata.get("empathy_label", ""),
                        "similarity_score": result.score,
                        "teen_context": {}
                    })
            
            # 유사도 순 정렬
            formatted_results.sort(key=lambda x: x["similarity_score"], reverse=True)
            logger.info(f"✅ 검색 완료: {len(formatted_results)}개 결과")
            
            return formatted_results[:top_k]
            
        except Exception as e:
            logger.error(f"❌ 검색 실패: {e}")
            return []

    def get_empathy_strategy(self, emotion: str) -> List[str]:
        """감정에 따른 공감 전략 추천"""
        strategies = {
            "기쁨": ["격려", "동조"],
            "당황": ["위로", "조언"], 
            "분노": ["위로", "동조"],
            "불안": ["위로", "격려"],
            "상처": ["위로", "격려"],
            "슬픔": ["위로", "동조"]
        }
        return strategies.get(emotion, ["위로", "격려"])


# 전역 인스턴스
_processor_instance = None


async def get_teen_empathy_processor() -> FixedTeenEmpathyDataProcessor:
    """청소년 공감 데이터 처리기 싱글톤 인스턴스"""
    global _processor_instance
    
    if _processor_instance is None:
        data_dir = os.getenv("AIHUB_DATA_DIR", "./data/aihub")
        _processor_instance = FixedTeenEmpathyDataProcessor(data_dir)
    
    return _processor_instance
