"""
Vector Store 관련 데이터 모델들
ChromaDB와 연동하는 Pydantic 모델들
"""

from pydantic import BaseModel, Field
from typing import Dict, List, Any, Optional
from datetime import datetime


class DocumentInput(BaseModel):
    """벡터 DB에 저장할 문서 입력 모델"""
    content: str = Field(..., description="문서 내용", min_length=1)
    metadata: Optional[Dict[str, Any]] = Field(default={}, description="문서 메타데이터")
    document_id: Optional[str] = Field(default=None, description="문서 고유 ID")

    class Config:
        json_schema_extra = {
            "example": {
                "content": "[불안] [친구] 친구가 나를 무시하는 것 같아서 속상해",
                "metadata": {
                    "emotion": "불안",
                    "relationship": "친구",
                    "empathy_label": "위로",
                    "data_source": "aihub"
                },
                "document_id": "session_001"
            }
        }


class SearchResult(BaseModel):
    """벡터 검색 결과 모델"""
    content: str = Field(..., description="검색된 문서 내용")
    metadata: Dict[str, Any] = Field(default={}, description="문서 메타데이터")
    score: float = Field(..., description="유사도 점수 (0~1)", ge=0, le=1)
    document_id: str = Field(..., description="문서 고유 ID")

    class Config:
        json_schema_extra = {
            "example": {
                "content": "[불안] [친구] 친구가 나를 무시하는 것 같아서 속상해",
                "metadata": {
                    "user_utterance": "친구가 나를 무시하는 것 같아서 속상해",
                    "system_response": "친구가 너를 무시한다고 느끼는 구체적인 상황이 있었나?",
                    "emotion": "불안",
                    "relationship": "친구",
                    "empathy_label": "위로"
                },
                "score": 0.95,
                "document_id": "session_001"
            }
        }


class VectorSearchRequest(BaseModel):
    """벡터 검색 요청 모델"""
    query: str = Field(..., description="검색 쿼리", min_length=1, max_length=500)
    top_k: int = Field(default=5, description="반환할 결과 수", ge=1, le=20)
    filter_metadata: Optional[Dict[str, Any]] = Field(default=None, description="메타데이터 필터")
    include_scores: bool = Field(default=True, description="유사도 점수 포함 여부")

    class Config:
        json_schema_extra = {
            "example": {
                "query": "친구와 싸웠어요",
                "top_k": 3,
                "filter_metadata": {
                    "emotion": "분노",
                    "data_source": "aihub"
                },
                "include_scores": True
            }
        }


class VectorSearchResponse(BaseModel):
    """벡터 검색 응답 모델"""
    results: List[SearchResult] = Field(..., description="검색 결과 목록")
    query: str = Field(..., description="검색 쿼리")
    total_results: int = Field(..., description="총 결과 수")
    search_time_ms: float = Field(..., description="검색 소요 시간 (밀리초)")

    class Config:
        json_schema_extra = {
            "example": {
                "results": [
                    {
                        "content": "[분노] [친구] 친구와 싸워서 화가 나",
                        "metadata": {
                            "emotion": "분노",
                            "relationship": "친구",
                            "empathy_label": "위로"
                        },
                        "score": 0.92,
                        "document_id": "session_123"
                    }
                ],
                "query": "친구와 싸웠어요",
                "total_results": 1,
                "search_time_ms": 45.2
            }
        }


class DocumentAddRequest(BaseModel):
    """문서 추가 요청 모델"""
    documents: List[DocumentInput] = Field(..., description="추가할 문서들", min_items=1)
    batch_size: int = Field(default=100, description="배치 크기", ge=1, le=1000)

    class Config:
        json_schema_extra = {
            "example": {
                "documents": [
                    {
                        "content": "[기쁨] [친구] 친구와 함께 시험을 잘 봤어요",
                        "metadata": {
                            "emotion": "기쁨",
                            "relationship": "친구",
                            "empathy_label": "격려"
                        }
                    }
                ],
                "batch_size": 50
            }
        }


class DocumentAddResponse(BaseModel):
    """문서 추가 응답 모델"""
    success: bool = Field(..., description="추가 성공 여부")
    added_count: int = Field(..., description="추가된 문서 수")
    document_ids: List[str] = Field(..., description="추가된 문서 ID 목록")
    processing_time_ms: float = Field(..., description="처리 소요 시간 (밀리초)")
    errors: List[str] = Field(default=[], description="오류 메시지 목록")

    class Config:
        json_schema_extra = {
            "example": {
                "success": True,
                "added_count": 5,
                "document_ids": ["doc_001", "doc_002", "doc_003"],
                "processing_time_ms": 1250.5,
                "errors": []
            }
        }


class VectorStoreStats(BaseModel):
    """벡터 스토어 통계 모델"""
    collection_name: str = Field(..., description="컬렉션 이름")
    total_documents: int = Field(..., description="총 문서 수")
    embedding_model: str = Field(..., description="사용중인 임베딩 모델")
    embedding_dimension: Optional[int] = Field(default=None, description="임베딩 차원")
    database_path: str = Field(..., description="데이터베이스 경로")
    status: str = Field(..., description="상태")
    last_updated: str = Field(default_factory=lambda: datetime.now().isoformat(), description="마지막 업데이트")

    class Config:
        json_schema_extra = {
            "example": {
                "collection_name": "teen_empathy_chat",
                "total_documents": 31821,
                "embedding_model": "jhgan/ko-sbert-multitask",
                "embedding_dimension": 768,
                "database_path": "./data/chromadb",
                "status": "healthy",
                "last_updated": "2024-01-01T12:00:00"
            }
        }