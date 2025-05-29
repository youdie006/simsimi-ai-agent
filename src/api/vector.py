"""
Vector Store API 라우터
ChromaDB 벡터 스토어 관련 엔드포인트들
"""

from fastapi import APIRouter, HTTPException, Depends, status
from typing import List
import time
from loguru import logger

from ..core.vector_store import get_vector_store
from ..models.vector_models import (
    VectorSearchRequest, VectorSearchResponse,
    DocumentAddRequest, DocumentAddResponse,
    VectorStoreStats, SearchResult
)


router = APIRouter()


@router.post("/search", response_model=VectorSearchResponse)
async def search_vectors(
    request: VectorSearchRequest,
    vector_store = Depends(get_vector_store)
):
    """
    🔍 벡터 유사도 검색

    - 쿼리와 유사한 문서들을 벡터 검색으로 찾기
    - 감정, 관계 등 메타데이터 필터링 지원
    - top_k 개수만큼 결과 반환
    """
    try:
        logger.info(f"벡터 검색 요청: '{request.query[:50]}...', top_k: {request.top_k}")
        start_time = time.time()

        # 벡터 검색 실행
        results = await vector_store.search(
            query=request.query,
            top_k=request.top_k,
            filter_metadata=request.filter_metadata
        )

        search_time_ms = (time.time() - start_time) * 1000

        return VectorSearchResponse(
            results=results,
            query=request.query,
            total_results=len(results),
            search_time_ms=search_time_ms
        )

    except Exception as e:
        logger.error(f"벡터 검색 실패: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"벡터 검색 중 오류가 발생했습니다: {str(e)}"
        )


@router.post("/documents", response_model=DocumentAddResponse)
async def add_documents(
    request: DocumentAddRequest,
    vector_store = Depends(get_vector_store)
):
    """
    📝 문서 추가

    - 새 문서들을 벡터 DB에 추가
    - 자동으로 임베딩 생성 및 인덱싱
    - 배치 처리로 효율적 추가
    """
    try:
        logger.info(f"문서 추가 요청: {len(request.documents)}개")
        start_time = time.time()

        # 문서 추가 실행
        document_ids = await vector_store.add_documents(request.documents)

        processing_time_ms = (time.time() - start_time) * 1000

        return DocumentAddResponse(
            success=True,
            added_count=len(document_ids),
            document_ids=document_ids,
            processing_time_ms=processing_time_ms,
            errors=[]
        )

    except Exception as e:
        logger.error(f"문서 추가 실패: {e}")
        return DocumentAddResponse(
            success=False,
            added_count=0,
            document_ids=[],
            processing_time_ms=0,
            errors=[str(e)]
        )


@router.get("/stats", response_model=VectorStoreStats)
async def get_vector_stats(vector_store = Depends(get_vector_store)):
    """
    📊 벡터 스토어 통계

    - 총 문서 수, 컬렉션 정보
    - 임베딩 모델 정보
    - 시스템 상태 확인
    """
    try:
        stats = await vector_store.get_collection_stats()
        return stats

    except Exception as e:
        logger.error(f"통계 조회 실패: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"통계 조회 중 오류가 발생했습니다: {str(e)}"
        )


@router.delete("/documents/{document_id}")
async def delete_document(
    document_id: str,
    vector_store = Depends(get_vector_store)
):
    """
    🗑️ 문서 삭제

    - 특정 문서를 벡터 DB에서 삭제
    """
    try:
        success = await vector_store.delete_documents([document_id])

        if success:
            return {"message": f"문서 {document_id} 삭제 완료", "success": True}
        else:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail=f"문서 {document_id}를 찾을 수 없습니다"
            )

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"문서 삭제 실패: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"문서 삭제 중 오류가 발생했습니다: {str(e)}"
        )


@router.post("/clear")
async def clear_collection(vector_store = Depends(get_vector_store)):
    """
    ⚠️ 컬렉션 초기화

    - 모든 문서 삭제 및 컬렉션 초기화
    - 주의: 모든 데이터가 삭제됩니다!
    """
    try:
        success = await vector_store.clear_collection()

        if success:
            return {
                "message": "컬렉션이 성공적으로 초기화되었습니다",
                "success": True,
                "warning": "모든 데이터가 삭제되었습니다"
            }
        else:
            raise HTTPException(
                status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                detail="컬렉션 초기화에 실패했습니다"
            )

    except Exception as e:
        logger.error(f"컬렉션 초기화 실패: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"컬렉션 초기화 중 오류가 발생했습니다: {str(e)}"
        )


@router.get("/health")
async def vector_health_check(vector_store = Depends(get_vector_store)):
    """
    💊 벡터 스토어 헬스 체크

    - 벡터 DB 연결 상태 확인
    - 임베딩 모델 상태 확인
    """
    try:
        stats = await vector_store.get_collection_stats()

        health_status = {
            "status": "healthy" if stats.status == "healthy" else "unhealthy",
            "collection_name": stats.collection_name,
            "total_documents": stats.total_documents,
            "embedding_model": stats.embedding_model,
            "database_path": stats.database_path,
            "checks": {
                "chromadb_connection": True,
                "embedding_model_loaded": stats.embedding_dimension is not None,
                "collection_accessible": stats.total_documents >= 0
            },
            "last_updated": stats.last_updated
        }

        return health_status

    except Exception as e:
        logger.error(f"헬스 체크 실패: {e}")
        return {
            "status": "unhealthy",
            "error": str(e),
            "checks": {
                "chromadb_connection": False,
                "embedding_model_loaded": False,
                "collection_accessible": False
            }
        }


@router.get("/search-demo")
async def search_demo():
    """
    🎯 검색 데모 쿼리 예시

    - 테스트용 검색 쿼리들
    - API 사용법 가이드
    """
    return {
        "demo_queries": [
            {
                "description": "기본 검색",
                "query": "친구와 싸웠어요",
                "example_request": {
                    "query": "친구와 싸웠어요",
                    "top_k": 5
                }
            },
            {
                "description": "감정 필터 검색",
                "query": "학교에서 스트레스 받아",
                "example_request": {
                    "query": "학교에서 스트레스 받아",
                    "top_k": 3,
                    "filter_metadata": {
                        "emotion": "분노"
                    }
                }
            },
            {
                "description": "관계 맥락 검색",
                "query": "잔소리 때문에 힘들어",
                "example_request": {
                    "query": "잔소리 때문에 힘들어",
                    "top_k": 5,
                    "filter_metadata": {
                        "relationship": "부모님",
                        "data_source": "aihub"
                    }
                }
            }
        ],
        "usage_tips": [
            "구체적인 상황을 포함한 쿼리가 더 좋은 결과를 제공합니다",
            "감정과 관계 맥락을 필터로 활용하면 정확도가 높아집니다",
            "top_k는 1-20 사이의 값을 권장합니다"
        ]
    }