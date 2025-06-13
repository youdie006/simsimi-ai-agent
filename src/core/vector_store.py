"""
ChromaDB 기반 Vector Store - 0.3.21 최종 호환 버전
"""

import chromadb
from chromadb.config import Settings
from sentence_transformers import SentenceTransformer
from typing import List, Dict, Any, Optional
from loguru import logger
import os
import uuid
import time
import math
from datetime import datetime

from ..models.vector_models import SearchResult, DocumentInput, VectorStoreStats


class ChromaVectorStore:
    """ChromaDB 기반 Vector Store - 0.3.21 최종 호환"""

    def __init__(self, collection_name: str = "teen_empathy_chat"):
        self.collection_name = collection_name
        self.client = None
        self.collection = None
        self.embedding_model = None
        self.model_name = "jhgan/ko-sbert-multitask"
        self.cache_dir = "/app/cache"

    async def initialize(self):
        """ChromaDB 및 임베딩 모델 초기화 - 0.3.21 최종 호환"""
        try:
            logger.info("ChromaDB Vector Store 초기화 시작...")

            db_path = os.getenv("CHROMADB_PATH", "/app/data/chromadb")
            os.makedirs(db_path, exist_ok=True)

            # 🔧 ChromaDB 0.3.21 호환 Settings (allow_reset 제거!)
            settings = Settings(
                chroma_db_impl="duckdb+parquet",
                persist_directory=db_path,
                anonymized_telemetry=False
            )

            # 0.3.21에서는 Client() 사용
            self.client = chromadb.Client(settings)

            # 임베딩 모델 로드
            logger.info(f"한국어 임베딩 모델 로드 중: {self.model_name}")
            self.embedding_model = SentenceTransformer(
                self.model_name,
                cache_folder=self.cache_dir,
                device='cpu'
            )
            logger.info(f"임베딩 모델 로드 완료 - 차원: {self.embedding_model.get_sentence_embedding_dimension()}")

            # 컬렉션 생성/연결 (0.3.21 방식)
            try:
                self.collection = self.client.get_collection(name=self.collection_name)
                logger.info(f"기존 컬렉션 연결: {self.collection_name}")
            except Exception:
                # 컬렉션이 없으면 생성
                self.collection = self.client.create_collection(
                    name=self.collection_name,
                    metadata={
                        "description": "Teen empathy conversation embeddings",
                        "created_at": datetime.now().isoformat()
                    }
                )
                logger.info(f"새 컬렉션 생성: {self.collection_name}")

            logger.info("✅ ChromaDB Vector Store 초기화 완료")

        except Exception as e:
            logger.error(f"❌ ChromaDB 초기화 실패: {e}")
            # 더 간단한 방식으로 재시도
            try:
                logger.info("🔄 간단한 방식으로 재시도...")
                self.client = chromadb.Client()

                # 임베딩 모델은 이미 시도했으므로 스킵하지 않음
                if not self.embedding_model:
                    logger.info(f"한국어 임베딩 모델 로드 중: {self.model_name}")
                    self.embedding_model = SentenceTransformer(
                        self.model_name,
                        cache_folder=self.cache_dir,
                        device='cpu'
                    )

                # 컬렉션 생성/연결
                try:
                    self.collection = self.client.get_collection(name=self.collection_name)
                    logger.info(f"기존 컬렉션 연결: {self.collection_name}")
                except Exception:
                    self.collection = self.client.create_collection(name=self.collection_name)
                    logger.info(f"새 컬렉션 생성: {self.collection_name}")

                logger.info("✅ ChromaDB Vector Store 초기화 완료 (간단한 방식)")

            except Exception as e2:
                logger.error(f"❌ 간단한 방식도 실패: {e2}")
                raise

    def create_embeddings(self, texts: List[str]) -> List[List[float]]:
        """임베딩 생성"""
        if not self.embedding_model:
            raise ValueError("임베딩 모델이 초기화되지 않았습니다")

        logger.info(f"임베딩 생성 중: {len(texts)}개 텍스트")

        try:
            embeddings = self.embedding_model.encode(texts, convert_to_tensor=False)
            embeddings_list = embeddings.tolist()
            logger.info(f"✅ 임베딩 생성 완료: {len(embeddings_list)}개")
            return embeddings_list
        except Exception as e:
            logger.error(f"❌ 임베딩 생성 실패: {e}")
            raise

    async def add_documents(self, documents: List[DocumentInput]) -> List[str]:
        """문서를 Vector DB에 추가"""
        if not self.collection:
            raise ValueError("컬렉션이 초기화되지 않았습니다")

        logger.info(f"문서 추가 시작: {len(documents)}개")

        texts = [doc.content for doc in documents]
        metadatas = [doc.metadata or {} for doc in documents]
        document_ids = [doc.document_id or str(uuid.uuid4()) for doc in documents]

        # 메타데이터에 타임스탬프 추가
        for metadata in metadatas:
            metadata.update({
                "timestamp": datetime.now().isoformat(),
                "content_length": len(texts[metadatas.index(metadata)]),
                "indexed_at": datetime.now().isoformat()
            })

        embeddings = self.create_embeddings(texts)

        # 배치 처리
        batch_size = 100
        for i in range(0, len(documents), batch_size):
            end_idx = min(i + batch_size, len(documents))

            self.collection.add(
                embeddings=embeddings[i:end_idx],
                documents=texts[i:end_idx],
                metadatas=metadatas[i:end_idx],
                ids=document_ids[i:end_idx]
            )

            logger.info(f"배치 {i//batch_size + 1} 추가 완료: {end_idx - i}개 문서")

        # 0.3.21에서는 persist() 명시적 호출
        try:
            if hasattr(self.client, 'persist'):
                self.client.persist()
        except Exception as e:
            logger.warning(f"persist() 호출 실패 (무시): {e}")

        logger.info(f"✅ 문서 {len(documents)}개 추가 완료")
        return document_ids

    def _calculate_similarity_from_distance(self, distance: float, method: str = "improved") -> float:
        """개선된 유사도 계산"""
        if method == "improved":
            return 1.0 / (1.0 + distance)
        elif method == "exponential":
            return math.exp(-distance)
        else:
            return 1.0 / (1.0 + distance)

    async def search(self, query: str, top_k: int = 5,
                    filter_metadata: Optional[Dict[str, Any]] = None,
                    similarity_method: str = "improved") -> List[SearchResult]:
        """🔍 유사도 기반 문서 검색 - ChromaDB 0.3.21 호환"""
        if not self.collection:
            raise ValueError("컬렉션이 초기화되지 않았습니다")

        start_time = time.time()
        logger.info(f"검색 시작 - 쿼리: '{query[:50]}...', top_k: {top_k}")

        # 쿼리 임베딩 생성
        query_embedding = self.create_embeddings([query])[0]

        # ChromaDB 0.3.21 검색 API
        search_kwargs = {
            "query_embeddings": [query_embedding],
            "n_results": top_k,
            "include": ["documents", "metadatas", "distances"]
        }

        # 필터링 시도 (실패해도 계속 진행)
        try:
            if filter_metadata:
                search_kwargs["where"] = filter_metadata
            results = self.collection.query(**search_kwargs)
        except Exception as e:
            logger.warning(f"필터 검색 실패, 일반 검색으로 대체: {e}")
            search_kwargs.pop("where", None)
            results = self.collection.query(**search_kwargs)

        # 결과 처리
        search_results = []
        if results.get("documents") and results["documents"][0]:
            distances = results.get("distances", [[]])[0]
            documents = results["documents"][0]
            metadatas = results.get("metadatas", [[]])[0]
            ids = results.get("ids", [[]])[0]

            # 통계 로깅
            if distances:
                min_dist = min(distances)
                max_dist = max(distances)
                avg_dist = sum(distances) / len(distances)
                logger.info(f"📊 거리 통계 - 최소: {min_dist:.3f}, 최대: {max_dist:.3f}, 평균: {avg_dist:.3f}")

            for i in range(len(documents)):
                distance = distances[i] if i < len(distances) else 1.0
                similarity_score = self._calculate_similarity_from_distance(distance, similarity_method)

                search_results.append(SearchResult(
                    content=documents[i],
                    metadata=metadatas[i] if i < len(metadatas) else {},
                    score=similarity_score,
                    document_id=ids[i] if i < len(ids) else f"result_{i}"
                ))

        search_time = (time.time() - start_time) * 1000
        logger.info(f"✅ 검색 완료: {len(search_results)}개 결과 ({search_time:.2f}ms)")

        # 유사도 순 정렬
        search_results.sort(key=lambda x: x.score, reverse=True)

        # 상위 결과 로깅
        for i, result in enumerate(search_results[:3]):
            logger.info(f"결과 {i+1}: 유사도={result.score:.3f}, 내용='{result.content[:50]}...'")

        return search_results

    async def get_collection_stats(self) -> VectorStoreStats:
        """컬렉션 통계 정보"""
        if not self.collection:
            raise ValueError("컬렉션이 초기화되지 않음")

        try:
            count = self.collection.count()
            return VectorStoreStats(
                collection_name=self.collection_name,
                total_documents=count,
                embedding_model=self.model_name,
                embedding_dimension=self.embedding_model.get_sentence_embedding_dimension() if self.embedding_model else None,
                database_path=os.getenv("CHROMADB_PATH", "/app/data/chromadb"),
                status="healthy" if count >= 0 else "error",
                last_updated=datetime.now().isoformat()
            )
        except Exception as e:
            logger.error(f"통계 조회 실패: {e}")
            raise

    async def delete_documents(self, document_ids: List[str]) -> bool:
        """문서 삭제"""
        if not self.collection:
            raise ValueError("컬렉션이 초기화되지 않았습니다")

        try:
            self.collection.delete(ids=document_ids)

            try:
                if hasattr(self.client, 'persist'):
                    self.client.persist()
            except Exception as e:
                logger.warning(f"persist() 실패 (무시): {e}")

            logger.info(f"{len(document_ids)}개 삭제 완료")
            return True
        except Exception as e:
            logger.error(f"❌ 문서 삭제 실패: {e}")
            return False

    async def clear_collection(self) -> bool:
        """컬렉션 전체 삭제"""
        if not self.collection:
            raise ValueError("컬렉션이 초기화되지 않았습니다")

        try:
            self.client.delete_collection(name=self.collection_name)
            self.collection = self.client.create_collection(
                name=self.collection_name,
                metadata={
                    "description": "Teen empathy conversation embeddings",
                    "created_at": datetime.now().isoformat()
                }
            )
            logger.info(f"✅ 컬렉션 {self.collection_name} 초기화 완료")
            return True
        except Exception as e:
            logger.error(f"❌ 컬렉션 초기화 실패: {e}")
            return False


# 전역 인스턴스
_vector_store_instance = None

async def get_vector_store() -> ChromaVectorStore:
    """Vector Store 싱글톤 인스턴스 반환"""
    global _vector_store_instance
    if _vector_store_instance is None:
        collection_name = os.getenv("COLLECTION_NAME", "teen_empathy_chat")
        _vector_store_instance = ChromaVectorStore(collection_name)
        await _vector_store_instance.initialize()
    return _vector_store_instance

def reset_vector_store():
    """Vector Store 인스턴스 리셋 (테스트용)"""
    global _vector_store_instance
    _vector_store_instance = None