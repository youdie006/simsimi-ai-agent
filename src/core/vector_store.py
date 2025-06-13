"""
ChromaDB 기반 Vector Store - 0.3.21 호환 버전
"""

import chromadb
from chromadb.config import Settings
from sentence_transformers import SentenceTransformer
from typing import List, Dict, Any, Optional
from loguru import logger
import os
import uuid
import time
from datetime import datetime

from ..models.vector_models import SearchResult, DocumentInput, VectorStoreStats


class ChromaVectorStore:
    """ChromaDB 기반 Vector Store - 0.3.21 호환"""

    def __init__(self, collection_name: str = "teen_empathy_chat"):
        self.collection_name = collection_name
        self.client = None
        self.collection = None
        self.embedding_model = None
        self.model_name = "jhgan/ko-sbert-multitask"
        self.cache_dir = "/app/cache"

    async def initialize(self):
        """ChromaDB 및 임베딩 모델 초기화"""
        try:
            logger.info("ChromaDB Vector Store 초기화 시작...")

            db_path = os.getenv("CHROMADB_PATH", "/app/data/chromadb")
            os.makedirs(db_path, exist_ok=True)

            # ChromaDB 0.3.21 설정
            self.client = chromadb.PersistentClient(
                path=db_path,
                settings=Settings(
                    allow_reset=True,
                    anonymized_telemetry=False
                )
            )

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
            except ValueError:
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

        logger.info(f"✅ 문서 {len(documents)}개 추가 완료")
        return document_ids

    async def search(self, query: str, top_k: int = 5,
                    filter_metadata: Optional[Dict[str, Any]] = None) -> List[SearchResult]:
        """🔍 유사도 기반 문서 검색"""
        if not self.collection:
            raise ValueError("컬렉션이 초기화되지 않았습니다")

        start_time = time.time()
        logger.info(f"검색 시작 - 쿼리: '{query[:50]}...', top_k: {top_k}")

        # 쿼리 임베딩 생성
        logger.info("임베딩 생성 중: 1개 텍스트")
        query_embedding = self.create_embeddings([query])[0]
        logger.info("✅ 임베딩 생성 완료: 1개")

        # 검색 수행 (ChromaDB 0.3.21 API)
        search_kwargs = {
            "query_embeddings": [query_embedding],
            "n_results": top_k,
            "include": ["documents", "metadatas", "distances"]
        }

        if filter_metadata:
            search_kwargs["where"] = filter_metadata

        results = self.collection.query(**search_kwargs)

        # 🔧 유사도 계산 (L2 거리를 유사도로 변환)
        search_results = []
        if results["documents"] and results["documents"][0]:
            for i in range(len(results["documents"][0])):
                distance = results["distances"][0][i]

                # L2 거리를 유사도로 변환
                if distance <= 0:
                    similarity_score = 1.0
                elif distance >= 2.0:
                    similarity_score = 0.0
                else:
                    similarity_score = max(0.0, 1.0 - (distance / 2.0))

                search_results.append(SearchResult(
                    content=results["documents"][0][i],
                    metadata=results["metadatas"][0][i] or {},
                    score=similarity_score,
                    document_id=results["ids"][0][i] if results.get("ids") else f"result_{i}"
                ))

        search_time = (time.time() - start_time) * 1000
        logger.info(f"✅ 검색 완료: {len(search_results)}개 결과 ({search_time:.2f}ms)")

        # 🔍 디버깅 정보 출력
        for i, result in enumerate(search_results):
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
            logger.info(f"{len(document_ids)}개 삭제 완료")
            return True
        except Exception as e:
            logger.error(f"❌ 문서 삭제 실패: {e}")
            return False

    async def update_document(self, document_id: str, document: DocumentInput) -> bool:
        """문서 업데이트"""
        if not self.collection:
            raise ValueError("컬렉션이 초기화되지 않았습니다")

        try:
            # 기존 문서 삭제
            await self.delete_documents([document_id])

            # 새 문서 추가
            document.document_id = document_id
            await self.add_documents([document])

            logger.info(f"{document_id} 업데이트 완료")
            return True
        except Exception as e:
            logger.error(f"❌ 문서 업데이트 실패: {e}")
            return False

    async def clear_collection(self) -> bool:
        """컬렉션 전체 삭제"""
        if not self.collection:
            raise ValueError("컬렉션이 초기화되지 않았습니다")

        try:
            # 컬렉션 삭제
            self.client.delete_collection(name=self.collection_name)

            # 새 컬렉션 생성
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