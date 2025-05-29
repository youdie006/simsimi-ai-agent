"""
ChromaDB 기반 무료 Vector Store 구현
sentence-transformers 한국어 임베딩 사용
새로운 모델들과 완벽 호환
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
    """ChromaDB 기반 무료 Vector Store"""

    def __init__(self, collection_name: str = "teen_empathy_chat"):
        self.collection_name = collection_name
        self.client = None
        self.collection = None
        self.embedding_model = None
        self.model_name = "jhgan/ko-sbert-multitask"  # 한국어 특화 모델

    async def initialize(self):
        """ChromaDB 및 임베딩 모델 초기화"""
        try:
            logger.info("ChromaDB Vector Store 초기화 시작...")

            # ChromaDB 클라이언트 생성 (로컬 저장)
            db_path = os.getenv("CHROMADB_PATH", "./data/chromadb")
            os.makedirs(db_path, exist_ok=True)

            self.client = chromadb.PersistentClient(
                path=db_path,
                settings=Settings(allow_reset=True, anonymized_telemetry=False)
            )

            # 한국어 임베딩 모델 로드
            logger.info(f"한국어 임베딩 모델 로드 중: {self.model_name}")
            self.embedding_model = SentenceTransformer(self.model_name)
            logger.info(f"임베딩 모델 로드 완료 - 차원: {self.embedding_model.get_sentence_embedding_dimension()}")

            # 컬렉션 생성/연결
            try:
                self.collection = self.client.get_collection(name=self.collection_name)
                logger.info(f"기존 컬렉션 연결: {self.collection_name}")
            except ValueError:
                # 컬렉션이 없으면 새로 생성
                self.collection = self.client.create_collection(
                    name=self.collection_name,
                    metadata={
                        "description": "Teen empathy conversation embeddings",
                        "embedding_model": self.model_name,
                        "created_at": datetime.now().isoformat()
                    }
                )
                logger.info(f"새 컬렉션 생성: {self.collection_name}")

            logger.info("✅ ChromaDB Vector Store 초기화 완료")

        except Exception as e:
            logger.error(f"❌ ChromaDB 초기화 실패: {e}")
            raise

    def create_embeddings(self, texts: List[str]) -> List[List[float]]:
        """무료 한국어 임베딩 생성"""
        try:
            if not self.embedding_model:
                raise ValueError("임베딩 모델이 초기화되지 않았습니다")

            logger.info(f"임베딩 생성 중: {len(texts)}개 텍스트")
            embeddings = self.embedding_model.encode(texts, convert_to_numpy=True)

            # numpy array를 list로 변환
            embeddings_list = embeddings.tolist()
            logger.info(f"✅ 임베딩 생성 완료: {len(embeddings_list)}개")

            return embeddings_list

        except Exception as e:
            logger.error(f"❌ 임베딩 생성 실패: {e}")
            raise

    async def add_documents(self, documents: List[DocumentInput]) -> List[str]:
        """문서들을 Vector DB에 추가"""
        try:
            if not self.collection:
                raise ValueError("컬렉션이 초기화되지 않았습니다")

            logger.info(f"문서 추가 시작: {len(documents)}개")

            # 텍스트와 메타데이터 분리
            texts = [doc.content for doc in documents]
            metadatas = []
            document_ids = []

            for doc in documents:
                # 고유 ID 생성
                doc_id = doc.document_id or str(uuid.uuid4())
                document_ids.append(doc_id)

                # 메타데이터에 타임스탬프 추가
                metadata = doc.metadata.copy() if doc.metadata else {}
                metadata.update({
                    "timestamp": datetime.now().isoformat(),
                    "content_length": len(doc.content),
                    "indexed_at": datetime.now().isoformat()
                })
                metadatas.append(metadata)

            # 임베딩 생성
            embeddings = self.create_embeddings(texts)

            # ChromaDB에 추가 (배치 처리)
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

        except Exception as e:
            logger.error(f"❌ 문서 추가 실패: {e}")
            raise

    async def search(self, query: str, top_k: int = 5,
                    filter_metadata: Optional[Dict[str, Any]] = None) -> List[SearchResult]:
        """유사도 기반 문서 검색"""
        try:
            if not self.collection:
                raise ValueError("컬렉션이 초기화되지 않았습니다")

            start_time = time.time()
            logger.info(f"검색 시작 - 쿼리: '{query[:50]}...', top_k: {top_k}")

            # 쿼리 임베딩 생성
            query_embedding = self.create_embeddings([query])[0]

            # ChromaDB 검색
            search_kwargs = {
                "query_embeddings": [query_embedding],
                "n_results": top_k,
                "include": ["documents", "metadatas", "distances"]
            }

            # 메타데이터 필터 적용 (선택사항)
            if filter_metadata:
                search_kwargs["where"] = filter_metadata

            results = self.collection.query(**search_kwargs)

            # 결과 포맷팅
            search_results = []

            if results["documents"] and results["documents"][0]:
                for i in range(len(results["documents"][0])):
                    # 거리를 유사도 점수로 변환 (거리가 낮을수록 유사도 높음)
                    distance = results["distances"][0][i]
                    similarity_score = max(0.0, 1.0 - distance)  # 0~1 사이로 정규화

                    search_results.append(SearchResult(
                        content=results["documents"][0][i],
                        metadata=results["metadatas"][0][i] if results["metadatas"][0] else {},
                        score=similarity_score,
                        document_id=results["ids"][0][i] if "ids" in results else f"result_{i}"
                    ))

            search_time = (time.time() - start_time) * 1000  # 밀리초 변환
            logger.info(f"✅ 검색 완료: {len(search_results)}개 결과 ({search_time:.2f}ms)")
            return search_results

        except Exception as e:
            logger.error(f"❌ 검색 실패: {e}")
            raise

    async def get_collection_stats(self) -> VectorStoreStats:
        """컬렉션 통계 정보"""
        try:
            if not self.collection:
                raise ValueError("컬렉션이 초기화되지 않음")

            count = self.collection.count()

            return VectorStoreStats(
                collection_name=self.collection_name,
                total_documents=count,
                embedding_model=self.model_name,
                embedding_dimension=self.embedding_model.get_sentence_embedding_dimension() if self.embedding_model else None,
                database_path=os.getenv("CHROMADB_PATH", "./data/chromadb"),
                status="healthy" if count >= 0 else "error",
                last_updated=datetime.now().isoformat()
            )

        except Exception as e:
            logger.error(f"통계 조회 실패: {e}")
            raise

    async def delete_documents(self, document_ids: List[str]) -> bool:
        """문서 삭제"""
        try:
            if not self.collection:
                raise ValueError("컬렉션이 초기화되지 않았습니다")

            self.collection.delete(ids=document_ids)
            logger.info(f"✅ 문서 {len(document_ids)}개 삭제 완료")
            return True

        except Exception as e:
            logger.error(f"❌ 문서 삭제 실패: {e}")
            return False

    async def update_document(self, document_id: str, document: DocumentInput) -> bool:
        """문서 업데이트"""
        try:
            if not self.collection:
                raise ValueError("컬렉션이 초기화되지 않았습니다")

            # 기존 문서 삭제
            await self.delete_documents([document_id])

            # 새 문서 추가
            document.document_id = document_id
            await self.add_documents([document])

            logger.info(f"✅ 문서 {document_id} 업데이트 완료")
            return True

        except Exception as e:
            logger.error(f"❌ 문서 업데이트 실패: {e}")
            return False

    async def clear_collection(self) -> bool:
        """컬렉션 전체 삭제"""
        try:
            if not self.collection:
                raise ValueError("컬렉션이 초기화되지 않았습니다")

            # 컬렉션 삭제 후 새로 생성
            self.client.delete_collection(self.collection_name)
            self.collection = self.client.create_collection(
                name=self.collection_name,
                metadata={
                    "description": "Teen empathy conversation embeddings",
                    "embedding_model": self.model_name,
                    "created_at": datetime.now().isoformat()
                }
            )

            logger.info(f"✅ 컬렉션 {self.collection_name} 초기화 완료")
            return True

        except Exception as e:
            logger.error(f"❌ 컬렉션 초기화 실패: {e}")
            return False


# 전역 인스턴스 및 팩토리 함수
_vector_store_instance = None


async def get_vector_store() -> ChromaVectorStore:
    """Vector Store 싱글톤 인스턴스 반환"""
    global _vector_store_instance

    if _vector_store_instance is None:
        collection_name = os.getenv("COLLECTION_NAME", "teen_empathy_chat")
        _vector_store_instance = ChromaVectorStore(collection_name)
        await _vector_store_instance.initialize()

    return _vector_store_instance


async def reset_vector_store() -> ChromaVectorStore:
    """Vector Store 인스턴스 리셋 (테스팅용)"""
    global _vector_store_instance
    _vector_store_instance = None
    return await get_vector_store()