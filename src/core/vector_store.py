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
import subprocess
import glob

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
        """ChromaDB 및 임베딩 모델 초기화 - ChromaDB 파일 직접 사용"""
        try:
            logger.info("ChromaDB Vector Store 초기화 시작...")

            # 🔍 환경 감지
            is_huggingface = bool(os.getenv("SPACE_ID") or os.getenv("SPACE_AUTHOR_NAME"))
            is_local_dev = bool(os.getenv("LOCAL_DEV") or os.getenv("DEVELOPMENT_MODE"))

            logger.info(f"🌍 환경 감지 - HuggingFace: {is_huggingface}, Local: {is_local_dev}")

            # 🤗 허깅페이스 환경에서 ChromaDB 다운로드
            if is_huggingface:
                logger.info("🤗 허깅페이스 환경 - ChromaDB 데이터셋 다운로드 시도")
                await self._download_chromadb_dataset()

            # ChromaDB 경로 설정
            db_path = os.getenv("CHROMADB_PATH", "/app/data/chromadb")
            os.makedirs(db_path, exist_ok=True)

            # 📂 ChromaDB 파일 존재 확인
            chroma_files = self._check_chromadb_files(db_path)
            logger.info(f"📂 ChromaDB 파일 상태: {chroma_files}")

            # ChromaDB 0.3.21 호환 Settings
            settings = Settings(
                chroma_db_impl="duckdb+parquet",
                persist_directory=db_path,
                anonymized_telemetry=False
            )

            self.client = chromadb.Client(settings)

            # 임베딩 모델 로드
            logger.info(f"한국어 임베딩 모델 로드 중: {self.model_name}")
            self.embedding_model = SentenceTransformer(
                self.model_name,
                cache_folder=self.cache_dir,
                device='cpu'
            )
            logger.info(f"임베딩 모델 로드 완료 - 차원: {self.embedding_model.get_sentence_embedding_dimension()}")

            # 🔍 기존 컬렉션 확인
            try:
                existing_collections = self.client.list_collections()
                collection_names = [c.name for c in existing_collections]
                logger.info(f"기존 컬렉션들: {collection_names}")

                # 타겟 컬렉션 찾기
                target_collection = None
                possible_names = [self.collection_name, "teen_empathy_chat", "empathy_chat", "chat_data"]

                for name in possible_names:
                    if name in collection_names:
                        target_collection = name
                        break

                if target_collection:
                    self.collection = self.client.get_collection(name=target_collection)
                    existing_count = self.collection.count()
                    logger.info(f"✅ 기존 컬렉션 연결: {target_collection} ({existing_count}개 문서)")

                    if existing_count > 0:
                        logger.info("🎉 기존 임베딩 데이터 사용 - 초기화 완료!")
                        return
                    else:
                        logger.warning("⚠️ 컬렉션은 있지만 문서가 없음")

            except Exception as e:
                logger.warning(f"⚠️ 기존 컬렉션 확인 실패: {e}")

            # 🔧 컬렉션이 없거나 빈 경우 - 새로 생성
            logger.info("🔧 새 컬렉션 생성 및 기본 데이터 추가")
            try:
                self.collection = self.client.create_collection(
                    name=self.collection_name,
                    metadata={
                        "description": "Teen empathy conversation embeddings",
                        "created_at": datetime.now().isoformat(),
                        "environment": "huggingface" if is_huggingface else "local"
                    }
                )
                logger.info(f"🆕 새 컬렉션 생성: {self.collection_name}")

                # 기본 데이터 추가
                await self._add_essential_data()

            except Exception as e:
                logger.error(f"❌ 컬렉션 생성 실패: {e}")
                raise

            logger.info("✅ ChromaDB Vector Store 초기화 완료")

        except Exception as e:
            logger.error(f"❌ ChromaDB 초기화 실패: {e}")
            # 간단한 방식으로 재시도
            await self._fallback_initialize()

    async def _download_chromadb_dataset(self):
        """허깅페이스에서 ChromaDB 데이터셋 다운로드"""
        try:
            logger.info("📥 ChromaDB 데이터셋 다운로드 시작...")

            # 다운로드 디렉토리 준비
            download_dir = "/app/data"
            os.makedirs(download_dir, exist_ok=True)

            # huggingface-cli를 사용해서 ChromaDB 파일들 다운로드
            cmd = [
                "huggingface-cli", "download",
                "youdie006/simsimi-ai-agent-data",
                "--repo-type", "dataset",
                "--local-dir", download_dir,
                "--local-dir-use-symlinks", "False"
            ]

            logger.info(f"📥 실행 명령: {' '.join(cmd)}")

            result = subprocess.run(
                cmd,
                capture_output=True,
                text=True,
                timeout=600,  # 10분 타임아웃
                cwd="/app"
            )

            if result.returncode == 0:
                logger.info("✅ ChromaDB 데이터셋 다운로드 완료")
                logger.info(f"📁 다운로드 내용: {result.stdout}")

                # 다운로드된 파일 확인
                downloaded_files = []
                for root, dirs, files in os.walk(download_dir):
                    for file in files:
                        file_path = os.path.join(root, file)
                        downloaded_files.append(file_path)

                logger.info(f"📂 다운로드된 파일들: {downloaded_files}")

                # ChromaDB 관련 파일 확인
                chromadb_files = [f for f in downloaded_files if any(ext in f.lower() for ext in ['.duckdb', '.parquet', 'chroma'])]
                logger.info(f"🗄️ ChromaDB 관련 파일들: {chromadb_files}")

            else:
                logger.warning(f"⚠️ 다운로드 실패: {result.stderr}")
                logger.warning(f"📤 출력: {result.stdout}")

        except subprocess.TimeoutExpired:
            logger.warning("⚠️ 다운로드 타임아웃 (10분 초과)")
        except Exception as e:
            logger.warning(f"⚠️ 다운로드 실패: {e}")

    def _check_chromadb_files(self, db_path: str) -> dict:
        """ChromaDB 파일 존재 확인"""
        try:
            files_info = {}

            # 일반적인 ChromaDB 파일 패턴
            patterns = [
                "*.duckdb",
                "*.parquet",
                "chroma.sqlite3",
                "index/*",
                "*.bin"
            ]

            for pattern in patterns:
                files = glob.glob(os.path.join(db_path, "**", pattern), recursive=True)
                if files:
                    files_info[pattern] = files

            # 전체 파일 목록
            all_files = []
            if os.path.exists(db_path):
                for root, dirs, files in os.walk(db_path):
                    for file in files:
                        all_files.append(os.path.join(root, file))

            files_info["all_files"] = all_files
            files_info["total_count"] = len(all_files)

            return files_info

        except Exception as e:
            logger.warning(f"파일 확인 실패: {e}")
            return {"error": str(e)}

    async def _add_essential_data(self):
        """필수 데이터 추가 (ChromaDB 파일이 없는 경우)"""
        try:
            logger.info("🔧 필수 데이터 추가 중...")

            essential_docs = [
                DocumentInput(
                    content="엄마가 계속 잔소리해서 화가 나요",
                    metadata={
                        "user_utterance": "엄마가 계속 잔소리해서 화가 나요",
                        "system_response": "부모님과의 갈등은 정말 힘들지. 엄마도 너를 걱정해서 그러는 건 알지만, 잔소리가 계속되면 스트레스받을 만해.",
                        "emotion": "분노",
                        "relationship": "부모님",
                        "data_source": "essential"
                    },
                    document_id="essential_parent_conflict"
                ),
                DocumentInput(
                    content="친구가 나를 무시하는 것 같아서 속상해",
                    metadata={
                        "user_utterance": "친구가 나를 무시하는 것 같아서 속상해",
                        "system_response": "친구가 너를 무시한다고 느끼는 구체적인 상황이 있었나? 그런 기분이 들면 정말 속상할 것 같아.",
                        "emotion": "상처",
                        "relationship": "친구",
                        "data_source": "essential"
                    },
                    document_id="essential_friend_hurt"
                ),
                DocumentInput(
                    content="시험이 걱정돼서 잠이 안 와요",
                    metadata={
                        "user_utterance": "시험이 걱정돼서 잠이 안 와요",
                        "system_response": "시험 스트레스는 정말 힘들어. 불안한 마음이 드는 건 당연해. 깊게 숨을 쉬고 차근차근 준비해보자.",
                        "emotion": "불안",
                        "relationship": "기타",
                        "data_source": "essential"
                    },
                    document_id="essential_exam_anxiety"
                ),
                DocumentInput(
                    content="요즘 우울해서 아무것도 하기 싫어",
                    metadata={
                        "user_utterance": "요즘 우울해서 아무것도 하기 싫어",
                        "system_response": "우울한 기분이 드는 건 정말 힘들어. 혼자 견디지 말고 주변 사람들과 이야기해보는 것도 좋을 것 같아.",
                        "emotion": "슬픔",
                        "relationship": "기타",
                        "data_source": "essential"
                    },
                    document_id="essential_depression"
                ),
                DocumentInput(
                    content="선생님이 나만 미워하는 것 같아",
                    metadata={
                        "user_utterance": "선생님이 나만 미워하는 것 같아",
                        "system_response": "선생님이 너를 미워한다고 느끼는 구체적인 상황이 있었나? 오해일 수도 있으니까 차근차근 생각해보자.",
                        "emotion": "당황",
                        "relationship": "선생님",
                        "data_source": "essential"
                    },
                    document_id="essential_teacher_conflict"
                )
            ]

            await self.add_documents(essential_docs)
            logger.info(f"✅ 필수 데이터 {len(essential_docs)}개 추가 완료")

        except Exception as e:
            logger.error(f"❌ 필수 데이터 추가 실패: {e}")

    async def _fallback_initialize(self):
        """폴백 초기화"""
        try:
            logger.info("🔄 폴백 초기화 시작...")

            self.client = chromadb.Client()

            if not self.embedding_model:
                self.embedding_model = SentenceTransformer(
                    self.model_name,
                    cache_folder=self.cache_dir,
                    device='cpu'
                )

            try:
                self.collection = self.client.get_collection(name=self.collection_name)
                count = self.collection.count()
                logger.info(f"기존 컬렉션 연결: {self.collection_name} ({count}개)")

                if count == 0:
                    await self._add_essential_data()

            except Exception:
                self.collection = self.client.create_collection(name=self.collection_name)
                logger.info(f"새 컬렉션 생성: {self.collection_name}")
                await self._add_essential_data()

            logger.info("✅ 폴백 초기화 완료")

        except Exception as e:
            logger.error(f"❌ 폴백 초기화도 실패: {e}")
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
        for i, metadata in enumerate(metadatas):
            metadata.update({
                "timestamp": datetime.now().isoformat(),
                "content_length": len(texts[i]),
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