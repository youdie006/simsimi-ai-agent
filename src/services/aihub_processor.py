"""
AI Hub 공감형 대화 데이터 처리기
청소년 맥락 변환 + ChromaDB 벡터 스토어 연동
새로운 모델들과 완벽 호환
"""

import json
import os
import re
from pathlib import Path
from typing import Dict, List, Optional, Tuple
from dataclasses import dataclass, asdict
from datetime import datetime
import uuid

from loguru import logger
from ..core.vector_store import get_vector_store
from ..models.vector_models import DocumentInput
from ..models.function_models import EmotionType, RelationshipType, EmpathyStrategy


@dataclass
class TeenDialogSession:
    """청소년 대화 세션 데이터"""
    session_id: str
    user_utterance: str
    system_response: str
    emotion: str
    relationship: str
    empathy_label: str
    teen_context: Dict[str, str]

@dataclass
class ProcessingStats:
    """데이터 처리 통계"""
    total_sessions: int
    teen_converted: int
    emotion_distribution: Dict[str, int]
    relationship_distribution: Dict[str, int]
    empathy_distribution: Dict[str, int]
    processing_time: str


class TeenEmpathyDataProcessor:
    """청소년 공감형 AI Hub 데이터 처리기"""

    def __init__(self, data_dir: str = "./data"):
        self.data_dir = Path(data_dir)
        self.data_dir.mkdir(parents=True, exist_ok=True)

        # 청소년 맥락 변환 매핑
        self.teen_context_mapping = {
            "relationship": {
                "부부": "부모님",
                "직장동료": "친구",
                "지인": "동급생",
                "연인": "좋아하는 사람",
                "부모자녀": "부모님",
                "형제자매": "형제자매",
                "친구": "친구"
            },
            "situation_words": {
                "직장": "학교",
                "회사": "학교",
                "업무": "공부",
                "동료": "친구",
                "상사": "선생님",
                "부장": "담임선생님",
                "팀장": "반장",
                "회의": "수업",
                "프로젝트": "과제",
                "출근": "등교",
                "퇴근": "하교",
                "야근": "야자",
                "휴가": "방학",
                "월급": "용돈",
                "승진": "성적 향상",
                "면접": "입시",
                "거래처": "다른 학교",
                "계약": "시험",
                "발표": "발표",
                "미팅": "수업"
            }
        }

        # 감정별 공감 전략 (Enum과 매핑)
        self.empathy_strategies = {
            EmotionType.JOY.value: [EmpathyStrategy.ENCOURAGE.value, EmpathyStrategy.AGREE.value],
            EmotionType.CONFUSION.value: [EmpathyStrategy.COMFORT.value, EmpathyStrategy.ADVISE.value],
            EmotionType.ANGER.value: [EmpathyStrategy.COMFORT.value, EmpathyStrategy.AGREE.value],
            EmotionType.ANXIETY.value: [EmpathyStrategy.COMFORT.value, EmpathyStrategy.ENCOURAGE.value],
            EmotionType.HURT.value: [EmpathyStrategy.COMFORT.value, EmpathyStrategy.ENCOURAGE.value],
            EmotionType.SADNESS.value: [EmpathyStrategy.COMFORT.value, EmpathyStrategy.AGREE.value]
        }

    def convert_to_teen_context(self, text: str, relationship: str) -> Tuple[str, Dict[str, str]]:
        """성인 대화를 청소년 맥락으로 변환"""
        converted_text = text
        conversion_log = {}

        # 관계 변환
        original_relationship = relationship
        if relationship in self.teen_context_mapping["relationship"]:
            teen_relationship = self.teen_context_mapping["relationship"][relationship]
            conversion_log["relationship"] = f"{relationship} → {teen_relationship}"

        # 상황/단어 변환
        for adult_word, teen_word in self.teen_context_mapping["situation_words"].items():
            if adult_word in converted_text:
                converted_text = converted_text.replace(adult_word, teen_word)
                conversion_log[adult_word] = teen_word

        # 존댓말 레벨 조정 (친구 관계는 반말로)
        if relationship in ["친구", "동급생", "형제자매"]:
            converted_text = self._adjust_to_casual_speech(converted_text)
            conversion_log["speech_level"] = "반말로 조정"

        return converted_text, conversion_log

    def _adjust_to_casual_speech(self, text: str) -> str:
        """존댓말을 반말로 조정 (간단한 패턴 매칭)"""
        patterns = [
            (r'습니다\.', '야.'),
            (r'해요\.', '해.'),
            (r'이에요\.', '이야.'),
            (r'에요\.', '야.'),
            (r'죠\.', '지.'),
            (r'하세요', '해'),
            (r'드려요', '줄게'),
            (r'입니다', '이야'),
            (r'까요\?', '까?'),
        ]

        for pattern, replacement in patterns:
            text = re.sub(pattern, replacement, text)

        return text

    def load_aihub_data(self, file_paths: List[str]) -> List[Dict]:
        """AI Hub JSON/TSV 파일들 로드"""
        all_data = []

        for file_path in file_paths:
            try:
                logger.info(f"AI Hub 데이터 로드 중: {file_path}")

                if not os.path.exists(file_path):
                    logger.warning(f"파일이 존재하지 않음: {file_path}")
                    continue

                # 파일 확장자에 따라 처리 방법 결정
                file_extension = Path(file_path).suffix.lower()

                if file_extension == '.tsv':
                    dialog_sessions = self._load_tsv_file(file_path)
                elif file_extension in ['.json', '.jsonl']:
                    dialog_sessions = self._load_json_file(file_path)
                else:
                    logger.warning(f"지원하지 않는 파일 형식: {file_extension}")
                    continue

                all_data.extend(dialog_sessions)
                logger.info(f"로드 완료: {len(dialog_sessions)}개 세션")

            except Exception as e:
                logger.error(f"파일 로드 실패 {file_path}: {e}")
                continue

        logger.info(f"전체 AI Hub 데이터 로드 완료: {len(all_data)}개 세션")
        return all_data

    def _load_json_file(self, file_path: str) -> List[Dict]:
        """JSON 파일 로드"""
        with open(file_path, 'r', encoding='utf-8') as f:
            data = json.load(f)

        # AI Hub JSON 데이터 구조에 따라 조정
        if isinstance(data, dict):
            if 'data' in data:
                return data['data']
            elif 'conversations' in data:
                return data['conversations']
            elif 'sessions' in data:
                return data['sessions']
            else:
                return [data]
        elif isinstance(data, list):
            return data
        else:
            return [data]

    def _load_tsv_file(self, file_path: str) -> List[Dict]:
        """TSV 파일 로드 및 대화 세션으로 변환"""
        import pandas as pd

        # 파일명에서 메타데이터 추출
        filename = Path(file_path).stem
        emotion, relationship = self._extract_metadata_from_filename(filename)

        # TSV 파일 읽기
        df = pd.read_csv(file_path, sep='\t', encoding='utf-8')

        # 대화 세션별로 그룹핑
        dialog_sessions = []
        session_groups = df.groupby('id')

        for session_id, group in session_groups:
            # 발화를 순서대로 정렬
            group = group.sort_values('utterance_id')

            # 사용자-시스템 발화 쌍으로 구성
            user_utterances = group[group['utterance_type'] == 0]['utterance_text'].tolist()
            system_utterances = group[group['utterance_type'] == 1]['utterance_text'].tolist()

            # 각 사용자-시스템 쌍을 별도 세션으로 처리
            for i in range(min(len(user_utterances), len(system_utterances))):
                dialog_sessions.append({
                    'session_id': f"{session_id}_{i+1}",
                    'user_utterance': user_utterances[i],
                    'system_response': system_utterances[i],
                    'emotion': emotion,
                    'relationship': relationship,
                    'empathy_label': self._infer_empathy_label(emotion),
                    'original_session_id': session_id,
                    'utterance_pair': i + 1,
                    'total_utterances': len(group),
                    'file_source': filename
                })

        return dialog_sessions

    def _extract_metadata_from_filename(self, filename: str) -> Tuple[str, str]:
        """파일명에서 감정과 관계 정보 추출"""
        # Empathy_기쁨_부모자녀_조손_8 형태의 파일명 파싱
        parts = filename.split('_')

        emotion = '불안'  # 기본값
        relationship = '친구'  # 기본값

        if len(parts) >= 3:
            # 감정 추출 (두 번째 부분)
            if parts[1] in ['기쁨', '당황', '분노', '불안', '상처', '슬픔']:
                emotion = parts[1]

            # 관계 추출 (세 번째 부분)
            if parts[2] in ['부모자녀', '부부', '형제자매', '연인', '친구', '직장동료', '지인']:
                relationship = parts[2]

        return emotion, relationship

    def _infer_empathy_label(self, emotion: str) -> str:
        """감정에 따른 기본 공감 라벨 추론"""
        empathy_mapping = {
            '기쁨': '격려',
            '당황': '위로',
            '분노': '위로',
            '불안': '위로',
            '상처': '위로',
            '슬픔': '위로'
        }
        return empathy_mapping.get(emotion, '위로')

    def process_dialog_session(self, raw_data: Dict) -> Optional[TeenDialogSession]:
        """단일 대화 세션을 청소년 맥락으로 처리"""
        try:
            # AI Hub 데이터 구조에 맞게 필드 추출 (다양한 필드명 지원)
            session_id = raw_data.get('session_id',
                        raw_data.get('id',
                        raw_data.get('conversation_id', str(uuid.uuid4()))))

            # 사용자 발화 (다양한 필드명 지원)
            user_utterance = (raw_data.get('user_utterance') or
                            raw_data.get('input') or
                            raw_data.get('user_input') or
                            raw_data.get('question') or
                            raw_data.get('user') or '')

            # 시스템 응답 (다양한 필드명 지원)
            system_response = (raw_data.get('system_response') or
                             raw_data.get('output') or
                             raw_data.get('system_output') or
                             raw_data.get('answer') or
                             raw_data.get('response') or
                             raw_data.get('assistant') or '')

            # 감정 (기본값: 불안)
            emotion = raw_data.get('emotion', raw_data.get('feeling', '불안'))

            # 관계 (기본값: 친구)
            relationship = raw_data.get('relationship',
                         raw_data.get('relation',
                         raw_data.get('context', '친구')))

            # 공감 라벨 (기본값: 위로)
            empathy_label = raw_data.get('empathy_label',
                           raw_data.get('label',
                           raw_data.get('strategy', '위로')))

            if not user_utterance or not system_response:
                logger.warning(f"필수 필드 누락 - 세션 ID: {session_id}")
                return None

            # 청소년 맥락 변환
            teen_user_utterance, user_conversion = self.convert_to_teen_context(
                user_utterance, relationship
            )
            teen_system_response, system_conversion = self.convert_to_teen_context(
                system_response, relationship
            )

            # 관계도 청소년 맥락으로 변환
            teen_relationship = self.teen_context_mapping["relationship"].get(
                relationship, relationship
            )

            return TeenDialogSession(
                session_id=str(session_id),
                user_utterance=teen_user_utterance,
                system_response=teen_system_response,
                emotion=emotion,
                relationship=teen_relationship,
                empathy_label=empathy_label,
                teen_context={
                    "original_relationship": relationship,
                    "user_conversions": user_conversion,
                    "system_conversions": system_conversion,
                    "conversion_applied": bool(user_conversion or system_conversion)
                }
            )

        except Exception as e:
            logger.error(f"세션 처리 실패: {e}")
            return None

    async def process_and_index_data(self, file_paths: List[str]) -> ProcessingStats:
        """전체 데이터 처리 및 벡터 인덱싱"""
        start_time = datetime.now()

        # 1. AI Hub 데이터 로드
        raw_data_list = self.load_aihub_data(file_paths)
        if not raw_data_list:
            logger.warning("로드된 데이터가 없습니다")
            return ProcessingStats(0, 0, {}, {}, {}, str(datetime.now() - start_time))

        # 2. 청소년 맥락으로 변환
        processed_sessions = []
        for raw_data in raw_data_list:
            session = self.process_dialog_session(raw_data)
            if session:
                processed_sessions.append(session)

        if not processed_sessions:
            logger.warning("처리된 세션이 없습니다")
            return ProcessingStats(0, 0, {}, {}, {}, str(datetime.now() - start_time))

        logger.info(f"데이터 처리 완료: {len(processed_sessions)}개 세션")

        # 3. 통계 계산
        stats = self._calculate_stats(processed_sessions)

        # 4. 벡터 스토어에 인덱싱
        await self._index_to_vector_store(processed_sessions)

        # 5. 처리 시간 기록
        processing_time = str(datetime.now() - start_time)
        stats.processing_time = processing_time

        # 6. 결과 저장
        await self._save_processing_results(processed_sessions, stats)

        logger.info(f"✅ 전체 처리 완료: {stats.total_sessions}개 세션 ({processing_time})")
        return stats

    def _calculate_stats(self, sessions: List[TeenDialogSession]) -> ProcessingStats:
        """처리 통계 계산"""
        emotion_dist = {}
        relationship_dist = {}
        empathy_dist = {}
        teen_converted = 0

        for session in sessions:
            # 감정 분포
            emotion_dist[session.emotion] = emotion_dist.get(session.emotion, 0) + 1

            # 관계 분포
            relationship_dist[session.relationship] = relationship_dist.get(session.relationship, 0) + 1

            # 공감 라벨 분포
            empathy_dist[session.empathy_label] = empathy_dist.get(session.empathy_label, 0) + 1

            # 청소년 변환 수
            if session.teen_context.get("conversion_applied"):
                teen_converted += 1

        return ProcessingStats(
            total_sessions=len(sessions),
            teen_converted=teen_converted,
            emotion_distribution=emotion_dist,
            relationship_distribution=relationship_dist,
            empathy_distribution=empathy_dist,
            processing_time=""  # 나중에 설정
        )

    async def _index_to_vector_store(self, sessions: List[TeenDialogSession]):
        """ChromaDB 벡터 스토어에 인덱싱"""
        try:
            vector_store = await get_vector_store()

            # DocumentInput 형태로 변환
            documents = []
            for session in sessions:
                # 검색용 텍스트 구성 (감정, 관계, 사용자 발화 조합)
                search_content = f"[{session.emotion}] [{session.relationship}] {session.user_utterance}"

                # 메타데이터에 모든 정보 포함
                metadata = {
                    "session_id": session.session_id,
                    "user_utterance": session.user_utterance,
                    "system_response": session.system_response,
                    "emotion": session.emotion,
                    "relationship": session.relationship,
                    "empathy_label": session.empathy_label,
                    "teen_context": json.dumps(session.teen_context),
                    "data_source": "aihub",
                    "processed_for": "teen_empathy",
                    "indexed_at": datetime.now().isoformat()
                }

                documents.append(DocumentInput(
                    content=search_content,
                    metadata=metadata,
                    document_id=session.session_id
                ))

            # 벡터 스토어에 추가 (배치 처리)
            logger.info(f"벡터 인덱싱 시작: {len(documents)}개 문서")
            await vector_store.add_documents(documents)
            logger.info("✅ 벡터 인덱싱 완료")

        except Exception as e:
            logger.error(f"❌ 벡터 인덱싱 실패: {e}")
            raise

    async def _save_processing_results(self, sessions: List[TeenDialogSession], stats: ProcessingStats):
        """처리 결과 저장"""
        try:
            # 통계 저장
            stats_file = self.data_dir / "teen_empathy_stats.json"
            with open(stats_file, 'w', encoding='utf-8') as f:
                json.dump(asdict(stats), f, ensure_ascii=False, indent=2)

            # 샘플 세션들 저장 (처음 100개)
            sample_sessions = sessions[:100]
            samples_file = self.data_dir / "teen_empathy_samples.json"
            with open(samples_file, 'w', encoding='utf-8') as f:
                json.dump([asdict(session) for session in sample_sessions],
                         f, ensure_ascii=False, indent=2)

            logger.info(f"처리 결과 저장 완료: {stats_file}, {samples_file}")

        except Exception as e:
            logger.error(f"결과 저장 실패: {e}")

    async def search_similar_contexts(self, query: str, emotion: str = None,
                                    relationship: str = None, top_k: int = 5) -> List[Dict]:
        """유사한 대화 맥락 검색"""
        try:
            vector_store = await get_vector_store()

            # 검색 쿼리 구성
            search_query = query
            if emotion:
                search_query = f"[{emotion}] {search_query}"
            if relationship:
                search_query = f"[{relationship}] {search_query}"

            # 메타데이터 필터 (AI Hub 데이터만)
            filter_metadata = {"data_source": "aihub"}

            # 벡터 검색
            results = await vector_store.search(
                query=search_query,
                top_k=top_k,
                filter_metadata=filter_metadata
            )

            # 결과 포맷팅
            formatted_results = []
            for result in results:
                formatted_results.append({
                    "user_utterance": result.metadata.get("user_utterance", ""),
                    "system_response": result.metadata.get("system_response", ""),
                    "emotion": result.metadata.get("emotion", ""),
                    "relationship": result.metadata.get("relationship", ""),
                    "empathy_label": result.metadata.get("empathy_label", ""),
                    "similarity_score": result.score,
                    "teen_context": json.loads(result.metadata.get("teen_context", "{}"))
                })

            return formatted_results

        except Exception as e:
            logger.error(f"유사 맥락 검색 실패: {e}")
            return []

    def get_empathy_strategy(self, emotion: str) -> List[str]:
        """감정에 따른 공감 전략 추천"""
        return self.empathy_strategies.get(emotion, [EmpathyStrategy.COMFORT.value, EmpathyStrategy.ENCOURAGE.value])

    async def create_sample_data(self) -> List[Dict]:
        """샘플 AI Hub 데이터 생성 (테스트용)"""
        sample_data = [
            {
                "session_id": "sample_001",
                "user_utterance": "직장에서 동료와 갈등이 있어서 너무 스트레스 받아요.",
                "system_response": "직장 내 갈등은 정말 힘들죠. 어떤 상황인지 더 자세히 말해주실 수 있나요?",
                "emotion": "분노",
                "relationship": "직장동료",
                "empathy_label": "위로"
            },
            {
                "session_id": "sample_002",
                "user_utterance": "부모님이 성적 때문에 계속 잔소리하셔서 짜증나요.",
                "system_response": "부모님의 기대가 부담스러우시겠어요. 부모님과 솔직하게 대화해보는 것은 어떨까요?",
                "emotion": "분노",
                "relationship": "부모자녀",
                "empathy_label": "조언"
            },
            {
                "session_id": "sample_003",
                "user_utterance": "친구들과 시험을 잘 봐서 정말 기뻐요!",
                "system_response": "와! 정말 축하해! 열심히 공부한 보람이 있겠다. 어떤 기분이야?",
                "emotion": "기쁨",
                "relationship": "친구",
                "empathy_label": "격려"
            },
            {
                "session_id": "sample_004",
                "user_utterance": "연인과 헤어져서 너무 슬퍼요...",
                "system_response": "정말 마음이 아플 것 같아. 지금은 많이 슬프겠지만 시간이 해결해줄 거야.",
                "emotion": "슬픔",
                "relationship": "연인",
                "empathy_label": "위로"
            },
            {
                "session_id": "sample_005",
                "user_utterance": "형제가 제 물건을 허락없이 써서 화가 나요.",
                "system_response": "형제끼리도 서로 허락을 구하는 게 좋겠어. 형제와 대화해보는 건 어때?",
                "emotion": "분노",
                "relationship": "형제자매",
                "empathy_label": "조언"
            }
        ]

        return sample_data


# 전역 인스턴스
_processor_instance = None

async def get_teen_empathy_processor() -> TeenEmpathyDataProcessor:
    """청소년 공감 데이터 처리기 싱글톤 인스턴스"""
    global _processor_instance

    if _processor_instance is None:
        data_dir = os.getenv("AIHUB_DATA_DIR", "./data/aihub")
        _processor_instance = TeenEmpathyDataProcessor(data_dir)

    return _processor_instance