"""
🗄️ 수정된 대화 저장 시스템 - SQLite 락 문제 해결
"""

import sqlite3
import json
import uuid
import os
import time
from datetime import datetime, timedelta
from typing import List, Dict, Optional, Tuple
from dataclasses import dataclass
from pathlib import Path
from contextlib import contextmanager

from loguru import logger


@dataclass
class ConversationMessage:
    """대화 메시지 모델"""
    id: Optional[int] = None
    session_id: str = None
    role: str = None  # 'user' or 'assistant'
    content: str = None
    timestamp: str = None
    emotion: Optional[str] = None
    empathy_strategy: Optional[str] = None
    topic: Optional[str] = None
    context_used: bool = False
    similarity_score: Optional[float] = None
    metadata: Optional[str] = None  # JSON string


class ConversationDatabase:
    """SQLite 기반 대화 DB 관리자 - 락 문제 해결 버전"""
    
    def __init__(self, db_path: str = None):
        if db_path is None:
            db_path = os.getenv("CONVERSATION_DB_PATH", "/app/data/conversations/conversations.db")
        
        self.db_path = Path(db_path)
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        self._ensure_tables()
        logger.info(f"✅ 대화 DB 초기화: {self.db_path}")
        
    def _ensure_tables(self):
        """DB 테이블 생성 - 락 방지 버전"""
        max_retries = 3
        for attempt in range(max_retries):
            try:
                with self._get_connection() as conn:
                    conn.executescript("""
                    CREATE TABLE IF NOT EXISTS conversations (
                        id INTEGER PRIMARY KEY AUTOINCREMENT,
                        session_id TEXT NOT NULL,
                        role TEXT NOT NULL CHECK (role IN ('user', 'assistant')),
                        content TEXT NOT NULL,
                        timestamp TEXT NOT NULL,
                        emotion TEXT,
                        empathy_strategy TEXT,
                        topic TEXT,
                        context_used BOOLEAN DEFAULT FALSE,
                        similarity_score REAL,
                        metadata TEXT,
                        created_at DATETIME DEFAULT CURRENT_TIMESTAMP
                    );
                    
                    CREATE TABLE IF NOT EXISTS sessions (
                        session_id TEXT PRIMARY KEY,
                        created_at TEXT NOT NULL,
                        last_activity TEXT NOT NULL,
                        message_count INTEGER DEFAULT 0,
                        topics TEXT DEFAULT '[]',
                        emotions TEXT DEFAULT '{}',
                        is_active BOOLEAN DEFAULT TRUE,
                        expires_at TEXT
                    );
                    
                    -- 성능 최적화 인덱스
                    CREATE INDEX IF NOT EXISTS idx_conversations_session ON conversations(session_id);
                    CREATE INDEX IF NOT EXISTS idx_conversations_timestamp ON conversations(timestamp);
                    CREATE INDEX IF NOT EXISTS idx_conversations_topic ON conversations(topic);
                    CREATE INDEX IF NOT EXISTS idx_conversations_emotion ON conversations(emotion);
                    CREATE INDEX IF NOT EXISTS idx_sessions_active ON sessions(is_active);
                    CREATE INDEX IF NOT EXISTS idx_sessions_expires ON sessions(expires_at);
                    """)
                    
                logger.info("✅ 대화 DB 테이블 초기화 완료")
                break
                
            except sqlite3.OperationalError as e:
                if "database is locked" in str(e) and attempt < max_retries - 1:
                    logger.warning(f"DB 락 감지, 재시도 {attempt + 1}/{max_retries}")
                    time.sleep(1)
                    continue
                else:
                    raise

    @contextmanager
    def _get_connection(self):
        """DB 연결 컨텍스트 매니저 - 락 방지 강화"""
        conn = None
        try:
            conn = sqlite3.connect(
                self.db_path,
                timeout=30.0,  # 30초 타임아웃
                isolation_level=None  # 자동 커밋 모드
            )
            conn.row_factory = sqlite3.Row  # dict-like access
            
            # WAL 모드 활성화 (동시성 향상)
            conn.execute("PRAGMA journal_mode=WAL")
            conn.execute("PRAGMA busy_timeout=30000")  # 30초 대기
            conn.execute("PRAGMA synchronous=NORMAL")
            
            yield conn
            
        except Exception as e:
            if conn:
                try:
                    conn.rollback()
                except:
                    pass
            raise e
        finally:
            if conn:
                try:
                    conn.close()
                except:
                    pass

    def create_session(self, retention_days: int = 7) -> str:
        """새 세션 생성 - 락 방지"""
        session_id = f"session_{uuid.uuid4().hex[:12]}"
        now = datetime.now().isoformat()
        expires_at = (datetime.now() + timedelta(days=retention_days)).isoformat()
        
        max_retries = 3
        for attempt in range(max_retries):
            try:
                with self._get_connection() as conn:
                    conn.execute("""
                        INSERT OR IGNORE INTO sessions (session_id, created_at, last_activity, expires_at)
                        VALUES (?, ?, ?, ?)
                    """, (session_id, now, now, expires_at))
                    
                logger.info(f"📝 새 세션 생성: {session_id}")
                return session_id
                
            except sqlite3.OperationalError as e:
                if "database is locked" in str(e) and attempt < max_retries - 1:
                    logger.warning(f"세션 생성 중 DB 락, 재시도 {attempt + 1}")
                    time.sleep(0.5)
                    continue
                else:
                    raise

    def save_message(self, message: ConversationMessage) -> int:
        """메시지 저장 - 락 방지"""
        max_retries = 3
        for attempt in range(max_retries):
            try:
                with self._get_connection() as conn:
                    cursor = conn.execute("""
                        INSERT INTO conversations 
                        (session_id, role, content, timestamp, emotion, empathy_strategy, 
                         topic, context_used, similarity_score, metadata)
                        VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                    """, (
                        message.session_id, message.role, message.content, 
                        message.timestamp, message.emotion, message.empathy_strategy,
                        message.topic, message.context_used, message.similarity_score,
                        message.metadata
                    ))
                    
                    message_id = cursor.lastrowid
                    
                    # 세션 메타데이터 업데이트
                    self._update_session_metadata(message.session_id, message)
                    
                    logger.info(f"💾 메시지 저장: ID {message_id}, {message.role}")
                    return message_id
                    
            except sqlite3.OperationalError as e:
                if "database is locked" in str(e) and attempt < max_retries - 1:
                    logger.warning(f"메시지 저장 중 DB 락, 재시도 {attempt + 1}")
                    time.sleep(0.5)
                    continue
                else:
                    logger.error(f"❌ 메시지 저장 실패: {e}")
                    raise
            except Exception as e:
                logger.error(f"❌ 메시지 저장 실패: {e}")
                raise

    def _update_session_metadata(self, session_id: str, message: ConversationMessage):
        """세션 메타데이터 업데이트 - 락 방지"""
        try:
            with self._get_connection() as conn:
                # 현재 메타데이터 조회
                row = conn.execute("""
                    SELECT topics, emotions, message_count FROM sessions WHERE session_id = ?
                """, (session_id,)).fetchone()
                
                if row:
                    topics = json.loads(row['topics']) if row['topics'] else []
                    emotions = json.loads(row['emotions']) if row['emotions'] else {}
                    message_count = row['message_count'] + 1
                    
                    # 주제 업데이트
                    if message.topic and message.topic not in topics:
                        topics.append(message.topic)
                    
                    # 감정 분포 업데이트
                    if message.emotion:
                        emotions[message.emotion] = emotions.get(message.emotion, 0) + 1
                    
                    # 메타데이터 저장
                    conn.execute("""
                        UPDATE sessions 
                        SET last_activity = ?, message_count = ?, topics = ?, emotions = ?
                        WHERE session_id = ?
                    """, (
                        datetime.now().isoformat(), message_count, 
                        json.dumps(topics), json.dumps(emotions), session_id
                    ))
        except Exception as e:
            logger.warning(f"세션 메타데이터 업데이트 실패: {e}")

    def get_smart_context(self, session_id: str, current_message: str, 
                         max_messages: int = 20) -> List[Dict]:
        """스마트 컨텍스트 검색 - 락 방지"""
        try:
            with self._get_connection() as conn:
                # 최근 메시지들만 간단히 조회 (락 위험 최소화)
                recent_messages = conn.execute("""
                    SELECT * FROM conversations 
                    WHERE session_id = ? 
                    ORDER BY timestamp DESC 
                    LIMIT ?
                """, (session_id, max_messages)).fetchall()
                
                # 결과 포맷팅
                formatted_messages = []
                for msg in recent_messages:
                    formatted_messages.append({
                        **dict(msg), 
                        'priority': 3  # 최근 메시지 우선순위
                    })
                
                # 시간순 정렬
                formatted_messages.sort(key=lambda x: x['timestamp'])
                
                logger.info(f"🧠 스마트 컨텍스트: {len(formatted_messages)}개 메시지")
                return formatted_messages
                
        except Exception as e:
            logger.warning(f"컨텍스트 검색 실패: {e}")
            return []

    def get_session_stats(self, session_id: str) -> Dict:
        """세션 통계 조회 - 락 방지"""
        try:
            with self._get_connection() as conn:
                # 세션 기본 정보
                session = conn.execute("""
                    SELECT * FROM sessions WHERE session_id = ?
                """, (session_id,)).fetchone()
                
                if not session:
                    return {}
                
                # 메시지 통계 (간단하게)
                message_count = conn.execute("""
                    SELECT COUNT(*) as total FROM conversations WHERE session_id = ?
                """, (session_id,)).fetchone()
                
                return {
                    'session_id': session_id,
                    'created_at': session['created_at'],
                    'last_activity': session['last_activity'],
                    'total_messages': message_count['total'] if message_count else 0,
                    'topics': json.loads(session['topics']) if session['topics'] else [],
                    'emotions': json.loads(session['emotions']) if session['emotions'] else {},
                    'is_active': bool(session['is_active'])
                }
                
        except Exception as e:
            logger.warning(f"세션 통계 조회 실패: {e}")
            return {}

    def _extract_topic(self, content: str) -> str:
        """주제 추출"""
        topic_keywords = {
            '친구': ['친구', '동급생', '반친구', '절친'],
            '가족': ['부모님', '엄마', '아빠', '형', '누나', '동생', '가족'],
            '학교': ['학교', '수업', '선생님', '시험', '성적', '공부'],
            '연애': ['좋아하는', '짝사랑', '남친', '여친', '데이트'],
            '진로': ['장래', '꿈', '직업', '진로', '미래'],
            '스트레스': ['스트레스', '힘들어', '우울', '불안', '걱정']
        }
        
        for topic, keywords in topic_keywords.items():
            if any(keyword in content for keyword in keywords):
                return topic
        return '일반'

    def get_all_sessions(self, active_only: bool = True) -> List[Dict]:
        """모든 세션 조회 - 락 방지"""
        try:
            with self._get_connection() as conn:
                query = "SELECT * FROM sessions"
                params = []
                
                if active_only:
                    query += " WHERE is_active = TRUE"
                
                query += " ORDER BY last_activity DESC LIMIT 10"  # 제한
                
                sessions = conn.execute(query, params).fetchall()
                return [dict(session) for session in sessions]
                
        except Exception as e:
            logger.warning(f"세션 목록 조회 실패: {e}")
            return []


class ConversationService:
    """대화 서비스 (API 통합) - 락 방지"""
    
    def __init__(self):
        self.db = ConversationDatabase()
        
    async def get_or_create_session(self, session_id: str = None) -> str:
        """세션 가져오기 또는 생성"""
        if session_id:
            # 기존 세션 확인
            stats = self.db.get_session_stats(session_id)
            if stats and stats.get('is_active'):
                logger.info(f"📋 기존 세션: {session_id}")
                return session_id
        
        # 새 세션 생성
        new_session = self.db.create_session()
        logger.info(f"🆕 새 세션: {new_session}")
        return new_session

    async def save_conversation_turn(self, session_id: str, user_message: str,
                                   assistant_response: str, metadata: Dict = None) -> Dict:
        """대화 턴 저장"""
        now = datetime.now().isoformat()
        metadata = metadata or {}
        
        try:
            # 사용자 메시지 저장
            user_msg = ConversationMessage(
                session_id=session_id,
                role='user',
                content=user_message,
                timestamp=now,
                topic=self.db._extract_topic(user_message),
                metadata=json.dumps(metadata.get('user', {}))
            )
            user_id = self.db.save_message(user_msg)
            
            # AI 응답 저장
            ai_msg = ConversationMessage(
                session_id=session_id,
                role='assistant',
                content=assistant_response,
                timestamp=now,
                emotion=metadata.get('emotion'),
                empathy_strategy=json.dumps(metadata.get('empathy_strategy', [])),
                topic=self.db._extract_topic(user_message),
                context_used=metadata.get('context_used', False),
                similarity_score=metadata.get('similarity_score'),
                metadata=json.dumps(metadata.get('assistant', {}))
            )
            ai_id = self.db.save_message(ai_msg)
            
            return {
                'user_message_id': user_id,
                'assistant_message_id': ai_id,
                'session_stats': self.db.get_session_stats(session_id)
            }
            
        except Exception as e:
            logger.error(f"대화 턴 저장 실패: {e}")
            # 오류 시에도 빈 응답 반환 (서비스 중단 방지)
            return {
                'user_message_id': 0,
                'assistant_message_id': 0,
                'session_stats': {}
            }

    async def get_enhanced_context(self, session_id: str, current_message: str) -> Dict:
        """향상된 컨텍스트 제공"""
        context_messages = self.db.get_smart_context(session_id, current_message)
        session_stats = self.db.get_session_stats(session_id)
        
        return {
            'conversation_history': [
                {
                    'role': msg['role'],
                    'content': msg['content'],
                    'timestamp': msg['timestamp'],
                    'emotion': msg['emotion'],
                    'priority': msg.get('priority', 1)
                }
                for msg in context_messages
            ],
            'session_stats': session_stats,
            'context_summary': {
                'total_messages': len(context_messages),
                'topics_covered': list(set(msg.get('topic') for msg in context_messages if msg.get('topic'))),
                'emotions_detected': list(set(msg.get('emotion') for msg in context_messages if msg.get('emotion'))),
                'context_depth': 'deep' if len(context_messages) > 10 else 'shallow'
            }
        }


# 전역 서비스 인스턴스
_conversation_service_instance = None


async def get_conversation_service() -> ConversationService:
    """대화 서비스 싱글톤 인스턴스"""
    global _conversation_service_instance
    
    if _conversation_service_instance is None:
        _conversation_service_instance = ConversationService()
        logger.info("✅ 대화 서비스 초기화 완료")
    
    return _conversation_service_instance
