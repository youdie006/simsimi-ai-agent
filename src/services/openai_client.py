"""
OpenAI GPT-4 클라이언트 서비스 - True RAG 시스템 지원
AI Hub 검색 결과 기반 응답 생성에 최적화
"""

import os
import time
from typing import List, Optional, Dict, Any
import asyncio
from openai import AsyncOpenAI
from loguru import logger

from ..models.function_models import (
    ChatMessage, OpenAICompletionRequest, OpenAICompletionResponse,
    EmotionAnalysisRequest, EmotionAnalysisResponse,
    EmotionType, RelationshipType, EmpathyStrategy
)


class OpenAIClient:
    """OpenAI GPT-4 클라이언트 - True RAG 전용"""

    def __init__(self):
        self.client = None
        self.api_key = os.getenv("OPENAI_API_KEY")
        self.default_model = os.getenv("OPENAI_MODEL", "gpt-4")
        self.max_retries = 3
        self.retry_delay = 1

        # True RAG 전용 시스템 프롬프트
        self.teen_empathy_system_prompt = """
당신은 "마음이"라는 이름의 13-19세 청소년 전용 상담 AI입니다.

🔥 **중요**: 이전 대화 맥락을 반드시 고려해서 일관성 있는 응답을 하세요.
- 사용자가 이전에 말한 문제와 연결해서 답변하세요
- 갑자기 다른 주제로 넘어가지 마세요
- 대화의 흐름을 유지하세요

**마음이의 성격:**
- 따뜻하고 친근한 AI 친구
- 청소년의 눈높이에서 대화하는 상담자
- 전문 상담사들의 조언을 청소년 친화적으로 전달
- 절대 판단하지 않고 항상 편에서 들어주는 친구

**대화 원칙:**
1. 🤗 **공감이 먼저**: "정말 힘들겠어", "그런 마음 충분히 이해해"
2. 💭 **청소년 언어**: 친근한 반말, 자연스러운 표현 사용
3. 💡 **구체적 조언**: 실제로 할 수 있는 현실적인 방법 제시
4. 🌟 **희망적 마무리**: 응원과 격려로 대화 마무리

**응답 스타일:**
✅ "그런 상황이면 나도 화가 날 것 같아"
✅ "한 번 이렇게 해보는 건 어때?"
✅ "네 마음이 제일 중요해"
✅ "언제든 다시 이야기해줘"

❌ 어른스러운 훈계나 설교 금지
❌ "아직 어려서", "나중에 알게 될 거야" 같은 표현 금지
❌ 너무 형식적이거나 딱딱한 조언 금지

전문 상담사의 지혜를 청소년이 받아들이기 쉬운 친구의 조언으로 전달하세요.
"""

    async def initialize(self):
        """OpenAI 클라이언트 초기화"""
        try:
            if not self.api_key:
                raise ValueError("OPENAI_API_KEY 환경변수가 설정되지 않았습니다")

            if "your_" in self.api_key.lower():
                raise ValueError("올바른 OpenAI API 키를 설정해주세요")

            self.client = AsyncOpenAI(
                api_key=self.api_key,
                timeout=30.0,
                max_retries=self.max_retries
            )

            await self._test_connection()
            logger.info("✅ OpenAI 클라이언트 초기화 완료 (True RAG 모드)")

        except Exception as e:
            logger.error(f"❌ OpenAI 클라이언트 초기화 실패: {e}")
            raise

    async def _test_connection(self):
        """OpenAI API 연결 테스트"""
        try:
            response = await self.client.chat.completions.create(
                model=self.default_model,
                messages=[{"role": "user", "content": "Hello"}],
                max_tokens=5,
                temperature=0
            )
            logger.info("OpenAI API 연결 테스트 성공")
        except Exception as e:
            logger.error(f"OpenAI API 연결 테스트 실패: {e}")
            raise

    async def create_completion(self,
                             messages: List[Dict[str, str]],
                             model: Optional[str] = None,
                             temperature: float = 0.7,
                             max_tokens: int = 500,
                             **kwargs) -> OpenAICompletionResponse:
        """GPT-4 채팅 완성 생성"""
        try:
            if not self.client:
                await self.initialize()

            start_time = time.time()
            model_name = model or self.default_model

            for attempt in range(self.max_retries):
                try:
                    response = await self.client.chat.completions.create(
                        model=model_name,
                        messages=messages,
                        temperature=temperature,
                        max_tokens=max_tokens,
                        **kwargs
                    )

                    processing_time = (time.time() - start_time) * 1000

                    return OpenAICompletionResponse(
                        content=response.choices[0].message.content,
                        model=response.model,
                        tokens_used=response.usage.total_tokens,
                        processing_time_ms=processing_time,
                        finish_reason=response.choices[0].finish_reason
                    )

                except Exception as e:
                    if attempt < self.max_retries - 1:
                        logger.warning(f"OpenAI API 호출 실패 (재시도 {attempt + 1}/{self.max_retries}): {e}")
                        await asyncio.sleep(self.retry_delay * (attempt + 1))
                        continue
                    else:
                        raise

        except Exception as e:
            logger.error(f"❌ OpenAI 완성 생성 실패: {e}")
            raise

    async def analyze_emotion_and_context(self,
                                        text: str,
                                        additional_context: str = None) -> EmotionAnalysisResponse:
        """텍스트에서 감정과 관계 맥락 분석"""
        try:
            emotions = [e.value for e in EmotionType]
            relationships = [r.value for r in RelationshipType]
            strategies = [s.value for s in EmpathyStrategy]

            analysis_prompt = f"""
다음 청소년의 메시지를 분석해주세요:

메시지: "{text}"
{f"추가 맥락: {additional_context}" if additional_context else ""}

**분석 요청:**
1. 주요 감정: {', '.join(emotions)} 중 선택
2. 관계 맥락: {', '.join(relationships)} 중 선택 (없으면 null)
3. 추천 공감 전략: {', '.join(strategies)} 중 1-3개 선택
4. 감정 신뢰도: 0-1 사이 숫자

JSON 형태로만 응답하세요:
{{
    "primary_emotion": "감정",
    "emotion_confidence": 0.85,
    "relationship_context": "관계" (또는 null),
    "recommended_strategies": ["전략1", "전략2"],
    "analysis_details": {{
        "keywords": ["키워드1", "키워드2"],
        "intensity": "low/medium/high"
    }}
}}
"""

            response = await self.create_completion(
                messages=[{"role": "user", "content": analysis_prompt}],
                temperature=0.3,
                max_tokens=200
            )

            import json
            result = json.loads(response.content.strip())

            # 유효성 검사 및 기본값 설정
            primary_emotion = result.get("primary_emotion", "불안")
            if primary_emotion not in emotions:
                primary_emotion = "불안"

            relationship_context = result.get("relationship_context")
            if relationship_context and relationship_context not in relationships:
                relationship_context = None

            recommended_strategies = result.get("recommended_strategies", ["위로"])
            recommended_strategies = [s for s in recommended_strategies if s in strategies]
            if not recommended_strategies:
                recommended_strategies = ["위로"]

            return EmotionAnalysisResponse(
                primary_emotion=EmotionType(primary_emotion),
                emotion_confidence=max(0.0, min(1.0, result.get("emotion_confidence", 0.7))),
                relationship_context=RelationshipType(relationship_context) if relationship_context else None,
                recommended_strategies=[EmpathyStrategy(s) for s in recommended_strategies],
                analysis_details=result.get("analysis_details", {})
            )

        except Exception as e:
            logger.error(f"감정 분석 실패: {e}")
            return EmotionAnalysisResponse(
                primary_emotion=EmotionType.ANXIETY,
                emotion_confidence=0.5,
                relationship_context=None,
                recommended_strategies=[EmpathyStrategy.COMFORT],
                analysis_details={"error": "분석 실패"}
            )

    async def adapt_expert_response_to_teen(self,
                                          original_expert_response: str,
                                          user_situation: str,
                                          emotion: str,
                                          relationship: str,
                                          adaptation_level: str = "moderate") -> str:
        """🔥 True RAG 핵심: 전문가 응답을 청소년 맥락으로 적응"""
        
        # 🔥 즉시 강력한 변환 적용
        original_expert_response = original_expert_response.replace('자기야', '너')
        original_expert_response = original_expert_response.replace('자기가', '네가')
        original_expert_response = original_expert_response.replace('자기도', '너도')
        original_expert_response = original_expert_response.replace('자기를', '너를')
        original_expert_response = original_expert_response.replace('자기의', '네')
        original_expert_response = original_expert_response.replace('무슨 공부이', '무슨 일이')
        original_expert_response = original_expert_response.replace('어떤 공부이', '어떤 일이')
        original_expert_response = original_expert_response.replace('하세요', '해')
        original_expert_response = original_expert_response.replace('어떠세요', '어때')

        if adaptation_level == "minimal":
            return await self._minimal_adaptation(original_expert_response, relationship)
        elif adaptation_level == "moderate":
            return await self._moderate_adaptation(original_expert_response, user_situation, emotion, relationship)
        else:
            return await self._full_adaptation(original_expert_response, user_situation, emotion, relationship)

    async def _minimal_adaptation(self, expert_response: str, relationship: str) -> str:
        """최소 적응: 기본 용어만 변환"""

        teen_response = expert_response

        # 기본 용어 변환 맵
        conversion_map = {
            "자기야": "야", "자기가": "너가", "자기도": "너도", "자기의": "네", "자기를": "너를",
            "자기한테": "너한테", "자기께서": "네가", "자기는": "너는", "자기와": "너와",
            "당신": "너", "당신이": "네가", "당신의": "네", "당신을": "너를", "당신과": "너와",
            
            # 성인 상황 → 청소년 상황
            "직장": "학교", "회사": "학교", "사무실": "교실", "사무소": "학교",
            "업무": "공부", "작업": "과제", "근무": "수업", "출근": "등교", "퇴근": "하교",
            "동료": "친구", "상사": "선생님", "부하직원": "후배", "직원": "학생",
            "회의": "수업", "미팅": "모임", "프로젝트": "과제", "출장": "현장학습",
            "야근": "야자", "휴가": "방학", "퇴사": "전학", "급여": "용돈",
            
            # 존댓말 → 반말 (친구 관계)
            "하십시오": "해", "하세요": "해", "하시죠": "하자", "해주세요": "해줘",
            "어떠십니까": "어때", "어떠세요": "어때", "해보세요": "해봐", "보세요": "봐",
            "그렇습니다": "그래", "그렇군요": "그렇구나", "이해합니다": "이해해",
            "말씀하세요": "말해줘", "생각해보세요": "생각해봐", "드릴게요": "줄게",
            
            # 성인 표현 → 청소년 표현
            "힘드시겠어요": "힘들겠어", "속상하시겠어요": "속상하겠어",
            "이해가 되시나요": "이해돼", "어떻게 생각하세요": "어떻게 생각해",
            "괜찮으시다면": "괜찮다면", "고민이 되시나요": "고민돼",
            
            # 오타 방지
            "무슨 일이": "무슨 일이", "어떤 일이": "어떤 일이",  # 명시적 보호
        }

        for adult_term, teen_term in conversion_map.items():
            teen_response = teen_response.replace(adult_term, teen_term)

        # 관계별 추가 조정
        if relationship in ["친구", "동급생"]:
            teen_response = teen_response.replace("당신", "너")
            teen_response = teen_response.replace("당신이", "네가")
            teen_response = teen_response.replace("당신의", "네")
            teen_response = teen_response.replace("귀하", "너")

        return teen_response

    async def _moderate_adaptation(self, expert_response: str, user_situation: str,
                                 emotion: str, relationship: str) -> str:
        """중간 적응: 구조 유지하되 내용 조정"""

        adaptation_prompt = f"""
다음은 AI Hub 전문 상담사의 실제 응답입니다. 이를 13-19세 청소년 상황에 맞게 적응시켜주세요.

전문가 원본 응답: "{expert_response}"
청소년 상황: "{user_situation}" (감정: {emotion}, 관계: {relationship})

적응 요구사항:
1. 원본의 공감 방식과 해결 접근법을 완전히 유지
2. 청소년에게 맞는 친근하고 따뜻한 표현으로 변환
3. 구체적이고 실행 가능한 조언 유지
4. 원본 길이와 비슷하게 유지
5. "너", "네가" 등 청소년 친화적 호칭 사용

청소년 맞춤 응답:
"""

        try:
            response = await self.create_completion(
                messages=[{"role": "user", "content": adaptation_prompt}],
                temperature=0.3,
                max_tokens=300
            )

            return response.content.strip()

        except Exception as e:
            logger.error(f"중간 적응 실패: {e}")
            return await self._minimal_adaptation(expert_response, relationship)

    async def _full_adaptation(self, expert_response: str, user_situation: str,
                             emotion: str, relationship: str) -> str:
        """완전 적응: 전면적 재구성"""

        full_adaptation_prompt = f"""
다음 전문 상담사 응답의 핵심 접근법을 참고하여, 청소년 상황에 완전히 맞는 새로운 응답을 생성하세요.

전문가 응답 (참고용): "{expert_response}"
청소년 상황: "{user_situation}" (감정: {emotion}, 관계: {relationship})

요구사항:
1. 전문가 응답의 공감 방식과 해결 접근법의 핵심만 참고
2. 청소년이 실제 할 수 있는 구체적 방법 제시
3. 13-19세 눈높이에 맞는 완전히 새로운 표현 사용
4. 친근하고 따뜻하면서도 실용적인 조언
5. 150자 내외로 간결하게

청소년 전용 응답:
"""

        try:
            response = await self.create_completion(
                messages=[{"role": "user", "content": full_adaptation_prompt}],
                temperature=0.5,
                max_tokens=300
            )

            return response.content.strip()

        except Exception as e:
            logger.error(f"완전 적응 실패: {e}")
            return await self._moderate_adaptation(expert_response, user_situation, emotion, relationship)

    async def combine_multiple_expert_responses(self,
                                              expert_responses: List[Dict[str, Any]],
                                              user_situation: str,
                                              emotion: str,
                                              relationship: str) -> str:
        """🔥 여러 전문가 응답을 조합하여 최적 응답 생성"""
        
        # 🔥 강력한 전처리: 모든 expert_responses 변환
        for response_data in expert_responses:
            if 'system_response' in response_data:
                original = response_data['system_response']
                # 연인 호칭 완전 제거
                converted = original
                converted = converted.replace('자기야', '너')
                converted = converted.replace('자기가', '네가')
                converted = converted.replace('자기도', '너도')
                converted = converted.replace('자기를', '너를')
                converted = converted.replace('자기의', '네')
                converted = converted.replace('자기한테', '너한테')
                converted = converted.replace('자기는', '너는')
                converted = converted.replace('당신이', '네가')
                converted = converted.replace('당신을', '너를')
                converted = converted.replace('당신의', '네')
                
                # 오타 방지
                converted = converted.replace('무슨 공부이', '무슨 일이')
                converted = converted.replace('어떤 공부이', '어떤 일이')
                converted = converted.replace('공부이 있었', '일이 있었')
                
                # 존댓말 → 반말
                converted = converted.replace('하세요', '해')
                converted = converted.replace('어떠세요', '어때')
                converted = converted.replace('해보세요', '해봐')
                converted = converted.replace('말씀하세요', '말해줘')
                
                response_data['system_response'] = converted

        combination_prompt = f"""
다음은 여러 전문 상담사들의 실제 조언들입니다. 
이들의 장점을 조합하여 현재 청소년 상황에 최적화된 하나의 응답을 만들어주세요.

청소년 상황: "{user_situation}" (감정: {emotion}, 관계: {relationship})

전문가 응답들:
"""

        for i, resp_data in enumerate(expert_responses, 1):
            expert_response = resp_data.get('system_response', resp_data.get('response', ''))
            similarity = resp_data.get('similarity_score', resp_data.get('score', 0))
            empathy_type = resp_data.get('empathy_label', resp_data.get('empathy_strategy', ''))
            original_situation = resp_data.get('user_utterance', '')

            combination_prompt += f"""
전문가 {i} (유사도: {similarity:.2f}, 전략: {empathy_type}):
원래 상황: "{original_situation}"
전문가 응답: "{expert_response}"

"""

        combination_prompt += f"""
위 전문가 응답들의 장점을 조합하여 청소년 맞춤 응답을 만들어주세요:

🔥 핵심 변환 원칙:
1. "자기야/자기가" → "너/네가"로 완전 변환
2. 존댓말 → 친근한 반말 ("하세요" → "해", "어떠세요" → "어때")
3. 성인 상황 → 청소년 상황으로 자연스럽게 변환
4. 각 전문가의 핵심 조언과 공감 방식을 모두 활용
5. 13-19세가 실제로 할 수 있는 구체적 방법 제시
6. 친한 선배가 후배에게 조언하는 자연스러운 톤

📋 요구사항:
- 원본들의 핵심 메시지 보존
- 청소년 친화적 표현으로 완전 변환
- 실행 가능한 구체적 조언 포함
- 100-150자 내외로 간결하게

청소년 맞춤 조합 응답:
"""

        try:
            response = await self.create_completion(
                messages=[{"role": "user", "content": combination_prompt}],
                temperature=0.4,
                max_tokens=400
            )

            return response.content.strip()

        except Exception as e:
            logger.error(f"다중 응답 조합 실패: {e}")
            # 폴백: 가장 유사도 높은 응답만 사용
            best_response = max(expert_responses, key=lambda x: x.get('similarity_score', x.get('score', 0)))
            expert_text = best_response.get('system_response', best_response.get('response', ''))
            return await self._moderate_adaptation(expert_text, user_situation, emotion, relationship)

    async def create_teen_empathy_response(self,
                                        user_message: str,
                                        conversation_history: List[ChatMessage] = None,
                                        context_info: str = None) -> str:
        """청소년 공감형 응답 생성 (폴백용)"""
        try:
            messages = [{"role": "system", "content": self.teen_empathy_system_prompt}]

            if conversation_history:
                recent_history = conversation_history[-6:]
                for msg in recent_history:
                    messages.append({
                        "role": msg.role.value,
                        "content": msg.content
                    })

            if context_info:
                messages.append({
                    "role": "system",
                    "content": f"참고 정보: {context_info}"
                })

            messages.append({"role": "user", "content": user_message})

            response = await self.create_completion(
                messages=messages,
                temperature=0.7,
                max_tokens=400
            )

            return response.content

        except Exception as e:
            logger.error(f"청소년 공감 응답 생성 실패: {e}")
            return "지금 많이 힘들 것 같아. 네 마음을 이해해. 무엇이든 편하게 이야기해줘, 함께 해결방법을 찾아보자! 💙"


# 전역 인스턴스
_openai_client_instance = None

async def get_openai_client() -> OpenAIClient:
    """OpenAI 클라이언트 싱글톤 인스턴스"""
    global _openai_client_instance

    if _openai_client_instance is None:
        _openai_client_instance = OpenAIClient()
        await _openai_client_instance.initialize()

    return _openai_client_instance