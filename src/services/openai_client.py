"""
OpenAI GPT-4 클라이언트 서비스
청소년 공감형 채팅에 최적화된 GPT-4 API 래퍼
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
    """OpenAI GPT-4 클라이언트"""

    def __init__(self):
        self.client = None
        self.api_key = os.getenv("OPENAI_API_KEY")
        self.default_model = os.getenv("OPENAI_MODEL", "gpt-4")
        self.max_retries = 3
        self.retry_delay = 1  # 초

        # 청소년 공감형 시스템 프롬프트
        self.teen_empathy_system_prompt = """
당신은 13-19세 청소년을 위한 전문 공감 상담사입니다.

**핵심 원칙:**
1. 청소년의 감정을 먼저 인정하고 공감하기
2. 판단하지 않고 따뜻하게 들어주기  
3. 구체적이고 실행 가능한 해결방안 제시하기
4. 청소년 눈높이에 맞는 친근한 언어 사용하기

**응답 방식:**
✅ "힘들겠다", "충분히 그럴 수 있어", "네 마음 이해해" 등 공감 표현
✅ 구체적인 행동 방안 2-3개 제시
✅ 따뜻하고 지지적인 톤 유지
✅ 청소년이 실제로 할 수 있는 현실적 방법들

❌ 추상적이거나 뻔한 조언 피하기
❌ 어른 관점의 설교하지 않기
❌ "그냥 참아라", "넌 아직 어려서" 같은 표현 피하기

청소년의 안전과 wellbeing을 최우선으로 고려하세요.
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

            # API 연결 테스트
            await self._test_connection()
            logger.info("✅ OpenAI 클라이언트 초기화 완료")

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

            # 재시도 로직
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

    async def create_teen_empathy_response(self,
                                        user_message: str,
                                        conversation_history: List[ChatMessage] = None,
                                        context_info: str = None) -> str:
        """청소년 공감형 응답 생성"""
        try:
            # 메시지 구성
            messages = [{"role": "system", "content": self.teen_empathy_system_prompt}]

            # 대화 히스토리 추가
            if conversation_history:
                for msg in conversation_history[-5:]:  # 최근 5개만
                    messages.append({
                        "role": msg.role.value,
                        "content": msg.content
                    })

            # 맥락 정보 추가
            if context_info:
                messages.append({
                    "role": "system",
                    "content": f"참고 정보: {context_info}"
                })

            # 현재 사용자 메시지
            messages.append({"role": "user", "content": user_message})

            response = await self.create_completion(
                messages=messages,
                temperature=0.7,
                max_tokens=400
            )

            return response.content

        except Exception as e:
            logger.error(f"청소년 공감 응답 생성 실패: {e}")
            # 폴백 응답
            return "지금 많이 힘들 것 같아. 네 마음을 이해해. 무엇이든 편하게 이야기해줘, 함께 해결방법을 찾아보자! 💙"

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

            # JSON 파싱
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
            # 폴백 응답
            return EmotionAnalysisResponse(
                primary_emotion=EmotionType.ANXIETY,
                emotion_confidence=0.5,
                relationship_context=None,
                recommended_strategies=[EmpathyStrategy.COMFORT],
                analysis_details={"error": "분석 실패"}
            )

    async def generate_react_response(self,
                                    user_message: str,
                                    similar_contexts: List[Dict[str, Any]] = None,
                                    emotion: str = None,
                                    relationship: str = None) -> tuple[str, List[Dict[str, str]]]:
        """ReAct 패턴으로 단계별 추론 후 응답 생성"""
        try:
            # 유사 맥락 정보 구성
            context_info = ""
            if similar_contexts:
                context_info = "\n**유사한 상황들:**\n"
                for i, ctx in enumerate(similar_contexts[:2], 1):
                    context_info += f"{i}. 상황: {ctx.get('user_utterance', '')}\n"
                    context_info += f"   해결: {ctx.get('system_response', '')}\n"
                    context_info += f"   공감방식: {ctx.get('empathy_label', '')}\n"

            react_prompt = f"""
{self.teen_empathy_system_prompt}

**현재 상황:**
- 사용자 메시지: "{user_message}"
{f"- 감지된 감정: {emotion}" if emotion else ""}
{f"- 관계 맥락: {relationship}" if relationship else ""}

{context_info}

**ReAct 패턴으로 단계별 추론하세요:**

**THOUGHT:** 사용자의 상황과 감정을 분석
**ACTION:** 유사 상황 참고 및 공감 전략 선택  
**OBSERVATION:** 분석 결과 및 적절한 접근법 결정
**RESPONSE:** 최종 공감형 응답

각 단계를 명확히 구분해서 작성해주세요.
"""

            response = await self.create_completion(
                messages=[{"role": "user", "content": react_prompt}],
                temperature=0.7,
                max_tokens=600
            )

            full_response = response.content

            # ReAct 단계 파싱
            react_steps = self._parse_react_steps(full_response)

            # 최종 응답 추출
            final_response = self._extract_final_response(full_response)

            return final_response, react_steps

        except Exception as e:
            logger.error(f"ReAct 응답 생성 실패: {e}")
            fallback_response = await self.create_teen_empathy_response(user_message)
            return fallback_response, []

    def _parse_react_steps(self, full_response: str) -> List[Dict[str, str]]:
        """ReAct 단계 파싱"""
        steps = []
        current_step = None
        current_content = []

        for line in full_response.split('\n'):
            line = line.strip()
            if line.startswith('**THOUGHT:**'):
                if current_step:
                    steps.append({
                        "step_type": current_step,
                        "content": '\n'.join(current_content).strip()
                    })
                current_step = "thought"
                current_content = [line.replace('**THOUGHT:**', '').strip()]
            elif line.startswith('**ACTION:**'):
                if current_step:
                    steps.append({
                        "step_type": current_step,
                        "content": '\n'.join(current_content).strip()
                    })
                current_step = "action"
                current_content = [line.replace('**ACTION:**', '').strip()]
            elif line.startswith('**OBSERVATION:**'):
                if current_step:
                    steps.append({
                        "step_type": current_step,
                        "content": '\n'.join(current_content).strip()
                    })
                current_step = "observation"
                current_content = [line.replace('**OBSERVATION:**', '').strip()]
            elif line.startswith('**RESPONSE:**'):
                if current_step:
                    steps.append({
                        "step_type": current_step,
                        "content": '\n'.join(current_content).strip()
                    })
                # RESPONSE는 별도로 처리
                break
            elif current_step and line:
                current_content.append(line)

        # 마지막 단계 추가
        if current_step and current_content:
            steps.append({
                "step_type": current_step,
                "content": '\n'.join(current_content).strip()
            })

        return steps

    def _extract_final_response(self, full_response: str) -> str:
        """최종 응답 추출"""
        if "**RESPONSE:**" in full_response:
            parts = full_response.split("**RESPONSE:**")
            if len(parts) > 1:
                return parts[-1].strip()

        # ReAct 키워드 이후 마지막 부분 찾기
        lines = full_response.split('\n')
        react_keywords = ["**THOUGHT:**", "**ACTION:**", "**OBSERVATION:**", "**RESPONSE:**"]

        final_lines = []
        found_response = False

        for line in lines:
            if "**RESPONSE:**" in line:
                found_response = True
                final_lines = [line.replace("**RESPONSE:**", "").strip()]
                continue
            elif found_response and not any(keyword in line for keyword in react_keywords):
                if line.strip():
                    final_lines.append(line.strip())
            elif found_response and any(keyword in line for keyword in react_keywords):
                break

        if final_lines:
            return '\n'.join(final_lines).strip()

        return full_response  # 폴백


# 전역 인스턴스
_openai_client_instance = None


async def get_openai_client() -> OpenAIClient:
    """OpenAI 클라이언트 싱글톤 인스턴스"""
    global _openai_client_instance

    if _openai_client_instance is None:
        _openai_client_instance = OpenAIClient()
        await _openai_client_instance.initialize()

    return _openai_client_instance