"""
🧠 청소년 공감형 AI 챗봇 (AI Hub 데이터 기반)
개발자: youdie006@naver.com
기술: ReAct 패턴 + RAG + GPT-4 + ChromaDB + Docker
Target: 13-19세 청소년의 고민 해결과 정서적 지원
"""

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse, HTMLResponse
from fastapi.staticfiles import StaticFiles
import os
import sys
from datetime import datetime
from dotenv import load_dotenv

# 환경 변수 로드
load_dotenv()

# FastAPI 앱 생성
app = FastAPI(
    title="💙 마음이 - 청소년 상담 챗봇",
    description="""
    **13-19세 청소년을 위한 AI 공감 상담사**
    
    📊 **31,821개 전문 상담 사례 활용**
    🤖 **단계별 상황 분석 시스템**
    💙 **6가지 감정 × 4가지 공감전략 맞춤 응답**
    🔍 **벡터 검색으로 유사 상황 참고**
    
    ---
    
    ### 🎯 주요 기능
    - **감정 자동 감지**: 기쁨, 당황, 분노, 불안, 상처, 슬픔
    - **관계 맥락 파악**: 부모님, 친구, 형제자매, 좋아하는 사람
    - **공감 전략**: 격려, 동조, 위로, 조언
    - **성인→청소년 맥락 변환**: 직장→학교, 동료→친구
    
    ### 🚀 핵심 엔드포인트
    - `POST /api/v1/chat/teen-chat` - 메인 채팅 API
    - `POST /api/v1/chat/analyze-emotion` - 감정 분석
    - `GET /api/v1/chat/empathy-strategies` - 공감 전략 조회
    """,
    version=os.getenv("VERSION", "2.0.0"),
    docs_url="/docs",
    redoc_url="/redoc",
    contact={
        "name": "youdie006",
        "email": "youdie006@naver.com",
        "url": "https://github.com/youdie006/simsimi-ai-agent"
    },
    license_info={
        "name": "MIT License",
        "url": "https://opensource.org/licenses/MIT"
    }
)

# CORS 설정
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# 정적 파일 서빙 (웹 인터페이스용)
try:
    app.mount("/static", StaticFiles(directory="static"), name="static")
    print("✅ 정적 파일 서빙 설정 완료: /static")
except Exception as e:
    print(f"⚠️ 정적 파일 디렉토리 없음: {e}")
    print("   static/ 폴더를 생성하면 웹 인터페이스를 사용할 수 있습니다.")

# 라우터 import 및 등록
try:
    from src.api import vector, openai, chat

    # 벡터 스토어 API
    app.include_router(
        vector.router,
        prefix="/api/v1/vector",
        tags=["🗄️ Vector Store"],
        responses={
            500: {"description": "벡터 스토어 오류"},
            404: {"description": "문서를 찾을 수 없음"}
        }
    )

    # OpenAI GPT-4 API
    app.include_router(
        openai.router,
        prefix="/api/v1/openai",
        tags=["🤖 OpenAI GPT-4"],
        responses={
            500: {"description": "OpenAI API 오류"},
            429: {"description": "API 사용량 초과"}
        }
    )

    # 청소년 공감형 채팅 API (메인)
    app.include_router(
        chat.router,
        prefix="/api/v1/chat",
        tags=["💙 Teen Empathy Chat"],
        responses={
            500: {"description": "채팅 처리 오류"},
            422: {"description": "입력 데이터 검증 오류"}
        }
    )

except ImportError as e:
    print(f"⚠️ API 라우터 import 실패: {e}")
    print("   일부 기능이 제한될 수 있습니다.")

@app.get("/", response_class=HTMLResponse)
async def web_chat_interface():
    """🌐 웹 채팅 인터페이스 (메인 페이지)"""
    try:
        # static/index.html 파일 읽기
        html_file = "static/index.html"
        if os.path.exists(html_file):
            with open(html_file, 'r', encoding='utf-8') as f:
                return HTMLResponse(content=f.read())
        else:
            # 파일이 없으면 간단한 설치 안내 페이지
            return HTMLResponse(content="""
            <!DOCTYPE html>
            <html lang="ko">
            <head>
                <meta charset="UTF-8">
                <meta name="viewport" content="width=device-width, initial-scale=1.0">
                <title>청소년 공감형 AI 챗봇</title>
                <style>
                    body { font-family: Arial, sans-serif; text-align: center; padding: 50px; background: #f0f2ff; }
                    .container { max-width: 600px; margin: 0 auto; }
                    h1 { color: #667eea; }
                    .code { background: #f8f9fa; padding: 20px; border-radius: 10px; margin: 20px 0; text-align: left; }
                </style>
            </head>
            <body>
                <div class="container">
                    <h1>🧠 청소년 공감형 AI 챗봇</h1>
                    <p>웹 인터페이스를 사용하려면 다음 단계를 따라주세요:</p>
                    
                    <div class="code">
                        <strong>1단계: static 폴더 생성</strong><br>
                        mkdir static
                    </div>
                    
                    <div class="code">
                        <strong>2단계: index.html 파일 생성</strong><br>
                        위에서 제공한 HTML 코드를 static/index.html에 저장
                    </div>
                    
                    <div class="code">
                        <strong>3단계: 서버 재시작</strong><br>
                        docker-compose restart
                    </div>
                    
                    <p><strong>임시로 API 테스트:</strong> <a href="/docs">/docs</a></p>
                    <p><strong>프로젝트 정보:</strong> <a href="/api/info">/api/info</a></p>
                </div>
            </body>
            </html>
            """)
    except Exception as e:
        return HTMLResponse(content=f"<h1>오류 발생</h1><p>{str(e)}</p>")


@app.get("/chat", response_class=HTMLResponse)
async def chat_page():
    """💬 채팅 페이지 (직접 접근)"""
    return await web_chat_interface()


@app.get("/api/info")
async def api_info():
    """🏠 메인 엔드포인트 - 프로젝트 개요"""
    return {
        "message": "💙 마음이가 여러분을 기다리고 있어요!",
        "project": {
            "name": "마음이 - 청소년 상담 챗봇",
            "description": "13-19세 청소년을 위한 따뜻한 상담 친구",
            "version": os.getenv("VERSION", "2.0.0"),
            "author": os.getenv("AUTHOR", "youdie006@naver.com"),
            "repository": "https://github.com/youdie006/simsimi-ai-agent",
            "license": "MIT",
            "development_period": "2일 MVP + Term Project"
        },
        "target_audience": {
            "age_range": "13-19세 청소년",
            "use_cases": [
                "학교생활 고민 상담",
                "친구관계 문제 해결",
                "부모님과의 갈등 조율",
                "감정 정리 및 스트레스 관리",
                "진로 및 학습 동기부여"
            ],
            "approach": "진짜 공감 + 구체적 행동 제안"
        },
        "ai_capabilities": {
            "data_source": "전문 상담사 31,821개 상담 경험",
            "emotions": ["기쁨", "당황", "분노", "불안", "상처", "슬픔"],
            "relationships": ["부모님", "친구", "형제자매", "좋아하는 사람", "동급생"],
            "empathy_strategies": {
                "격려": "23% - 응원하고 동기부여",
                "동조": "33% - 함께 공감하고 이해",
                "위로": "23% - 따뜻하게 달래주기",
                "조언": "21% - 구체적 해결방안 제시"
            },
            "teen_context_conversion": "성인 대화 → 청소년 맥락 자동 변환"
        },
        "technical_architecture": {
            "ai_pattern": "ReAct (Reason + Act + Observe)",
            "llm": "OpenAI GPT-4",
            "embedding": "jhgan/ko-sbert-multitask (한국어 특화)",
            "vector_db": "ChromaDB",
            "backend": "FastAPI + Docker",
            "data_processing": "실시간 맥락 변환 + 벡터 검색"
        },
        "environment": {
            "platform": "docker",
            "python_version": f"{sys.version_info.major}.{sys.version_info.minor}.{sys.version_info.micro}",
            "debug_mode": os.getenv("DEBUG", "false").lower() == "true"
        },
        "timestamp": datetime.now().isoformat(),
        "api_endpoints": {
            "main_chat": "/api/v1/chat/teen-chat",
            "emotion_analysis": "/api/v1/chat/analyze-emotion",
            "empathy_strategies": "/api/v1/chat/empathy-strategies",
            "similar_contexts": "/api/v1/chat/search-context",
            "vector_search": "/api/v1/vector/search",
            "system_health": "/api/v1/health",
            "api_docs": "/docs"
        }
    }


@app.get("/api/v1/health")
async def health_check():
    """🏥 시스템 헬스 체크"""

    # 각 서비스별 상태 체크
    services_status = {}

    try:
        # OpenAI 서비스 체크
        openai_key = os.getenv("OPENAI_API_KEY", "")
        services_status["openai"] = bool(openai_key and "your_" not in openai_key)
    except:
        services_status["openai"] = False

    try:
        # ChromaDB 체크 (파일 존재 여부로 간단 체크)
        chromadb_path = os.getenv("CHROMADB_PATH", "./data/chromadb")
        services_status["chromadb"] = os.path.exists(chromadb_path)
    except:
        services_status["chromadb"] = False

    try:
        # AI Hub 데이터 디렉토리 체크
        aihub_dir = os.getenv("AIHUB_DATA_DIR", "./data/aihub")
        services_status["aihub_data"] = os.path.exists(aihub_dir)
    except:
        services_status["aihub_data"] = False

    # 전체 상태 결정
    overall_status = "healthy" if all(services_status.values()) else "degraded"

    return {
        "status": overall_status,
        "service": "teen-empathy-chatbot",
        "core_features": {
            "expert_knowledge": f"✅ 전문 상담사 지식 데이터베이스 {'Ready' if services_status.get('aihub_data') else 'Not Ready'}",
            "smart_analysis": "✅ 상황별 맞춤 분석 시스템 (Thought→Action→Observation→Response)",
            "emotion_detection": "✅ 6가지 감정 자동 인식 (기쁨/당황/분노/불안/상처/슬픔)",
            "similar_cases": f"✅ 비슷한 상황 찾기 {'Ready' if services_status.get('chromadb') else 'Not Ready'}",
            "teen_friendly": "✅ 청소년 친화적 조언 변환 (직장→학교, 동료→친구)",
            "counseling_methods": "✅ 4가지 상담 방법 (격려/동조/위로/조언)"
        },
        "services": {
            "docker": "✅ Running",
            "python": f"🐍 {sys.version_info.major}.{sys.version_info.minor}",
            "fastapi": "⚡ Ready",
            "openai_gpt4": f"🤖  {'Ready' if services_status.get('openai') else 'API Key Required'}",
            "chromadb": f"🗄️ {'Ready' if services_status.get('chromadb') else 'Initializing'}",
            "korean_embedding": "🇰🇷 jhgan/ko-sbert-multitask Ready"
        },
        "data_processing": {
            "language_understanding": "한국어 전문 이해 시스템",
            "knowledge_base": "전문 상담 지식 데이터베이스",
            "teen_adaptation": "청소년 눈높이 맞춤 변환",
            "empathy_mapping": "감정 → 공감전략 매핑",
            "search_algorithm": "코사인 유사도 벡터 검색"
        },
        "safety_measures": {
            "teen_protection": "✅ 청소년 데이터 보호 원칙 준수",
            "conversation_privacy": "✅ 대화 내용 익명화 처리",
            "no_personal_storage": "✅ 개인정보 저장하지 않음",
            "data_protection": "✅ 개인정보 보호 및 데이터 보안",
            "content_filtering": "✅ 부적절한 내용 필터링"
        },
        "timestamp": datetime.now().isoformat(),
        "author": "youdie006@naver.com"
    }


@app.get("/api/v1/project-info")
async def project_info():
    """📋 프로젝트 상세 정보"""
    return {
        "project_overview": {
            "title": "청소년 공감형 AI 챗봇",
            "mission": "13-19세 청소년의 고민 해결과 정서적 지원",
            "vision": "모든 청소년이 따뜻한 공감과 실질적 도움을 받을 수 있는 AI 상담사",
            "approach": "진짜 공감 + 구체적 행동 제안",
            "development_period": "2일 MVP + Term Project",
            "target_impact": "청소년 정신건강 증진 및 문제해결 역량 강화"
        },
        "technical_innovation": {
            "react_pattern": {
                "description": "단계별 추론 기반 응답 생성",
                "steps": [
                    "THOUGHT: 사용자 상황과 감정 분석",
                    "ACTION: 유사 상황 검색 + 공감 전략 선택",
                    "OBSERVATION: 검색 결과와 맥락 분석",
                    "RESPONSE: 공감 + 구체적 행동 방안 제시"
                ],
                "benefit": "단순 패턴 매칭이 아닌 논리적 추론 기반 응답"
            },
            "teen_context_conversion": {
                "purpose": "성인 중심 AI Hub 데이터를 청소년 관점으로 자동 변환",
                "conversion_examples": {
                    "workplace_to_school": "직장 스트레스 → 학교 스트레스",
                    "colleague_to_friend": "동료와의 갈등 → 친구와의 갈등",
                    "boss_to_teacher": "상사 관계 → 선생님 관계",
                    "overtime_to_study": "야근 → 야자 (야간자율학습)"
                },
                "impact": "청소년이 실제 공감할 수 있는 맥락으로 변환"
            },
            "vector_search_rag": {
                "description": "유사한 상황의 성공적 해결 사례 검색",
                "embedding_model": "jhgan/ko-sbert-multitask (한국어 특화)",
                "search_accuracy": "맥락 기반 코사인 유사도 검색",
                "benefit": "과거 성공 사례를 참고한 검증된 조언 제공"
            }
        },
        "aihub_data_utilization": {
            "total_sessions": "31,821개 공감형 대화 세션 전체 활용",
            "data_categories": {
                "emotions": {
                    "기쁨": "긍정적 상황에서의 공감과 격려",
                    "당황": "예상치 못한 상황에서의 위로와 조언",
                    "분노": "화가나는 상황에서의 공감과 해결방안",
                    "불안": "걱정되는 상황에서의 위로와 격려",
                    "상처": "마음이 아픈 상황에서의 치유적 접근",
                    "슬픔": "슬픈 상황에서의 따뜻한 동반"
                },
                "relationships": {
                    "부모자녀": "가정 내 갈등 → 부모님과의 관계",
                    "친구": "또래 관계 → 친구 관계",
                    "연인": "연애 관계 → 좋아하는 사람",
                    "형제자매": "형제 관계 → 형제자매",
                    "직장동료": "업무 관계 → 친구 관계",
                    "부부": "부부 관계 → 부모님 관계"
                }
            },
            "empathy_distribution": {
                "격려 (23%)": "응원하고 동기부여하는 접근",
                "동조 (33%)": "함께 공감하고 이해하는 접근",
                "위로 (23%)": "따뜻하게 달래주는 접근",
                "조언 (21%)": "구체적 해결방안을 제시하는 접근"
            }
        },
        "development_methodology": {
            "phase1": "✅ AI Hub 데이터 처리 모듈 구현",
            "phase2": "✅ ChromaDB 벡터 스토어 구축",
            "phase3": "✅ OpenAI GPT-4 클라이언트 개발",
            "phase4": "✅ ReAct 패턴 채팅 시스템 구현",
            "phase5": "✅ 청소년 맥락 변환 시스템 완성",
            "phase6": "🔄 테스트 및 성능 최적화 진행중",
            "phase7": "📋 사용자 피드백 수집 예정"
        },
        "future_enhancements": [
            "대화 품질 평가 시스템 구축",
            "청소년 전용 위기상황 감지 및 전문기관 연계",
            "개인화된 상담 히스토리 관리 (익명화)",
            "다국어 지원 확장",
            "음성 채팅 인터페이스 추가",
            "청소년 멘탈헬스 리포트 생성"
        ],
        "social_impact": {
            "primary_benefit": "청소년의 정서적 안정감 증진",
            "secondary_benefits": [
                "문제해결 능력 향상",
                "자존감 및 자신감 증진",
                "타인과의 소통 능력 개발",
                "스트레스 관리 능력 향상"
            ],
            "accessibility": "24시간 언제든지 접근 가능한 AI 상담사",
            "scalability": "동시에 수많은 청소년에게 개별 맞춤 상담 제공"
        },
        "author": "youdie006@naver.com",
        "last_updated": datetime.now().isoformat()
    }


@app.get("/api/v1/quick-start")
async def quick_start_guide():
    """🚀 빠른 시작 가이드"""
    return {
        "quick_start": {
            "step1": {
                "title": "🔑 API 키 설정",
                "description": "OpenAI API 키를 환경변수에 설정",
                "command": "export OPENAI_API_KEY=your_openai_api_key",
                "note": ".env 파일에 저장하는 것을 권장합니다"
            },
            "step2": {
                "title": "📊 샘플 데이터 생성",
                "description": "테스트용 AI Hub 샘플 데이터 생성 및 인덱싱",
                "endpoint": "POST /api/v1/chat/create-sample-data",
                "note": "실제 AI Hub 데이터가 없어도 테스트 가능합니다"
            },
            "step3": {
                "title": "💬 채팅 테스트",
                "description": "청소년 공감형 채팅 API 테스트",
                "endpoint": "POST /api/v1/chat/teen-chat",
                "example_request": {
                    "message": "친구가 나를 무시하는 것 같아서 기분이 나빠",
                    "include_reasoning": True
                }
            }
        },
        "test_scenarios": [
            {
                "category": "친구관계",
                "message": "친구들이 나만 빼고 단톡방을 만들어서 서운해",
                "expected_emotion": "상처",
                "expected_strategies": ["위로", "격려"]
            },
            {
                "category": "가족관계",
                "message": "부모님이 성적 때문에 계속 잔소리하셔서 스트레스 받아",
                "expected_emotion": "분노",
                "expected_strategies": ["위로", "조언"]
            },
            {
                "category": "학업",
                "message": "시험을 정말 잘 봐서 기분이 너무 좋아!",
                "expected_emotion": "기쁨",
                "expected_strategies": ["격려", "동조"]
            }
        ],
        "api_testing_tools": [
            {
                "tool": "FastAPI Docs",
                "url": "/docs",
                "description": "인터랙티브 API 문서에서 직접 테스트"
            },
            {
                "tool": "curl",
                "example": """curl -X POST "http://localhost:8000/api/v1/chat/teen-chat" \\
  -H "Content-Type: application/json" \\
  -d '{"message": "친구와 싸웠어요", "include_reasoning": true}'"""
            },
            {
                "tool": "Python requests",
                "example": """import requests
response = requests.post(
    'http://localhost:8000/api/v1/chat/teen-chat',
    json={'message': '학교에서 스트레스 받아요'}
)
print(response.json())"""
            }
        ]
    }


@app.exception_handler(Exception)
async def global_exception_handler(request, exc):
    """🚨 전역 예외 처리"""
    return JSONResponse(
        status_code=500,
        content={
            "error": "Internal Server Error",
            "message": str(exc),
            "service": "teen-empathy-chatbot",
            "timestamp": datetime.now().isoformat(),
            "contact": "youdie006@naver.com",
            "support_message": "청소년의 안전과 wellbeing을 최우선으로 합니다. 문제가 지속되면 연락해주세요.",
            "quick_fixes": [
                "OpenAI API 키가 올바르게 설정되어 있는지 확인",
                "필요한 디렉토리가 생성되어 있는지 확인 (./data/chromadb, ./data/aihub)",
                "Docker 컨테이너가 정상적으로 실행되고 있는지 확인"
            ]
        }
    )


if __name__ == "__main__":
    import uvicorn
    print("🧠 청소년 공감형 AI 챗봇 시작 중...")
    print("📊 AI Hub 31,821개 공감 대화 세션 기반")
    print("🤖 ReAct 패턴 + GPT-4 + ChromaDB")
    print("💙 13-19세 청소년 전용 공감 상담사")
    print("---")

    uvicorn.run(
        "main:app",
        host=os.getenv("API_HOST", "0.0.0.0"),
        port=int(os.getenv("API_PORT", 8000)),
        reload=os.getenv("DEBUG", "true").lower() == "true"
    )# Auto reload trigger Sat May 31 03:32:39 UTC 2025
# Context fix Sat May 31 03:47:46 UTC 2025
# Debug logs Sat May 31 03:53:09 UTC 2025

@app.get("/debug/rag-check")
async def debug_rag_check():
    """🔍 RAG 시스템 상태 확인"""
    try:
        from src.core.vector_store import get_vector_store
        from src.services.aihub_processor import get_teen_empathy_processor
        
        vector_store = await get_vector_store()
        processor = await get_teen_empathy_processor()
        
        # 1. 컬렉션 통계
        stats = await vector_store.get_collection_stats()
        
        # 2. 샘플 검색 테스트
        test_query = "엄마가 용돈을 안줘서 화가 나"
        search_results = await vector_store.search(test_query, top_k=3)
        
        # 3. 전문가 응답 검색 테스트
        expert_results = await processor.search_similar_contexts(
            query=test_query,
            emotion="분노",
            relationship="부모님",
            top_k=3
        )
        
        return {
            "vector_store_stats": {
                "total_documents": stats.total_documents,
                "collection_name": stats.collection_name,
                "status": stats.status
            },
            "raw_search_results": [
                {
                    "content": r.content[:100] + "...",
                    "score": r.score,
                    "metadata": r.metadata
                } for r in search_results
            ],
            "expert_search_results": [
                {
                    "user_utterance": r.get("user_utterance", "")[:50] + "...",
                    "system_response": r.get("system_response", "")[:100] + "...",
                    "similarity_score": r.get("similarity_score", 0),
                    "emotion": r.get("emotion", ""),
                    "empathy_label": r.get("empathy_label", "")
                } for r in expert_results
            ],
            "diagnosis": {
                "vector_search_working": len(search_results) > 0,
                "expert_search_working": len(expert_results) > 0,
                "high_quality_results": len([r for r in expert_results if r.get("similarity_score", 0) > 0.7]) > 0,
                "usable_results": len([r for r in expert_results if r.get("similarity_score", 0) > 0.3]) > 0
            }
        }
        
    except Exception as e:
        return {"error": str(e), "debug_info": "RAG 시스템 체크 실패"}

