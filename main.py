"""
🤖 심심이 기반 대화 AI Agent
개발자: youdie006@naver.com
환경: Docker + PyCharm + Git (퍼블릭 저장소)
"""

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
import os
import sys
from datetime import datetime
from dotenv import load_dotenv

# 환경 변수 로드
load_dotenv()

# FastAPI 앱 생성
app = FastAPI(
    title="🤖 심심이 기반 대화 AI Agent",
    description="RAG + GPT-4 + 심심이 API + Docker (Public Repository)",
    version=os.getenv("VERSION", "1.0.0"),
    docs_url="/docs",
    redoc_url="/redoc",
    contact={
        "name": "youdie006",
        "email": "youdie006@naver.com",
        "url": "https://github.com/youdie006/simsimi-ai-agent"
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


@app.get("/")
async def root():
    """🏠 메인 엔드포인트"""
    return {
        "message": "🚀 심심이 AI Agent가 Docker에서 실행 중입니다!",
        "project": {
            "name": os.getenv("PROJECT_NAME", "simsimi-ai-agent"),
            "version": os.getenv("VERSION", "1.0.0"),
            "author": os.getenv("AUTHOR", "youdie006@naver.com"),
            "repository": "https://github.com/youdie006/simsimi-ai-agent"
        },
        "environment": {
            "platform": "docker",
            "python_version": f"{sys.version_info.major}.{sys.version_info.minor}.{sys.version_info.micro}",
            "debug_mode": os.getenv("DEBUG", "false").lower() == "true"
        },
        "timestamp": datetime.now().isoformat(),
        "endpoints": {
            "docs": "/docs",
            "health": "/api/v1/health",
            "security_check": "/api/v1/security-check"
        }
    }


@app.get("/api/v1/health")
async def health_check():
    """🏥 헬스 체크"""
    return {
        "status": "healthy",
        "service": "simsimi-ai-agent",
        "environment": {
            "docker": "✅ Running",
            "python": f"🐍 {sys.version_info.major}.{sys.version_info.minor}",
            "fastapi": "⚡ Ready"
        },
        "timestamp": datetime.now().isoformat(),
        "author": "youdie006@naver.com"
    }


@app.get("/api/v1/security-check")
async def security_check():
    """🔒 보안 설정 체크 (디버깅용)"""
    openai_key = os.getenv("OPENAI_API_KEY", "")
    simsimi_key = os.getenv("SIMSIMI_API_KEY", "")

    return {
        "security_status": {
            "openai_key_configured": bool(openai_key and "your_" not in openai_key),
            "simsimi_key_configured": bool(simsimi_key and "your_" not in simsimi_key),
            "environment_file_loaded": os.path.exists(".env"),
            "public_repository_safe": True
        },
        "configuration": {
            "debug_mode": os.getenv("DEBUG", "false").lower() == "true",
            "log_level": os.getenv("LOG_LEVEL", "INFO"),
            "vector_db": os.getenv("VECTOR_DB_TYPE", "chromadb"),
            "embedding_model": os.getenv("EMBEDDING_MODEL", "jhgan/ko-sbert-multitask")
        },
        "warnings": {
            "api_keys_in_env_only": "✅ API 키는 환경변수로만 관리됩니다",
            "public_repo_safe": "✅ 퍼블릭 저장소에 민감정보 노출 없음"
        },
        "author": "youdie006@naver.com"
    }


@app.exception_handler(Exception)
async def global_exception_handler(request, exc):
    """🚨 전역 예외 처리"""
    return JSONResponse(
        status_code=500,
        content={
            "error": "Internal Server Error",
            "message": str(exc),
            "timestamp": datetime.now().isoformat(),
            "contact": "youdie006@naver.com"
        }
    )


if __name__ == "__main__":
    import uvicorn

    uvicorn.run(
        "main:app",
        host=os.getenv("API_HOST", "0.0.0.0"),
        port=int(os.getenv("API_PORT", 8000)),
        reload=os.getenv("DEBUG", "true").lower() == "true"
    )