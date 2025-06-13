# Dockerfile - 의존성 충돌 및 캐시 문제 완전 해결

FROM python:3.10-slim

# 메타데이터
LABEL maintainer="youdie006@naver.com"
LABEL description="SimSimi AI Agent - Safe Build"
LABEL version="1.0.4"

# 시스템 의존성 설치
RUN apt-get update && apt-get install -y \
    gcc \
    g++ \
    curl \
    git \
    git-lfs \
    && git lfs install \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# 🛡️ 캐시 및 환경 변수 설정 (캐시 마이그레이션 완전 방지)
ENV HF_HOME=/app/cache
ENV HF_DATASETS_CACHE=/app/cache
ENV TRANSFORMERS_CACHE=/app/cache
ENV HF_HUB_CACHE=/app/cache
ENV PYTHONPATH=/app
ENV PYTHONDONTWRITEBYTECODE=1
ENV PYTHONUNBUFFERED=1

# 🔧 캐시 마이그레이션 및 경고 완전 비활성화
ENV HF_HUB_DISABLE_PROGRESS_BARS=1
ENV TRANSFORMERS_NO_ADVISORY_WARNINGS=1
ENV TOKENIZERS_PARALLELISM=false
ENV HF_HUB_DISABLE_TELEMETRY=1

# 캐시 디렉토리 생성 및 권한 설정
RUN mkdir -p /app/cache /app/data /app/logs /app/static && \
    chmod -R 777 /app/cache /app/data /app/logs

# 🔧 미리 캐시 구조 생성 (마이그레이션 트리거 방지)
RUN mkdir -p /app/cache/hub /app/cache/datasets /app/cache/transformers && \
    mkdir -p /app/cache/models /app/cache/tokenizers && \
    touch /app/cache/.no_migration_needed

# Python 의존성 설치
COPY requirements.txt .
RUN pip install --upgrade pip
RUN pip install --no-cache-dir -r requirements.txt

# 애플리케이션 코드 복사
COPY . .

# 🛡️ 안전한 데이터 다운로드 (타임아웃 및 실패 허용)
RUN timeout 180 huggingface-cli download \
    youdie006/simsimi-ai-agent-data \
    --repo-type dataset \
    --local-dir /app/data \
    --local-dir-use-symlinks False || \
    echo "⚠️ 데이터 다운로드 건너뜀 - 런타임에 처리"

# 🔧 안전한 모델 사전 다운로드 (실패해도 계속 진행)
RUN echo "📥 임베딩 모델 사전 다운로드 시도..." && \
    timeout 300 python -c "\
import os; \
os.environ['TRANSFORMERS_CACHE'] = '/app/cache'; \
os.environ['HF_HOME'] = '/app/cache'; \
os.environ['TRANSFORMERS_NO_ADVISORY_WARNINGS'] = '1'; \
try: \
    print('모델 다운로드 시작...'); \
    from sentence_transformers import SentenceTransformer; \
    model = SentenceTransformer('jhgan/ko-sbert-multitask', cache_folder='/app/cache'); \
    print(f'✅ 모델 다운로드 완료 - 차원: {model.get_sentence_embedding_dimension()}'); \
except Exception as e: \
    print(f'⚠️ 모델 사전 다운로드 실패: {e}'); \
    print('런타임에 다시 시도합니다.'); \
" || echo "모델 사전 다운로드 실패 - 런타임에 처리"

# 스마트 시작 스크립트
COPY start.sh /app/start.sh
RUN chmod +x /app/start.sh

# 포트 노출
EXPOSE 7860

# 관대한 헬스체크
HEALTHCHECK --interval=60s --timeout=30s --start-period=600s --retries=5 \
    CMD curl -f http://localhost:7860/api/v1/health || exit 1

# 시작
CMD ["/app/start.sh"]