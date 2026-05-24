# ── Base ──────────────────────────────────────────────────────────────────────
# Python 3.12 slim — balance entre tamaño y compatibilidad
FROM python:3.12-slim

# ── Metadatos ─────────────────────────────────────────────────────────────────
LABEL maintainer="ADIPA Lab"
LABEL description="PAP Turn Classifier — FastAPI service"

# ── Variables de entorno ──────────────────────────────────────────────────────
# ANTHROPIC_API_KEY se inyecta en runtime (no se hardcodea aquí)
ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PORT=8000

# ── Directorio de trabajo ─────────────────────────────────────────────────────
WORKDIR /app

# ── Dependencias del sistema (mínimas) ───────────────────────────────────────
RUN apt-get update && apt-get install -y --no-install-recommends \
    curl \
    && rm -rf /var/lib/apt/lists/*

# ── Dependencias Python ───────────────────────────────────────────────────────
# Copiamos requirements primero para aprovechar la caché de capas de Docker.
# Si el código cambia pero requirements no, no se reinstala todo.
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# ── Código fuente ─────────────────────────────────────────────────────────────
COPY classifier/ ./classifier/
COPY api/ ./api/

# ── Usuario no-root (buena práctica de seguridad) ─────────────────────────────
RUN useradd --create-home --shell /bin/bash appuser
USER appuser

# ── Puerto expuesto ───────────────────────────────────────────────────────────
EXPOSE 8000

# ── Health check nativo de Docker ─────────────────────────────────────────────
HEALTHCHECK --interval=30s --timeout=10s --start-period=5s --retries=3 \
    CMD curl -f http://localhost:8000/health || exit 1

# ── Arranque ──────────────────────────────────────────────────────────────────
# --workers 1: clasificador tiene caché en memoria por proceso;
#              para escalar horizontalmente usar múltiples contenedores.
# --timeout-keep-alive 30: sesiones de voz pueden tener pausas largas.
CMD ["uvicorn", "api.main:app", \
     "--host", "0.0.0.0", \
     "--port", "8000", \
     "--workers", "1", \
     "--timeout-keep-alive", "30", \
     "--log-level", "info"]
