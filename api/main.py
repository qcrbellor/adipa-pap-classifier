import os
import logging
from contextlib import asynccontextmanager

from fastapi import FastAPI, HTTPException, Request
from fastapi.responses import JSONResponse
from fastapi.middleware.cors import CORSMiddleware

from classifier import PAPClassifier
from api.schemas import ClassifyRequest, ClassifyResponse, HealthResponse, ErrorResponse

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)-8s  %(name)s  %(message)s",
)
logger = logging.getLogger(__name__)

VERSION = "0.1.0"
MODEL = "claude-haiku-4-5-20251001"

FASE_DESCRIPCIONES = {
    "A": "A — Escucha Activa / Recepción",
    "B": "B — Regulación Emocional",
    "C": "C — Categorización de Necesidades",
    "D": "D — Derivación / Red de Apoyo",
    "E": "E — Psicoeducación / Esperanza",
}

_classifier: PAPClassifier | None = None


@asynccontextmanager
async def lifespan(app: FastAPI):
    global _classifier
    api_key = os.getenv("ANTHROPIC_API_KEY", "")
    _classifier = PAPClassifier(api_key=api_key, use_cache=True)
    configured = bool(api_key)
    logger.info(f"PAPClassifier inicializado | LLM configurado: {configured} | versión: {VERSION}")
    yield
    logger.info("Servicio detenido.")


app = FastAPI(
    title="ADIPA PAP Classifier",
    description=(
        "Clasificador de turnos del operador en sesiones de "
        "Primeros Auxilios Psicológicos (PAP). "
        "Predice la fase ABCDE y los actos verbales de cada turno."
    ),
    version=VERSION,
    lifespan=lifespan,
    docs_url="/docs",
    redoc_url="/redoc",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["GET", "POST"],
    allow_headers=["*"],
)


@app.exception_handler(Exception)
async def generic_exception_handler(request: Request, exc: Exception):
    logger.error(f"Unhandled exception on {request.url}: {exc}", exc_info=True)
    return JSONResponse(
        status_code=500,
        content={"detail": "Error interno del servidor. Por favor intente de nuevo."},
    )


@app.get(
    "/health",
    response_model=HealthResponse,
    summary="Health check",
    description="Verifica que el servicio está activo y muestra la configuración.",
    tags=["infraestructura"],
)
async def health():
    api_key = os.getenv("ANTHROPIC_API_KEY", "")
    return HealthResponse(
        status="ok",
        model=MODEL,
        llm_configured=bool(api_key),
        version=VERSION,
    )


@app.post(
    "/classify",
    response_model=ClassifyResponse,
    responses={
        200: {"description": "Clasificación exitosa."},
        422: {
            "description": "Input inválido (texto vacío o demasiado largo).",
            "model": ErrorResponse,
        },
        500: {
            "description": "Error interno del servidor.",
            "model": ErrorResponse,
        },
    },
    summary="Clasificar turno del operador",
    description=(
        "Recibe un turno del operador (y opcionalmente el contexto del paciente) "
        "y devuelve la fase ABCDE y los actos verbales con su confianza."
    ),
    tags=["clasificación"],
)
async def classify(request: ClassifyRequest) -> ClassifyResponse:
    if _classifier is None:
        raise HTTPException(status_code=503, detail="Clasificador no inicializado.")

    logger.info(
        f"classify | texto={request.operador_texto[:60]!r}... "
        f"contexto={bool(request.contexto_paciente_previo)}"
    )

    result = _classifier.classify(
        operator_text=request.operador_texto,
        patient_context=request.contexto_paciente_previo,
    )

    return ClassifyResponse(
        fase=result.fase,
        actos_verbales=result.actos_verbales,
        confianza=result.confianza,
        llm_available=result.llm_available,
        fase_descripcion=FASE_DESCRIPCIONES[result.fase],
    )


if __name__ == "__main__":
    import uvicorn
    uvicorn.run("api.main:app", host="0.0.0.0", port=8000, reload=True, log_level="info")
