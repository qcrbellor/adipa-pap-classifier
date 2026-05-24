from pydantic import BaseModel, Field, field_validator
from typing import Literal

FasePAP = Literal["A", "B", "C", "D", "E"]

ActoVerbal = Literal[
    "validacion",
    "pregunta_abierta",
    "pregunta_cerrada",
    "reflejo",
    "interpretacion",
    "directivo",
    "silencio_contencion",
    "confrontacion",
    "otro",
]


class ClassifyRequest(BaseModel):
    operador_texto: str = Field(
        ...,
        min_length=1,
        max_length=2000,
        description="Texto del turno del operador a clasificar.",
        examples=["Inhale contando cuatro: uno… dos… tres… cuatro. "
                  "Mantenga: uno… dos… tres… cuatro. "
                  "Suelte: uno… dos… tres… cuatro."],
    )
    contexto_paciente_previo: str = Field(
        default="",
        max_length=1000,
        description="Último turno del paciente (contexto opcional).",
        examples=["No puedo respirar. Me tiemblan las manos."],
    )

    @field_validator("operador_texto")
    @classmethod
    def no_solo_espacios(cls, v: str) -> str:
        if not v.strip():
            raise ValueError("operador_texto no puede ser solo espacios en blanco.")
        return v.strip()

    @field_validator("contexto_paciente_previo")
    @classmethod
    def limpiar_contexto(cls, v: str) -> str:
        return v.strip()

    model_config = {
        "json_schema_extra": {
            "example": {
                "operador_texto": (
                    "Inhale contando cuatro: uno… dos… tres… cuatro. "
                    "Mantenga. Suelte el aire lento."
                ),
                "contexto_paciente_previo": "No puedo respirar. Me tiemblan las manos.",
            }
        }
    }


class ClassifyResponse(BaseModel):
    fase: FasePAP = Field(
        ...,
        description="Fase del protocolo PAP: A=Escucha, B=Regulación, "
                    "C=Categorización, D=Derivación, E=Psicoeducación.",
        examples=["B"],
    )
    actos_verbales: list[ActoVerbal] = Field(
        ...,
        description="Actos verbales detectados en el turno (multi-etiqueta).",
        examples=[["directivo", "validacion"]],
    )
    confianza: float = Field(
        ...,
        ge=0.0,
        le=1.0,
        description="Confianza del clasificador entre 0.0 y 1.0. "
                    "0.0 indica fallback (LLM no disponible).",
        examples=[0.95],
    )
    llm_available: bool = Field(
        ...,
        description="True si la respuesta viene del LLM. "
                    "False si se usó el fallback determinístico.",
        examples=[True],
    )
    fase_descripcion: str = Field(
        ...,
        description="Descripción legible de la fase clasificada.",
        examples=["B — Regulación Emocional"],
    )

    model_config = {
        "json_schema_extra": {
            "example": {
                "fase": "B",
                "actos_verbales": ["directivo", "validacion"],
                "confianza": 0.95,
                "llm_available": True,
                "fase_descripcion": "B — Regulación Emocional",
            }
        }
    }


class HealthResponse(BaseModel):
    status: Literal["ok"] = "ok"
    model: str = Field(description="Modelo LLM configurado.")
    llm_configured: bool = Field(description="True si hay API key configurada.")
    version: str = Field(description="Versión del servicio.")


class ErrorResponse(BaseModel):
    detail: str = Field(description="Descripción del error.")
