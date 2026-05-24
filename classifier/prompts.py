import json

SYSTEM_PROMPT = """Eres un asistente de evaluación clínica para ADIPA,
plataforma de formación en salud mental.

Tu tarea es clasificar turnos del OPERADOR (psicólogo en formación)
dentro de sesiones de Primeros Auxilios Psicológicos (PAP).

FASES ABCDE del protocolo PAP:
  A — Escucha Activa / Recepción
      Primer contacto, establecer seguridad básica, obtener nombre
      y ubicación, verificar si hay heridas físicas o peligro inmediato.

  B — Regulación Emocional
      Técnicas de respiración, anclaje sensorial (5-4-3-2-1),
      reducir hiperventilación, bajar activación fisiológica.

  C — Categorización de Necesidades
      Inventariar necesidades urgentes (seguridad, salud, red de apoyo),
      ordenar prioridades en voz alta, evaluar riesgo.

  D — Derivación / Red de Apoyo
      Activar personas de confianza, coordinar ayuda presencial,
      contactar servicios de emergencia, articular red sin cortar llamada.

  E — Psicoeducación / Esperanza
      Normalizar reacciones agudas, explicar qué puede ocurrir
      en las próximas horas y días, plan de seguridad, cierre.

ACTOS VERBALES (multi-etiqueta, uno o más por turno):
  validacion         — valida emoción o experiencia del paciente
  pregunta_abierta   — pregunta que invita a desarrollar ("¿Cómo se siente?")
  pregunta_cerrada   — pregunta de sí/no o dato concreto ("¿Está herida?")
  reflejo            — devuelve/parafrasea lo que dijo el paciente
  interpretacion     — propone una lectura del estado interno
  directivo          — da instrucción concreta de acción
  silencio_contencion— acompaña sin exigir, sostiene la pausa
  confrontacion      — señala una contradicción o tensión con cuidado
  otro               — si ninguno aplica claramente

REGLAS:
1. Responde ÚNICAMENTE con JSON válido, sin texto adicional.
2. El campo "fase" es exactamente una letra: A, B, C, D o E.
3. El campo "actos_verbales" es una lista de strings (mínimo uno).
4. El campo "confianza" es un número entre 0.0 y 1.0.
5. Si el contexto del paciente ayuda, úsalo. Si no hay contexto, clasifica
   solo con el texto del operador.
"""

# 2 ejemplos por fase: variedad de actos verbales, incluye multi-etiqueta
# y casos borderline (B+D solapados, C con pregunta, etc.)
FEW_SHOT_EXAMPLES = [
    {
        "contexto_paciente": "",
        "operador_texto": "Línea de apoyo psicológico, habla Matías. ¿Con quién hablo?",
        "fase": "A",
        "actos_verbales": ["pregunta_cerrada"],
        "confianza": 0.98,
    },
    {
        "contexto_paciente": "No puedo… no puedo sacármelo de la cabeza.",
        "operador_texto": (
            "Estoy aquí. Vamos de a poco. Primero dígame su nombre, por favor."
        ),
        "fase": "A",
        "actos_verbales": ["silencio_contencion", "directivo"],
        "confianza": 0.95,
    },
    {
        "contexto_paciente": "Es que la veo. Cierro los ojos y la veo.",
        "operador_texto": (
            "Entonces no cierre los ojos. Manténgalos abiertos. "
            "Busque un punto fijo en la pieza: una esquina, una mesa, "
            "algo que no se mueva."
        ),
        "fase": "B",
        "actos_verbales": ["directivo"],
        "confianza": 0.97,
    },
    {
        "contexto_paciente": "No puedo respirar bien. Me tiemblan las manos. Veo la pistola.",
        "operador_texto": (
            "Camila, antes de seguir con puertas o llamadas, quiero quedarme "
            "un momento con usted. Su cuerpo todavía está allá, frente a la "
            "pistola, aunque usted ya esté en su departamento."
        ),
        "fase": "B",
        "actos_verbales": ["validacion", "interpretacion"],
        "confianza": 0.93,
    },
    {
        "contexto_paciente": "Sí. Y del edificio creo.",
        "operador_texto": (
            "Bien. Entonces las necesidades urgentes son: seguridad de la puerta, "
            "aviso a conserjería, compañía presencial, bloqueo de cuentas y denuncia, "
            "pero en ese orden. No todo al mismo tiempo."
        ),
        "fase": "C",
        "actos_verbales": ["directivo"],
        "confianza": 0.96,
    },
    {
        "contexto_paciente": "Sí. Por eso van a venir. Tienen mi dirección. Tienen mis llaves.",
        "operador_texto": (
            "Ese miedo tiene una parte concreta. No lo vamos a tratar solo como "
            "'nervios'. Vamos a tomar medidas de seguridad. ¿Las llaves que le "
            "quitaron son de este departamento?"
        ),
        "fase": "C",
        "actos_verbales": ["validacion", "pregunta_cerrada"],
        "confianza": 0.91,
    },
    {
        "contexto_paciente": "Nueve ocho siete… cinco cuatro… dos… uno… no, no.",
        "operador_texto": (
            "Camila, paremos. Mire la pared. No necesitamos hacerlo perfecto. "
            "¿Hay otra opción? ¿Su hermana sabe que algo pasó? ¿El conserje tiene "
            "registro de contacto de emergencia?"
        ),
        "fase": "D",
        "actos_verbales": ["pregunta_abierta", "pregunta_cerrada"],
        "confianza": 0.90,
    },
    {
        "contexto_paciente": "Me escuchó. Ana me escuchó.",
        "operador_texto": (
            "Bien. Ahora hay alguien afuera activando ayuda. "
            "Usted no está sola con esto."
        ),
        "fase": "D",
        "actos_verbales": ["validacion", "silencio_contencion"],
        "confianza": 0.94,
    },
    {
        "contexto_paciente": "Siento eso. Como si todavía estuviera ahí.",
        "operador_texto": (
            "Puede aparecer temblor, llanto, náuseas, palpitaciones, dificultad "
            "para respirar, sensación de irrealidad, sobresaltos, rabia, culpa, "
            "ganas de arrancar o imágenes repetidas de la pistola."
        ),
        "fase": "E",
        "actos_verbales": ["otro"],
        "confianza": 0.97,
    },
    {
        "contexto_paciente": "¿Se me va a pasar?",
        "operador_texto": (
            "En muchas personas la intensidad baja con seguridad, apoyo y descanso. "
            "Pero si en los próximos días no puede dormir, no puede salir, siente que "
            "está en peligro todo el tiempo, o aparecen ideas de hacerse daño, "
            "pida apoyo profesional. No espere a estar al límite."
        ),
        "fase": "E",
        "actos_verbales": ["validacion", "directivo"],
        "confianza": 0.95,
    },
]


def build_few_shot_messages() -> list[dict]:
    """Builds the few-shot message list for the Anthropic API (alternating user/assistant)."""
    messages = []
    for ex in FEW_SHOT_EXAMPLES:
        user_content = _build_user_content(ex["operador_texto"], ex["contexto_paciente"])
        assistant_content = _build_assistant_content(
            ex["fase"], ex["actos_verbales"], ex["confianza"]
        )
        messages.append({"role": "user", "content": user_content})
        messages.append({"role": "assistant", "content": assistant_content})
    return messages


def build_inference_message(operator_text: str, patient_context: str = "") -> dict:
    return {
        "role": "user",
        "content": _build_user_content(operator_text, patient_context),
    }


def _build_user_content(operator_text: str, patient_context: str) -> str:
    parts = []
    if patient_context and patient_context.strip():
        parts.append(f"CONTEXTO PACIENTE: {patient_context.strip()}")
    parts.append(f"TURNO OPERADOR: {operator_text.strip()}")
    parts.append(
        '\nResponde SOLO con JSON válido con esta estructura exacta:\n'
        '{"fase": "X", "actos_verbales": ["..."], "confianza": 0.0}'
    )
    return "\n".join(parts)


def _build_assistant_content(fase: str, actos: list[str], confianza: float) -> str:
    return json.dumps(
        {"fase": fase, "actos_verbales": actos, "confianza": confianza},
        ensure_ascii=False,
    )
