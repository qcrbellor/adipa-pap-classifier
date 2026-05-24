# ADIPA PAP Classifier

Clasificador de turnos del operador en sesiones de **Primeros Auxilios Psicológicos (PAP)**.
Predice la **fase ABCDE** y los **actos verbales** de cada turno del psicólogo.

Construido para la prueba técnica de Machine Learning Engineer — ADIPA Lab.

---

## Levantar el servicio en un comando

```bash
docker build -t adipa-pap-classifier .
docker run -e ANTHROPIC_API_KEY=sk-ant-... -p 8000:8000 adipa-pap-classifier
```

El servicio queda disponible en `http://localhost:8000`.

---

## Endpoints

| Método | Ruta        | Descripción                          |
|--------|-------------|--------------------------------------|
| GET    | `/health`   | Health check y estado del LLM        |
| POST   | `/classify` | Clasifica un turno del operador      |
| GET    | `/docs`     | Documentación interactiva (Swagger)  |

---

## Ejemplos de uso

### Health check

```bash
curl http://localhost:8000/health
```

```json
{
  "status": "ok",
  "model": "claude-haiku-4-5-20251001",
  "llm_configured": true,
  "version": "0.1.0"
}
```

---

### Clasificar un turno — Fase B (Regulación)

```bash
curl -X POST http://localhost:8000/classify \
  -H "Content-Type: application/json" \
  -d '{
    "operador_texto": "Inhale contando cuatro: uno, dos, tres, cuatro. Mantenga. Suelte el aire lento.",
    "contexto_paciente_previo": "No puedo respirar. Me tiemblan las manos."
  }'
```

```json
{
  "fase": "B",
  "actos_verbales": ["directivo"],
  "confianza": 0.97,
  "llm_available": true,
  "fase_descripcion": "B — Regulación Emocional"
}
```

---

### Clasificar un turno — Fase C (Categorización)

```bash
curl -X POST http://localhost:8000/classify \
  -H "Content-Type: application/json" \
  -d '{
    "operador_texto": "Voy a ordenar lo urgente. Uno: la puerta está cerrada. Dos: necesitamos activar a una persona de confianza. ¿Tiene el número de su hermana?",
    "contexto_paciente_previo": "Tienen mis llaves. Tienen mi dirección."
  }'
```

```json
{
  "fase": "C",
  "actos_verbales": ["directivo", "pregunta_cerrada"],
  "confianza": 0.91,
  "llm_available": true,
  "fase_descripcion": "C — Categorización de Necesidades"
}
```

---

### Input inválido → 422 (nunca 500)

```bash
curl -X POST http://localhost:8000/classify \
  -H "Content-Type: application/json" \
  -d '{"operador_texto": "   "}'
```

```json
{
  "detail": [
    {
      "type": "value_error",
      "loc": ["body", "operador_texto"],
      "msg": "Value error, operador_texto no puede ser solo espacios en blanco."
    }
  ]
}
```

---

## Desarrollo local (sin Docker)

```bash
pip install -r requirements.txt
export ANTHROPIC_API_KEY=sk-ant-...
uvicorn api.main:app --reload --port 8000
```

---

## Construir el dataset desde el .docx original

```bash
pip install python-docx
python data/build_dataset.py Prueba_Tecnica_ADIPA.docx
```

Genera `data/dataset.csv` y `data/dataset_splits.json`.

---

## Evaluar el clasificador

```bash
export ANTHROPIC_API_KEY=sk-ant-...
python evaluation/evaluate.py
```

Genera reporte con precision/recall/F1 por fase, matriz de confusión
y ejemplos de errores — separado por split intra-train y held-out.

---

## Estructura del proyecto

```
adipa-pap-classifier/
├── data/
│   ├── build_dataset.py      # extrae turnos del .docx → dataset.csv
│   ├── dataset.csv           # dataset construido (1284 turnos)
│   └── dataset_splits.json   # resumen del split por caso
├── classifier/
│   ├── classifier.py         # PAPClassifier — LLM few-shot + fallback
│   ├── prompts.py            # system prompt + 10 ejemplos few-shot
│   └── __init__.py
├── api/
│   ├── main.py               # FastAPI app — /health + /classify
│   ├── schemas.py            # contratos Pydantic request/response
│   └── __init__.py
├── evaluation/
│   └── evaluate.py           # arnés de evaluación reproducible
├── writeup/
│   └── writeup.md            # análisis y preguntas de juicio (2 páginas)
├── Dockerfile
├── requirements.txt
└── README.md
```

---

## Fases del protocolo PAP

| Fase | Nombre                        | Descripción                                               |
|------|-------------------------------|-----------------------------------------------------------|
| A    | Escucha Activa / Recepción    | Primer contacto, seguridad básica, verificar lesiones     |
| B    | Regulación Emocional          | Respiración, anclaje sensorial, bajar activación          |
| C    | Categorización de Necesidades | Inventariar urgencias, evaluar riesgo, ordenar prioridades|
| D    | Derivación / Red de Apoyo     | Activar personas de confianza, coordinar ayuda presencial |
| E    | Psicoeducación / Esperanza    | Normalizar reacciones, plan de seguridad, cierre          |

---

## Decisiones técnicas clave

**¿Por qué LLM few-shot y no modelo entrenado?**
Con ~800 turnos de entrenamiento (8 casos) y etiquetas débiles derivadas de
headers estructurales, el riesgo de overfitting es alto. Un LLM preentrenado
con comprensión semántica clínica generaliza mejor con pocos datos.
La señal para migrar a fine-tuning es acumular >5.000 turnos con etiquetas
de calidad revisadas por clínicos.

**¿Por qué `claude-haiku-4-5-20251001` y no Sonnet?**
Haiku ofrece ~200ms de latencia vs ~800ms de Sonnet. El objetivo de producción
es ~900ms por turno en sesión de voz — Haiku lo cumple holgadamente con menor
costo por request.

**¿Qué le falta para producción real?**
Observabilidad (trazas por turno, métricas de latencia y drift),
autenticación en el endpoint, rate limiting, CI/CD con evaluación automática
al merge, y monitoreo de calidad con revisión humana en muestra aleatoria.
Ver `writeup/writeup.md` para análisis completo.
