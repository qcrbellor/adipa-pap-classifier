# ADIPA PAP Classifier: Writeup

## 1. LLM few-shot vs modelo entrenado: decisión y señal de migración

Con 8 casos de entrenamiento y etiquetas derivadas de headers estructurales,
elegí **LLM few-shot** por tres razones concretas.

**Datos insuficientes para fine-tuning confiable.** Con aproximadamente 1.000 turnos de train
repartidos en 5 clases desbalanceadas (fase E concentra el 40% del dataset), un
encoder fine-tuneado tiende a colapsar hacia la mayoría. Fase D, la más crítica
para activar redes de apoyo, tiene apenas 94 ejemplos en train. En ese rango,
el fine-tuning está aprendiendo el estilo de redacción del guion, no la semántica
clínica.

**Etiquetas débiles.** Las fases se asignan por posición en el guion (heredadas del
último encabezado ABCDE visible), no por anotación humana turno a turno. Un modelo
supervisado que aprenda esa señal generaliza mal a turnos reales de alumnos, donde
la secuencia de fases no sigue el orden limpio de los guiones formativos.

**El LLM ya conoce el dominio.** Un modelo preentrenado entiende "inhale contando
cuatro" o "¿ha pensado en hacerse daño?" sin ejemplos explícitos. Los 10 pares
few-shot anclan el formato de salida y calibran el vocabulario de etiquetas.
El clasificador muestra un gap train, held-out pequeño o negativo, lo que indica que
no está sobreajustando al estilo de redacción de los guiones de entrenamiento.

**Señal para migrar a fine-tuning:** mayor a 5.000 turnos con etiquetas revisadas por
clínicos, o latencia sostenida por encima de 900 ms. En ese punto, LoRA sobre
`xlm-roberta-base` es el camino natural: fine-tuning eficiente, multilingüe, con
menos de 1M parámetros entrenables. `PAPClassifier` es intercambiable sin tocar
la API ni el arnés de evaluación.

---

## 2. Detección de riesgo: ideación suicida, autolesión, daño a terceros

La detección de riesgo **no es un clasificador más**, es una capa con consecuencias
asimétricas. En clasificación de fases, un error es pedagógicamente costoso pero
recuperable. En riesgo, un **falso negativo** puede tener consecuencias irreversibles.

**Métrica prioritaria: recall, no F1.** El objetivo no es equilibrar precisión y
recall; es maximizar recall sobre la clase positiva aceptando una tasa de FP elevada.
Fijaría un umbral bajo (score > 0,3, alerta), lo que infla FP pero protege contra FN.
La curva PR importa más que ROC porque la clase positiva es minoritaria y costosa.

**Revisión humana obligatoria**, no automatización. El sistema tendría tres
componentes: (1) detector automático como señal temprana, LLM con prompt específico
o clasificador binario sobre frases de riesgo validadas clínicamente; (2) cola de
revisión donde cada turno flaggeado pasa por un clínico antes de que la evaluación
llegue al alumno; (3) trazabilidad completa, texto, score, revisor y decisión final
para auditoría y mejora del modelo. El rol de la IA es reducir la carga de revisión,
no reemplazarla.

---

## 3. Latencia en vivo (~900 ms/turno): LLM externo vs modelo local

`claude-haiku-4-5-20251001` con temperature = 0 y max_tokens = 120 tiene latencia
~150–300 ms por request. Con overhead de FastAPI, el p50 estimado es ~350 ms y
el p95 ~700 ms, dentro del objetivo de 900 ms.

| Dimensión | LLM externo (actual) | Modelo local fine-tuned |
|---|---|---|
| Latencia p50 | ~350 ms | ~30–80 ms CPU / ~10 ms GPU |
| Latencia p95 | ~700 ms | ~120 ms CPU |
| Costo/turno | ~$0,0003 USD | Fijo (infra) + entrenamiento |
| Calidad con pocos datos | Alta | Baja sin >5k ejemplos |
| Disponibilidad | Depende de Anthropic | 100% controlada |

Con el volumen actual, el LLM externo es la decisión correcta. El costo por sesión
(~25 turnos) es ~$0,007 USD, negligible comparado con el costo de entrenar y
mantener un modelo local que todavía no tiene datos suficientes para superarlo.

**Qué le falta para producción real:**

*Observabilidad.* Trazas por turno (latencia, tokens, confianza), métricas de data
drift sobre la distribución de fases predichas, alertas si la tasa de fallback supera
un umbral. Sin esto, la degradación silenciosa es invisible.

*Autenticación y rate limiting.* El endpoint `/classify` está abierto. En producción
necesita API key o JWT más límites por usuario para evitar abuso.

*CI/CD con evaluación automática.* Cada cambio en el prompt debe correr el arnés
sobre el held-out antes de llegar a producción. Un PR que baje el macro-F1 más
de 0,03 no se mergea automáticamente.

*Contexto de sesión.* El clasificador es stateless: cada turno se clasifica de forma
independiente. Pasar los últimos 2–3 turnos reduciría la confusión entre fases
adyacentes (A-B, B-C), que es el patrón de error más frecuente observado en el arnés.
