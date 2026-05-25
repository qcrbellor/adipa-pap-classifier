"""
Reads the ADIPA technical test .docx and produces:
    data/dataset.csv          — one operator turn per row with phase and verbal act labels
    data/dataset_splits.json  — train/test split summary by case

Usage:
    python build_dataset.py                     # searches for .docx in current and parent dir
    python build_dataset.py path/to/file.docx   # explicit path

Phase labels come from structural headers ("A — ESCUCHA ACTIVA", etc.) in the scripts,
so they are weak labels derived from position rather than turn-level human annotation.
Verbal acts are assigned via regex heuristics as a silver-standard reference.
Mercedes and Luis are held out as test cases; the remaining 8 form the training set.
"""

import re
import csv
import json
import sys
from pathlib import Path
from collections import Counter

try:
    from docx import Document
except ImportError:
    print("ERROR: Instala python-docx primero:  pip install python-docx")
    sys.exit(1)


CASE_SPLITS = {
    "Camila":   "train",
    "Javiera":  "train",
    "Patricio": "train",
    "Hernán":   "train",
    "Rosa":     "train",
    "Matías":   "train",
    "Julia":    "train",
    "Carolina": "train",
    "Mercedes": "test",
    "Luis":     "test",
}

PHASE_HEADERS = {
    "A": re.compile(r"^A\s*[—–-]\s*(ESCUCHA|RECEPCI)", re.IGNORECASE),
    "B": re.compile(r"^B\s*[—–-]\s*(REGULACI)", re.IGNORECASE),
    "C": re.compile(r"^C\s*[—–-]\s*(CATEGORI)", re.IGNORECASE),
    "D": re.compile(r"^D\s*[—–-]\s*(DERIVACI|RED)", re.IGNORECASE),
    "E": re.compile(r"^E\s*[—–-]\s*(PSICOEDUC|ESPERANZA)", re.IGNORECASE),
}

OPERATOR_NAMES = r"OPERADOR[A]?|GABRIEL|ANDR[EÉ]S|MARCELA|VALENTINA|CLAUDIA|ISABEL"
OPERATOR_RE = re.compile(rf"^({OPERATOR_NAMES})\s+(.+)$", re.IGNORECASE)

PATIENT_RE = re.compile(
    r"^(CAMILA|JAVIERA|DON PATRICIO|PATRICIO|DON HERN[AÁ]N|HERN[AÁ]N|"
    r"SEÑORA ROSA|ROSA|MAT[IÍ]AS|JULIA|RODRIGO|MERCEDES|NOLASCO|"
    r"LUIS|CAROLINA|NIÑO|HIJO|HIJA|MADRE|VECIN[AO]|CONSERJE|"
    r"HERMANA|ENCARGADA|MARTA|BOMBERO|CARABINERO|PARAM[EÉ]DICO|"
    r"PROFE|PROFESOR[A]|ROJAS|VOZ EXTERNA|AGRESOR|VOZ DE ALBERGUE|"
    r"VOZ DE MEGÁFONO)\s+(.*)$",
    re.IGNORECASE,
)

CASE_HEADER_RE = re.compile(r"Guion\s*[—–-]?\s*Caso\s+(\w+)", re.IGNORECASE)

VERBAL_ACT_PATTERNS = {
    "validacion": re.compile(
        r"(entiendo|tiene sentido|comprendo|es normal|puede pasar|"
        r"es esperable|lamento|eso duele|eso es muy|fue una reacci|"
        r"tiene raz[oó]n|eso importa|es comprensible)",
        re.IGNORECASE,
    ),
    "pregunta_abierta": re.compile(
        r"\b(qu[eé]|c[oó]mo|cu[aá]ndo|por qu[eé]|cu[aá]l|"
        r"qu[eé] pas[oó]|qu[eé] siente|qu[eé] necesita)\b.*\?",
        re.IGNORECASE,
    ),
    "pregunta_cerrada": re.compile(
        r"\b(est[aá][n]?|tiene|hay|puede|va a|sigue|lleg[oó]|"
        r"est[aá]n|reconoce|sabe)\b.*\?",
        re.IGNORECASE,
    ),
    "reflejo": re.compile(
        r"^(dice que|siente que|escuch[oó] que|est[aá] en|"
        r"la imagen|perdieron|estaba|fue una|qued[oó]|"
        r"su compañero|su hijo|su esposo|su esposa)",
        re.IGNORECASE,
    ),
    "directivo": re.compile(
        r"^(no (abra|salga|corra|levante|vuelva|tome|diga|haga|"
        r"intente|corte|decida|forcejee|siga|use)|"
        r"quédese|ponga|apoye|mire|respire|inhale|suelte|"
        r"diga|repita|haga|vaya|camine|llame|pregunte|siéntese|"
        r"acérquese|levántese|cierre|abra|cambie)",
        re.IGNORECASE,
    ),
    "silencio_contencion": re.compile(
        r"(sigo aqu[ií]|no tiene que hablar|t[oó]mese su tiempo|"
        r"puede quedarse|eso basta|estoy con usted|sigo en la llamada|"
        r"no tiene que decir|puede llorar|no tiene que calmarse)",
        re.IGNORECASE,
    ),
    "interpretacion": re.compile(
        r"(parece que|suena como si|puede ser que|quizás|"
        r"su cuerpo cree|su mente busca|una parte de usted|"
        r"eso puede significar|lo que describe)",
        re.IGNORECASE,
    ),
    "confrontacion": re.compile(
        r"(sin embargo|pero tambi[eé]n|y aun as[ií]|"
        r"reconocemos ambas|eso no justifica|no es solo|"
        r"y al mismo tiempo|aunque)",
        re.IGNORECASE,
    ),
}


def clean(text: str) -> str:
    return re.sub(r"\*+", "", text).strip()


def detect_phase(text: str) -> str | None:
    t = clean(text)
    for phase, pat in PHASE_HEADERS.items():
        if pat.search(t):
            return phase
    return None


def detect_verbal_acts(text: str) -> list[str]:
    acts = [act for act, pat in VERBAL_ACT_PATTERNS.items() if pat.search(text)]
    return acts if acts else ["otro"]


def is_stage_direction(text: str) -> bool:
    t = text.strip()
    return bool(
        t.startswith("*")
        or t.startswith("(")
        or t.startswith("INT.")
        or t.startswith("EXT.")
        or re.match(r"^Se escucha", t)
        or re.match(r"^(Pausa|Silencio|Pasa)", t)
        or re.match(r"^\d+\.", t)
    )


def extract_paragraphs(docx_path: Path) -> list[str]:
    doc = Document(docx_path)
    return [p.text for p in doc.paragraphs]


def parse(paragraphs: list[str]) -> list[dict]:
    records = []
    current_case = None
    current_phase = "A"
    prev_patient_text = ""
    global_id = 0
    case_turn_counter = 0

    op_buffer: list[str] = []
    collecting = False

    def flush():
        nonlocal global_id, case_turn_counter
        if not op_buffer or current_case is None:
            op_buffer.clear()
            return
        text = " ".join(op_buffer).strip()
        text = re.sub(r'\s*RAMIFICACI[OÓ]N\s+\d+[^.]*\.?', '', text).strip()
        if len(text) < 3:
            op_buffer.clear()
            return
        records.append({
            "caso": current_case,
            "turno_id": f"{current_case}_{case_turn_counter:03d}",
            "global_id": global_id,
            "fase": current_phase,
            "operador_texto": text,
            "contexto_paciente_previo": prev_patient_text,
            "actos_verbales": "|".join(detect_verbal_acts(text)),
            "split": CASE_SPLITS.get(current_case, "train"),
        })
        global_id += 1
        op_buffer.clear()

    for raw in paragraphs:
        line = clean(raw)
        if not line:
            continue

        case_match = CASE_HEADER_RE.search(line)
        if case_match:
            if collecting:
                flush()
                collecting = False
            current_case = case_match.group(1).capitalize()
            current_phase = "A"
            prev_patient_text = ""
            case_turn_counter = 0
            continue

        if current_case is None:
            continue

        phase = detect_phase(line)
        if phase:
            if collecting:
                flush()
                collecting = False
            current_phase = phase
            continue

        if is_stage_direction(raw):
            if collecting:
                flush()
                collecting = False
            continue

        op_match = OPERATOR_RE.match(line)
        if op_match:
            if collecting:
                flush()
            case_turn_counter += 1
            collecting = True
            op_buffer = [op_match.group(2).strip()]
            continue

        pat_match = PATIENT_RE.match(line)
        if pat_match:
            if collecting:
                flush()
                collecting = False
            captured = pat_match.group(2).strip() if pat_match.lastindex and pat_match.lastindex >= 2 else line
            prev_patient_text = captured
            continue

        if collecting:
            op_buffer.append(line)

    if collecting:
        flush()

    return records


def save(records: list[dict], out_dir: Path):
    out_dir.mkdir(parents=True, exist_ok=True)

    fields = [
        "caso", "turno_id", "global_id", "fase",
        "operador_texto", "contexto_paciente_previo",
        "actos_verbales", "split",
    ]
    csv_path = out_dir / "dataset.csv"
    with open(csv_path, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=fields)
        w.writeheader()
        w.writerows(records)

    summary: dict = {}
    for r in records:
        summary.setdefault(r["split"], {}).setdefault(r["caso"], 0)
        summary[r["split"]][r["caso"]] += 1

    split_path = out_dir / "dataset_splits.json"
    with open(split_path, "w", encoding="utf-8") as f:
        json.dump(summary, f, ensure_ascii=False, indent=2)

    return csv_path, split_path


def report(records: list[dict]):
    print(f"\n{'='*55}")
    print(f"  DATASET PAP — Resumen de construcción")
    print(f"{'='*55}")
    print(f"  Total de turnos extraídos : {len(records)}")

    phase_counts = Counter(r["fase"] for r in records)
    print(f"\n  Distribución por fase:")
    labels = {"A": "Escucha Activa", "B": "Regulación",
              "C": "Categorización", "D": "Derivación", "E": "Psicoeducación"}
    for p in "ABCDE":
        n = phase_counts.get(p, 0)
        bar = "█" * (n // 15)
        print(f"    {p} ({labels[p]:<20}) {n:>4}  {bar}")

    split_counts = Counter(r["split"] for r in records)
    print(f"\n  Split:")
    print(f"    train (8 casos) : {split_counts.get('train', 0)}")
    print(f"    test  (held-out): {split_counts.get('test',  0)}")
    print(f"    → Casos held-out: Mercedes, Luis")

    act_counts: Counter = Counter()
    for r in records:
        for act in r["actos_verbales"].split("|"):
            act_counts[act] += 1
    print(f"\n  Actos verbales (heurístico silver-standard):")
    for act, n in act_counts.most_common():
        print(f"    {act:<25} {n:>4}")

    print(f"\n  Turnos por caso:")
    case_counts = Counter(r["caso"] for r in records)
    for case, n in sorted(case_counts.items()):
        split = CASE_SPLITS.get(case, "train")
        tag = "🔒 test" if split == "test" else "  train"
        print(f"    {tag}  {case:<12} {n:>4} turnos")
    print(f"{'='*55}\n")


if __name__ == "__main__":
    if len(sys.argv) > 1:
        docx_path = Path(sys.argv[1])
    else:
        candidates = list(Path(__file__).parent.glob("*.docx")) + \
                     list(Path(__file__).parent.parent.glob("*.docx"))
        if not candidates:
            print("ERROR: No se encontró ningún .docx.")
            print("Uso: python build_dataset.py ruta/a/Prueba_Tecnica_ADIPA.docx")
            sys.exit(1)
        docx_path = candidates[0]

    print(f"📄 Leyendo: {docx_path.name}")

    paragraphs = extract_paragraphs(docx_path)
    print(f"   {len(paragraphs)} párrafos extraídos del documento")

    records = parse(paragraphs)

    out_dir = Path(__file__).parent
    csv_path, split_path = save(records, out_dir)

    report(records)

    print(f"  {csv_path.name}    guardado en {csv_path.parent}")
    print(f"  {split_path.name} guardado en {split_path.parent}")
