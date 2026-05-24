"""
Evaluation harness for the PAP turn classifier.

Runs the classifier over the full dataset (train + held-out), reports
precision/recall/F1 per phase, confusion matrices, and failure examples.
Results are saved to evaluation/results/.

Usage:
    export ANTHROPIC_API_KEY=sk-ant-...
    python evaluation/evaluate.py

    # Without API key — uses simulated predictions to exercise the harness
    python evaluation/evaluate.py --mock
"""

import csv
import json
import sys
import os
import argparse
import random
import time
from pathlib import Path
from collections import Counter
from datetime import datetime

PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from classifier import PAPClassifier, ClassificationResult

PHASES = ["A", "B", "C", "D", "E"]
PHASE_NAMES = {
    "A": "Escucha Activa",
    "B": "Regulación",
    "C": "Categorización",
    "D": "Derivación",
    "E": "Psicoeducación",
}
RESULTS_DIR = PROJECT_ROOT / "evaluation" / "results"


def load_dataset(csv_path: Path) -> list[dict]:
    with open(csv_path, encoding="utf-8") as f:
        return list(csv.DictReader(f))


class MockClassifier:
    """Simulates the classifier with predictions biased toward the correct phase (p=0.65).
    Used to exercise the harness without consuming API tokens."""

    def __init__(self, seed: int = 42):
        random.seed(seed)

    def classify(self, operator_text: str, patient_context: str = "") -> ClassificationResult:
        time.sleep(0.002)
        return ClassificationResult(
            fase=random.choice(PHASES),
            actos_verbales=["otro"],
            confianza=round(random.uniform(0.6, 0.95), 2),
            llm_available=False,
        )

    def classify_batch(self, turns: list[dict], verbose: bool = False) -> list[ClassificationResult]:
        results = []
        for i, t in enumerate(turns):
            if verbose and i % 100 == 0:
                print(f"  [{i+1}/{len(turns)}]", flush=True)
            correct_phase = t.get("fase", "A")
            if random.random() < 0.65:
                fase = correct_phase
            else:
                fase = random.choice([p for p in PHASES if p != correct_phase])
            results.append(ClassificationResult(
                fase=fase,
                actos_verbales=["otro"],
                confianza=round(random.uniform(0.6, 0.95), 2),
                llm_available=False,
            ))
        return results


def compute_metrics(
    y_true: list[str],
    y_pred: list[str],
    labels: list[str] = PHASES,
) -> dict:
    """Computes precision, recall, and F1 per class plus macro and weighted averages."""
    tp = {c: 0 for c in labels}
    fp = {c: 0 for c in labels}
    fn = {c: 0 for c in labels}

    for true, pred in zip(y_true, y_pred):
        if true == pred:
            tp[true] += 1
        else:
            fp[pred] += 1
            fn[true] += 1

    per_class = {}
    for c in labels:
        p = tp[c] / (tp[c] + fp[c]) if (tp[c] + fp[c]) > 0 else 0.0
        r = tp[c] / (tp[c] + fn[c]) if (tp[c] + fn[c]) > 0 else 0.0
        f = 2 * p * r / (p + r) if (p + r) > 0 else 0.0
        support = sum(1 for y in y_true if y == c)
        per_class[c] = {
            "precision": round(p, 4),
            "recall": round(r, 4),
            "f1": round(f, 4),
            "support": support,
        }

    macro_p = sum(per_class[c]["precision"] for c in labels) / len(labels)
    macro_r = sum(per_class[c]["recall"] for c in labels) / len(labels)
    macro_f = sum(per_class[c]["f1"] for c in labels) / len(labels)

    total = len(y_true)
    w_p = sum(per_class[c]["precision"] * per_class[c]["support"] for c in labels) / total if total else 0
    w_r = sum(per_class[c]["recall"] * per_class[c]["support"] for c in labels) / total if total else 0
    w_f = sum(per_class[c]["f1"] * per_class[c]["support"] for c in labels) / total if total else 0
    accuracy = sum(1 for t, p in zip(y_true, y_pred) if t == p) / total if total else 0

    return {
        "per_class": per_class,
        "macro": {"precision": round(macro_p, 4), "recall": round(macro_r, 4), "f1": round(macro_f, 4)},
        "weighted": {"precision": round(w_p, 4), "recall": round(w_r, 4), "f1": round(w_f, 4)},
        "accuracy": round(accuracy, 4),
        "total": total,
    }


def confusion_matrix(
    y_true: list[str],
    y_pred: list[str],
    labels: list[str] = PHASES,
) -> dict[str, dict[str, int]]:
    """Returns the confusion matrix as dict[true][pred] = count."""
    matrix = {t: {p: 0 for p in labels} for t in labels}
    for true, pred in zip(y_true, y_pred):
        if true in matrix and pred in labels:
            matrix[true][pred] += 1
    return matrix


def find_failures(
    rows: list[dict],
    predictions: list[ClassificationResult],
    n: int = 15,
) -> list[dict]:
    """Returns up to n misclassified examples."""
    failures = []
    for row, pred in zip(rows, predictions):
        if row["fase"] != pred.fase:
            failures.append({
                "caso": row["caso"],
                "turno_id": row["turno_id"],
                "fase_real": row["fase"],
                "fase_pred": pred.fase,
                "confianza": pred.confianza,
                "actos_pred": "|".join(pred.actos_verbales),
                "contexto": row["contexto_paciente_previo"][:100],
                "operador_texto": row["operador_texto"][:150],
            })
        if len(failures) >= n:
            break
    return failures


def format_metrics_table(metrics: dict, title: str) -> str:
    lines = [f"\n{'═'*60}", f"  {title}", f"{'═'*60}"]
    lines.append(f"  Accuracy: {metrics['accuracy']:.1%}  |  Total turnos: {metrics['total']}")
    lines.append(f"\n  {'Fase':<22} {'Precision':>10} {'Recall':>10} {'F1':>10} {'Support':>9}")
    lines.append(f"  {'-'*62}")
    for phase in PHASES:
        m = metrics["per_class"][phase]
        name = f"{phase} — {PHASE_NAMES[phase]}"
        lines.append(
            f"  {name:<22} {m['precision']:>10.3f} {m['recall']:>10.3f} "
            f"{m['f1']:>10.3f} {m['support']:>9}"
        )
    lines.append(f"  {'-'*62}")
    m = metrics["macro"]
    lines.append(f"  {'macro avg':<22} {m['precision']:>10.3f} {m['recall']:>10.3f} {m['f1']:>10.3f}")
    m = metrics["weighted"]
    lines.append(f"  {'weighted avg':<22} {m['precision']:>10.3f} {m['recall']:>10.3f} {m['f1']:>10.3f}")
    return "\n".join(lines)


def format_confusion_matrix(matrix: dict, title: str) -> str:
    lines = [f"\n  Matriz de confusión — {title}"]
    lines.append(f"  {'':8}" + "".join(f"  pred_{p}" for p in PHASES))
    for true in PHASES:
        row = f"  true_{true}  " + "".join(f"{matrix[true][p]:>7}" for p in PHASES)
        lines.append(row)
    return "\n".join(lines)


def format_failures(failures: list[dict], title: str) -> str:
    if not failures:
        return f"\n  ✅ Sin errores en {title}"
    lines = [f"\n  Errores de clasificación — {title} (primeros {len(failures)})"]
    lines.append(f"  {'─'*70}")
    for i, f in enumerate(failures, 1):
        lines.append(
            f"\n  [{i}] {f['turno_id']}  "
            f"real={f['fase_real']} → pred={f['fase_pred']}  "
            f"conf={f['confianza']:.2f}"
        )
        if f["contexto"]:
            lines.append(f"      PAC: {f['contexto']}")
        lines.append(f"      OP:  {f['operador_texto']}")
    return "\n".join(lines)


def save_results(
    split_name: str,
    metrics: dict,
    matrix: dict,
    failures: list[dict],
    results_dir: Path,
):
    results_dir.mkdir(parents=True, exist_ok=True)

    with open(results_dir / f"metrics_{split_name}.json", "w", encoding="utf-8") as f:
        json.dump(metrics, f, ensure_ascii=False, indent=2)

    cm_path = results_dir / f"confusion_matrix_{split_name}.csv"
    with open(cm_path, "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["true\\pred"] + PHASES)
        for true in PHASES:
            w.writerow([true] + [matrix[true][p] for p in PHASES])

    if failures:
        fail_path = results_dir / f"failures_{split_name}.csv"
        fields = ["caso", "turno_id", "fase_real", "fase_pred",
                  "confianza", "actos_pred", "contexto", "operador_texto"]
        with open(fail_path, "w", newline="", encoding="utf-8") as f:
            w = csv.DictWriter(f, fieldnames=fields)
            w.writeheader()
            w.writerows(failures)


def run_evaluation(use_mock: bool = False):
    print(f"\n{'█'*60}")
    print(f"  ADIPA PAP — Arnés de Evaluación")
    print(f"  {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"  Modo: {'MOCK (sin API)' if use_mock else 'LLM real'}")
    print(f"{'█'*60}\n")

    csv_path = PROJECT_ROOT / "data" / "dataset.csv"
    if not csv_path.exists():
        print(f"ERROR: No se encontró {csv_path}")
        print("Ejecuta primero: python data/build_dataset.py <ruta_docx>")
        sys.exit(1)

    all_rows = load_dataset(csv_path)
    train_rows = [r for r in all_rows if r["split"] == "train"]
    test_rows = [r for r in all_rows if r["split"] == "test"]
    print(f"  Dataset cargado: {len(train_rows)} train | {len(test_rows)} test\n")

    if use_mock:
        clf = MockClassifier()
        print("  ⚠️  Usando clasificador simulado (--mock)\n")
    else:
        api_key = os.getenv("ANTHROPIC_API_KEY", "")
        if not api_key:
            print("  ⚠️  ANTHROPIC_API_KEY no configurada — usando mock automáticamente\n")
            clf = MockClassifier()
        else:
            clf = PAPClassifier(api_key=api_key, use_cache=True)
            print("  ✅  Clasificador LLM inicializado\n")

    print(f"  Clasificando {len(train_rows)} turnos de TRAIN...")
    t0 = time.time()
    train_preds = clf.classify_batch(train_rows, verbose=True)
    train_time = time.time() - t0
    print(f"  ✅ Train completado en {train_time:.1f}s "
          f"({train_time/len(train_rows)*1000:.0f}ms/turno)\n")

    print(f"  Clasificando {len(test_rows)} turnos de TEST (held-out: Mercedes + Luis)...")
    t0 = time.time()
    test_preds = clf.classify_batch(test_rows, verbose=True)
    test_time = time.time() - t0
    print(f"  ✅ Test completado en {test_time:.1f}s "
          f"({test_time/len(test_rows)*1000:.0f}ms/turno)\n")

    train_true = [r["fase"] for r in train_rows]
    train_pred = [p.fase for p in train_preds]
    train_metrics = compute_metrics(train_true, train_pred)
    train_matrix = confusion_matrix(train_true, train_pred)
    train_failures = find_failures(train_rows, train_preds)

    test_true = [r["fase"] for r in test_rows]
    test_pred = [p.fase for p in test_preds]
    test_metrics = compute_metrics(test_true, test_pred)
    test_matrix = confusion_matrix(test_true, test_pred)
    test_failures = find_failures(test_rows, test_preds)

    per_case_metrics = {}
    for case in ["Mercedes", "Luis"]:
        case_rows = [r for r in test_rows if r["caso"] == case]
        case_preds = [p for r, p in zip(test_rows, test_preds) if r["caso"] == case]
        if case_rows:
            per_case_metrics[case] = compute_metrics(
                [r["fase"] for r in case_rows],
                [p.fase for p in case_preds],
            )

    f1_gap = train_metrics["macro"]["f1"] - test_metrics["macro"]["f1"]

    report_lines = []
    report_lines.append(format_metrics_table(train_metrics, "INTRA-TRAIN (8 casos)"))
    report_lines.append(format_confusion_matrix(train_matrix, "TRAIN"))
    report_lines.append(format_failures(train_failures, "train"))

    report_lines.append("\n")
    report_lines.append(format_metrics_table(test_metrics, "HELD-OUT TEST (Mercedes + Luis)"))
    report_lines.append(format_confusion_matrix(test_matrix, "TEST"))
    report_lines.append(format_failures(test_failures, "test"))

    report_lines.append(f"\n{'═'*60}")
    report_lines.append("  Desglose por caso held-out")
    report_lines.append(f"{'═'*60}")
    for case, m in per_case_metrics.items():
        report_lines.append(
            f"  {case:<12}  accuracy={m['accuracy']:.1%}  "
            f"macro-F1={m['macro']['f1']:.3f}  n={m['total']}"
        )

    report_lines.append(f"\n{'═'*60}")
    report_lines.append("  Gap train → held-out")
    report_lines.append(f"{'═'*60}")
    report_lines.append(f"  Macro-F1 train : {train_metrics['macro']['f1']:.3f}")
    report_lines.append(f"  Macro-F1 test  : {test_metrics['macro']['f1']:.3f}")
    gap_sign = "▼" if f1_gap > 0.05 else ("▲" if f1_gap < 0 else "≈")
    report_lines.append(f"  Gap            : {f1_gap:+.3f}  {gap_sign}")
    if f1_gap > 0.15:
        report_lines.append("  ⚠️  Gap > 0.15 sugiere sobreajuste al estilo de redacción del train.")
    elif f1_gap < 0.05:
        report_lines.append("  ✅ Gap pequeño: el clasificador generaliza bien a arquetipos nuevos.")
    report_lines.append(f"{'═'*60}\n")

    full_report = "\n".join(report_lines)
    print(full_report)

    RESULTS_DIR.mkdir(parents=True, exist_ok=True)

    save_results("train", train_metrics, train_matrix, train_failures, RESULTS_DIR)
    save_results("test", test_metrics, test_matrix, test_failures, RESULTS_DIR)

    full_json = {
        "timestamp": datetime.now().isoformat(),
        "mock": use_mock,
        "train": {**train_metrics, "tiempo_segundos": round(train_time, 2)},
        "test": {**test_metrics, "tiempo_segundos": round(test_time, 2)},
        "per_case": per_case_metrics,
        "f1_gap": round(f1_gap, 4),
    }
    with open(RESULTS_DIR / "metrics.json", "w", encoding="utf-8") as f:
        json.dump(full_json, f, ensure_ascii=False, indent=2)

    with open(RESULTS_DIR / "report.txt", "w", encoding="utf-8") as f:
        f.write(f"ADIPA PAP Classifier — Reporte de Evaluación\n")
        f.write(f"Generado: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
        f.write(f"Modo: {'MOCK' if use_mock else 'LLM real'}\n\n")
        f.write(full_report)

    print(f"\n  Resultados guardados en: {RESULTS_DIR}/")
    print(f"  ├── metrics.json")
    print(f"  ├── metrics_train.json")
    print(f"  ├── metrics_test.json")
    print(f"  ├── confusion_matrix_train.csv")
    print(f"  ├── confusion_matrix_test.csv")
    print(f"  ├── failures_train.csv")
    print(f"  ├── failures_test.csv")
    print(f"  └── report.txt\n")

    return full_json


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Arnés de evaluación PAP Classifier")
    parser.add_argument(
        "--mock",
        action="store_true",
        help="Usar clasificador simulado (no consume API tokens).",
    )
    args = parser.parse_args()
    run_evaluation(use_mock=args.mock)
