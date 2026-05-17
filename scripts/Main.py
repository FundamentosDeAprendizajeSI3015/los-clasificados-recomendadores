"""
Main.py — Orquestador del pipeline completo.

Ejecuta todos los pasos en secuencia y genera reports en:
  - reports/graficas/  → imágenes para humanos
  - reports/json/      → datos estructurados para IA/BD

Uso:
    cd scripts
    python Main.py
"""

import time
import sys
import warnings
from pathlib import Path

# Asegurar que el directorio del script esté en el path
sys.path.insert(0, str(Path(__file__).resolve().parent))
warnings.filterwarnings("ignore")

from config import limpiar_reports, aplicar_tema

import paso01_preprocessing
import paso02_eda
import paso03_clustering
import paso04_dimensionality
import paso05_supervised
import paso06_metrics
import paso07_comparison

# ═══════════════════════════════════════════════════════════════════
# CONFIGURACIÓN DEL PIPELINE
# ═══════════════════════════════════════════════════════════════════
# Paso 01 es OPCIONAL: solo activar si se necesita regenerar
# data/dataset_processed.csv desde data/dataset.csv
RUN_PREPROCESSING = False


def main():
    print("=" * 70)
    print("  PIPELINE — Los Clasificados Recomendadores")
    print("=" * 70)
    print()

    # Aplicar tema visual
    aplicar_tema()

    # Limpiar reports/ al inicio de cada ejecución
    limpiar_reports()

    pasos = [
        ("paso01", "Preprocessing (opcional)", paso01_preprocessing.run,
         RUN_PREPROCESSING),
        ("paso02", "Análisis Exploratorio (EDA)", paso02_eda.run, True),
        ("paso03", "Clustering No Supervisado", paso03_clustering.run, True),
        ("paso04", "Reducción de Dimensionalidad", paso04_dimensionality.run,
         True),
        ("paso05", "Modelos Supervisados", paso05_supervised.run, True),
        ("paso06", "Consolidación de Métricas", paso06_metrics.run, True),
        ("paso07", "Comparación Final", paso07_comparison.run, True),
    ]

    resultados = {}
    tiempos = {}

    for paso_id, nombre, funcion, activo in pasos:
        if not activo:
            print(f"\n  >> {paso_id} — {nombre} [SALTADO]\n")
            continue

        print(f"\n{'=' * 70}")
        print(f"  {paso_id.upper()} — {nombre}")
        print(f"{'=' * 70}")

        t0 = time.time()
        try:
            resultado = funcion(resultados)
        except Exception as e:
            print(f"\n  [ERROR] {paso_id}: {e}")
            resultado = None
        t1 = time.time()

        tiempos[paso_id] = t1 - t0
        if resultado is not None:
            resultados[paso_id] = resultado

        print(f"\n  [tiempo] {nombre}: {t1 - t0:.1f}s")

    # ── Resumen final ────────────────────────────────────────────────
    print(f"\n{'=' * 70}")
    print("  PIPELINE COMPLETADO")
    print(f"{'=' * 70}")

    total = sum(tiempos.values())
    for paso, t in tiempos.items():
        print(f"  {paso}: {t:.1f}s")
    print(f"  {'─' * 30}")
    print(f"  Total: {total:.1f}s")
    print(f"\n  Reports en: reports/graficas/ y reports/json/")
    print(f"{'=' * 70}\n")


if __name__ == "__main__":
    main()
