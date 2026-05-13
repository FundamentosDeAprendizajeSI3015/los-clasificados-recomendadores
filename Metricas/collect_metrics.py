import json
import pandas as pd
from pathlib import Path

def collect_all_metrics():
    """
    Recopila las métricas de todos los modelos supervisados y las consolida
    en un archivo CSV para análisis comparativo.
    """
    
    # Ruta base de modelos supervisados
    supervised_path = Path(__file__).parent.parent / "Supervised"
    
    metrics_data = []
    
    # Recorrer cada carpeta de modelo (Árbol de Decisión, Regresión Logística, etc.)
    for model_folder in supervised_path.iterdir():
        if not model_folder.is_dir() or model_folder.name == "README.md":
            continue
        
        model_name = model_folder.name
        reports_path = model_folder / "reports"
        
        if not reports_path.exists():
            continue
        
        # Recorrer cada tarea (genero_libro_rec, genero_musical_rec, etc.)
        for task_folder in reports_path.iterdir():
            if not task_folder.is_dir():
                continue

            task_name = task_folder.name

            evaluation_file = task_folder / "evaluation.json"

            if evaluation_file.exists():
                # Estructura estándar: reports/{target}/evaluation.json
                candidates = [(model_name, evaluation_file)]
            else:
                # Estructura anidada (SVM): reports/{target}/{variante}/evaluation.json
                candidates = []
                for variant_folder in sorted(task_folder.iterdir()):
                    if not variant_folder.is_dir():
                        continue
                    nested_eval = variant_folder / "evaluation.json"
                    if nested_eval.exists():
                        candidates.append((f"{model_name} ({variant_folder.name})", nested_eval))

            for display_model, eval_path in candidates:
                with open(eval_path, 'r', encoding='utf-8') as f:
                    eval_data = json.load(f)

                metric_row = {
                    "Modelo": display_model,
                    "Tarea": task_name,
                    "Accuracy": eval_data.get("accuracy", None),
                    "F1_Macro": eval_data.get("f1_macro", None),
                    "Precision_Macro": eval_data.get("classification_report", {}).get("macro avg", {}).get("precision", None),
                    "Recall_Macro": eval_data.get("classification_report", {}).get("macro avg", {}).get("recall", None),
                    "Precision_Weighted": eval_data.get("classification_report", {}).get("weighted avg", {}).get("precision", None),
                    "Recall_Weighted": eval_data.get("classification_report", {}).get("weighted avg", {}).get("recall", None),
                }

                metrics_data.append(metric_row)
    
    # Crear DataFrame
    df_metrics = pd.DataFrame(metrics_data)
    
    # Guardar en CSV
    output_csv = Path(__file__).parent / "metricas_consolidadas.csv"
    df_metrics.to_csv(output_csv, index=False, encoding='utf-8')
    print(f"[OK] Métricas consolidadas guardadas en: {output_csv}")
    
    # Guardar en JSON con formato más legible
    output_json = Path(__file__).parent / "metricas_consolidadas.json"
    with open(output_json, 'w', encoding='utf-8') as f:
        json.dump(metrics_data, f, indent=2, ensure_ascii=False)
    print(f"[OK] Métricas consolidadas guardadas en: {output_json}")
    
    # Mostrar resumen
    print("\n[RESUMEN DE MÉTRICAS]")
    print("=" * 80)
    print(df_metrics.to_string(index=False))
    
    # Resumen por modelo
    print("\n" + "=" * 80)
    print("[COMPARATIVA POR MODELO]")
    print("=" * 80)
    resumen_modelo = df_metrics.groupby("Modelo")[["Accuracy", "F1_Macro"]].mean()
    print(resumen_modelo.to_string())
    
    # Resumen por tarea
    print("\n" + "=" * 80)
    print("[COMPARATIVA POR TAREA]")
    print("=" * 80)
    resumen_tarea = df_metrics.groupby("Tarea")[["Accuracy", "F1_Macro"]].mean()
    print(resumen_tarea.to_string())
    
    return df_metrics

def get_metrics_dataframe():
    """
    Retorna un DataFrame con las métricas consolidadas.
    Útil para usar desde otros scripts.
    """
    metrics_csv = Path(__file__).parent / "metricas_consolidadas.csv"
    
    if metrics_csv.exists():
        return pd.read_csv(metrics_csv)
    else:
        print("[ADVERTENCIA] Archivo de métricas no encontrado. Ejecuta collect_all_metrics() primero.")
        return None

if __name__ == "__main__":
    print("[INICIANDO] Recopilando métricas de todos los modelos...")
    print("=" * 80)
    df = collect_all_metrics()
