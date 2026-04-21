from __future__ import annotations

import argparse
import json
from datetime import datetime
from pathlib import Path

import joblib
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, confusion_matrix, precision_recall_fscore_support
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler


TARGET_COLUMN = "genero_serie_rec"
SCRIPT_NAME = Path(__file__).stem
BASE_DIR = Path(__file__).resolve().parent
RESULTS_DIR = BASE_DIR / "resultados"


def parse_args() -> argparse.Namespace:
	parser = argparse.ArgumentParser(description="Entrena Regresion Logistica para genero_serie_rec")
	parser.add_argument("--data", type=Path, default=None, help="Ruta al CSV con features y targets")
	parser.add_argument("--test-size", type=float, default=0.2, help="Proporcion de test")
	parser.add_argument("--random-state", type=int, default=42, help="Semilla de reproducibilidad")
	return parser.parse_args()


def resolve_dataset_path(data_path: Path | None) -> Path:
	if data_path is not None:
		path = data_path.resolve()
		if not path.exists():
			raise FileNotFoundError(f"No existe el dataset especificado: {path}")
		return path

	candidates = [
		Path.cwd() / "data" / "processed" / "dataset_supervisado.csv",
		Path.cwd() / "data" / "processed" / "no_supervisado_con_targets.csv",
		Path.cwd() / "data" / "processed" / "dataset_con_targets.csv",
		Path.cwd() / "data" / "dataset_supervisado.csv",
		Path.cwd() / "dataset_supervisado.csv",
	]

	for candidate in candidates:
		if candidate.exists():
			return candidate.resolve()

	for csv_path in sorted(Path.cwd().glob("**/*.csv")):
		if "resultados" in csv_path.parts:
			continue
		try:
			columns = pd.read_csv(csv_path, nrows=0).columns.tolist()
			if TARGET_COLUMN in columns:
				return csv_path.resolve()
		except Exception:
			continue

	raise FileNotFoundError(
		"No se encontro un CSV con la columna target 'genero_serie_rec'. "
		"Pasa la ruta manualmente con --data <ruta_csv>."
	)


def build_pipeline(x_train: pd.DataFrame, random_state: int) -> Pipeline:
	numeric_features = x_train.select_dtypes(include=["number", "bool"]).columns.tolist()
	categorical_features = [col for col in x_train.columns if col not in numeric_features]

	numeric_pipeline = Pipeline(
		steps=[
			("imputer", SimpleImputer(strategy="median")),
			("scaler", StandardScaler()),
		]
	)
	categorical_pipeline = Pipeline(
		steps=[
			("imputer", SimpleImputer(strategy="most_frequent")),
			("onehot", OneHotEncoder(handle_unknown="ignore")),
		]
	)

	preprocessor = ColumnTransformer(
		transformers=[
			("num", numeric_pipeline, numeric_features),
			("cat", categorical_pipeline, categorical_features),
		]
	)

	model = LogisticRegression(
		random_state=random_state,
		max_iter=2000,
		class_weight="balanced",
	)

	return Pipeline(steps=[("preprocessor", preprocessor), ("model", model)])


def main() -> None:
	args = parse_args()
	RESULTS_DIR.mkdir(parents=True, exist_ok=True)

	dataset_path = resolve_dataset_path(args.data)
	df = pd.read_csv(dataset_path)

	if TARGET_COLUMN not in df.columns:
		raise ValueError(f"La columna target '{TARGET_COLUMN}' no existe en {dataset_path}")

	df = df.dropna(subset=[TARGET_COLUMN]).copy()
	x = df.drop(columns=[TARGET_COLUMN])
	y = df[TARGET_COLUMN].astype(str)

	class_counts = y.value_counts()
	stratify_y = y if y.nunique() > 1 and class_counts.min() >= 2 else None

	x_train, x_test, y_train, y_test = train_test_split(
		x,
		y,
		test_size=args.test_size,
		random_state=args.random_state,
		stratify=stratify_y,
	)

	pipeline = build_pipeline(x_train, args.random_state)
	pipeline.fit(x_train, y_train)
	y_pred = pipeline.predict(x_test)

	accuracy = accuracy_score(y_test, y_pred)
	precision, recall, f1, _ = precision_recall_fscore_support(
		y_test, y_pred, average="macro", zero_division=0
	)
	labels = sorted(set(y_test.tolist()) | set(y_pred.tolist()))
	cm = confusion_matrix(y_test, y_pred, labels=labels)

	model_path = RESULTS_DIR / f"{SCRIPT_NAME}_model.joblib"
	metrics_path = RESULTS_DIR / f"{SCRIPT_NAME}_metrics.json"
	predictions_path = RESULTS_DIR / f"{SCRIPT_NAME}_predicciones.csv"

	joblib.dump(pipeline, model_path)

	pd.DataFrame({"y_real": y_test.values, "y_pred": y_pred}).to_csv(predictions_path, index=False)

	metrics = {
		"script": SCRIPT_NAME,
		"target": TARGET_COLUMN,
		"dataset": str(dataset_path),
		"timestamp": datetime.now().isoformat(timespec="seconds"),
		"n_filas": int(df.shape[0]),
		"n_features": int(x.shape[1]),
		"n_clases": int(y.nunique()),
		"accuracy": float(accuracy),
		"precision_macro": float(precision),
		"recall_macro": float(recall),
		"f1_macro": float(f1),
		"labels_confusion_matrix": labels,
		"confusion_matrix": cm.tolist(),
		"model_path": str(model_path),
		"predictions_path": str(predictions_path),
	}

	with metrics_path.open("w", encoding="utf-8") as f:
		json.dump(metrics, f, ensure_ascii=False, indent=2)

	print(f"Modelo guardado en: {model_path}")
	print(f"Metricas guardadas en: {metrics_path}")
	print(f"Predicciones guardadas en: {predictions_path}")


if __name__ == "__main__":
	main()


