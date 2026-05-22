from pathlib import Path
import sys
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
import json
import joblib


ROOT = next(
    (p for p in [Path.cwd(), *Path.cwd().parents] if (p / "project_paths.py").exists()),
    None
)
if ROOT is None:
    raise RuntimeError("Racine du projet introuvable (project_paths.py non trouvé).")

if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))


from project_paths import PROCESSED_DATA_DIR  # noqa: E402
from src.data.commun_preprocessing import (  # noqa: E402
    temporal_user_split,
    add_features,
    prepare_xy,
    add_test_features_from_train
)
from src.data.evluation_fonc import evaluate_model  # noqa: E402

MODEL_DIR = ROOT / "outputs" / "models"
METRICS_DIR = ROOT / "outputs" / "metrics"


def main() -> None:
    dataset_path = PROCESSED_DATA_DIR / "nextbuy.pkl.gz"
    df = pd.read_pickle(dataset_path, compression="gzip")

    train_df, test_df = temporal_user_split(df)

    train_df = add_features(train_df)
    test_df = add_test_features_from_train(train_df, test_df)

    X_train, y_train = prepare_xy(train_df)
    X_test, y_test = prepare_xy(test_df)

    model = RandomForestClassifier(
        n_estimators=100,
        max_depth=12,
        min_samples_split=10,
        class_weight="balanced",
        random_state=42,
        n_jobs=-1,
    )

    model.fit(X_train, y_train)

    metrics = evaluate_model(model, X_test, y_test)

    print("Random Forest metrics:")
    for key, value in metrics.items():
        print(f"{key}: {value:.4f}")

    model_path = MODEL_DIR / "random_forest.pkl"
    joblib.dump(model, model_path)

    metrics_path = METRICS_DIR / "random_forest_metrics.json"
    with open(metrics_path, "w", encoding="utf-8") as f:
        json.dump(metrics, f, indent=4)

    print(f"\nModel saved to: {model_path}")
    print(f"Metrics saved to: {metrics_path}")


if __name__ == "__main__":
    main()
