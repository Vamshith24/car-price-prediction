from pathlib import Path
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestRegressor
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

BASE_DIR = Path(__file__).resolve().parent.parent
DATA_DIR = BASE_DIR / "data"
OUTPUT_DIR = BASE_DIR / "outputs"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)


def find_dataset_csv(data_dir: Path) -> Path:
    csv_files = list(data_dir.glob("*.csv"))
    if not csv_files:
        raise FileNotFoundError(
            f"No CSV file found in {data_dir}. Download a Kaggle dataset into data/ first."
        )
    return csv_files[0]


def find_target_column(df: pd.DataFrame) -> str:
    candidates = ["Selling_Price", "selling_price", "Price", "price", "target"]
    for col in candidates:
        if col in df.columns:
            return col
    for col in df.columns:
        if "price" in col.lower():
            return col
    raise ValueError(f"Could not find a price target column in columns: {list(df.columns)}")


def main() -> None:
    csv_path = find_dataset_csv(DATA_DIR)
    df = pd.read_csv(csv_path)
    target_col = find_target_column(df)

    # Drop obvious ID or free-text name columns if present
    drop_cols = [
        c for c in df.columns
        if c != target_col and c.lower() in {"id", "car_id", "car_name", "name", "model"}
    ]
    df = df.drop(columns=drop_cols).dropna(subset=[target_col])

    X = df.drop(columns=[target_col])
    y = pd.to_numeric(df[target_col], errors="coerce")
    valid_mask = y.notna()
    X = X.loc[valid_mask]
    y = y.loc[valid_mask]

    numeric_cols = X.select_dtypes(include=[np.number]).columns.tolist()
    categorical_cols = X.select_dtypes(exclude=[np.number]).columns.tolist()

    numeric_transformer = Pipeline(steps=[
        ("imputer", SimpleImputer(strategy="median")),
        ("scaler", StandardScaler()),
    ])

    categorical_transformer = Pipeline(steps=[
        ("imputer", SimpleImputer(strategy="most_frequent")),
        ("onehot", OneHotEncoder(handle_unknown="ignore")),
    ])

    preprocessor = ColumnTransformer(transformers=[
        ("num", numeric_transformer, numeric_cols),
        ("cat", categorical_transformer, categorical_cols),
    ])

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )

    models = {
        "Linear Regression": LinearRegression(),
        "Random Forest": RandomForestRegressor(n_estimators=150, random_state=42),
    }

    report_lines = [
        f"Dataset: {csv_path.name}",
        f"Rows used: {len(df)} | Features: {X.shape[1]} | Target: {target_col}",
        "-" * 50,
    ]

    best_preds = None
    best_r2 = float("-inf")
    best_name = ""

    for name, regressor in models.items():
        pipe = Pipeline(steps=[
            ("preprocessor", preprocessor),
            ("model", regressor),
        ])
        pipe.fit(X_train, y_train)
        preds = pipe.predict(X_test)

        mae = mean_absolute_error(y_test, preds)
        rmse = np.sqrt(mean_squared_error(y_test, preds))
        r2 = r2_score(y_test, preds)

        line = f"{name} -> MAE: {mae:.4f} | RMSE: {rmse:.4f} | R2: {r2:.4f}"
        print(line)
        report_lines.append(line)

        if r2 > best_r2:
            best_r2 = r2
            best_preds = preds
            best_name = name

    metrics_path = OUTPUT_DIR / "metrics.txt"
    metrics_path.write_text("\n".join(report_lines), encoding="utf-8")

    plt.figure(figsize=(8, 6))
    plt.scatter(y_test, best_preds, alpha=0.65)
    min_val = min(y_test.min(), best_preds.min())
    max_val = max(y_test.max(), best_preds.max())
    plt.plot([min_val, max_val], [min_val, max_val], linestyle="--")
    plt.xlabel(f"Actual {target_col}")
    plt.ylabel(f"Predicted {target_col}")
    plt.title(f"Actual vs Predicted ({best_name})")
    plt.tight_layout()

    plot_path = OUTPUT_DIR / "actual_vs_predicted.png"
    plt.savefig(plot_path, dpi=150)
    plt.close()

    print(f"\nSaved metrics to: {metrics_path}")
    print(f"Saved plot to: {plot_path}")


if __name__ == "__main__":
    main()