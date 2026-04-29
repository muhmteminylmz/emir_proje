"""
train.py - Model Egitim Scripti
--------------------------------
Calistirmak icin:
    python train.py

Bu script:
  1. data/processed/raw_dataset.csv dosyasini yukler
     (yoksa data/raw/ dizinindeki CSV dosyalarindan birlestirir)
  2. Feature cikarimi yapar (FeaturePipeline)
  3. Zaman bazli train/test bolumleme uygular (2023 train, 2024 test)
  4. RandomForest, XGBoost ve Logistic Regression modellerini egitir
  5. Adversarial saldirilar calistirir
  6. Savunmali (robust) modeller olusturur
  7. Tum modelleri data/models/ klasorune kaydeder

Sunucu baslatmak icin:
    streamlit run app.py
"""

import sys
import os
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

import json
import warnings
warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd
from pathlib import Path
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from xgboost import XGBClassifier

from src.feature_extractor import FeaturePipeline, time_based_split
from src.adversarial import AttackRunner
from src.defense import build_defended_models
from src.utils import (
    compute_metrics, metrics_table, plot_confusion_matrix,
    plot_roc_curve, plot_feature_importance, save_model,
    print_section, DATA_RAW, DATA_PROCESSED, DATA_MODELS, ensure_dirs,
)

# ─────────────────────────────────────────────
# SABITLER
# ─────────────────────────────────────────────

RAW_DATASET_PATH = DATA_PROCESSED / "raw_dataset.csv"
FEATURE_MATRIX_PATH = DATA_PROCESSED / "feature_matrix.csv"


# ─────────────────────────────────────────────
# 1. VERI YUKLEME
# ─────────────────────────────────────────────

def load_raw_data() -> pd.DataFrame:
    """
    Ham URL verisini yukler.
    Once data/processed/raw_dataset.csv'yi dener,
    bulunamazsa data/raw/ klasoründeki CSV'leri birlestir.
    """
    if RAW_DATASET_PATH.exists():
        print(f"Veri yukleniyor: {RAW_DATASET_PATH}")
        df = pd.read_csv(RAW_DATASET_PATH)
        df["timestamp"] = pd.to_datetime(df["timestamp"], utc=True, errors="coerce")
        df = df.dropna(subset=["url", "label", "timestamp"])
        df["label"] = df["label"].astype(int)
        print(f"  {len(df):,} URL yuklendi (Phishing: {df['label'].sum():,}, "
              f"Legitimate: {(df['label']==0).sum():,})")
        return df

    # raw CSV'lerden yukle
    from src.data_collector import KaggleLoader, DatasetMerger
    loader = KaggleLoader()
    merger = DatasetMerger()

    raw_csvs = list(DATA_RAW.glob("*.csv"))
    if not raw_csvs:
        raise FileNotFoundError(
            f"Veri bulunamadi! Lutfen data/raw/ klasorune bir phishing CSV dosyasi ekleyin.\n"
            f"Veya PhishTank ve Kaggle'dan veri toplamak icin 'notebooks/1_data_collection.ipynb' calistirin."
        )

    dfs = []
    for csv_path in raw_csvs:
        try:
            dfs.append(loader.load(str(csv_path)))
        except Exception as e:
            print(f"  {csv_path.name} yuklenemedi: {e}")

    if not dfs:
        raise ValueError("Hicbir CSV dosyasi yuklenemedi.")

    combined = merger.merge(dfs[0], dfs[1] if len(dfs) > 1 else dfs[0])
    combined.to_csv(RAW_DATASET_PATH, index=False)
    print(f"  Birlestirilen veri kaydedildi: {RAW_DATASET_PATH}")
    return combined


# ─────────────────────────────────────────────
# 2. FEATURE CIKARIMI
# ─────────────────────────────────────────────

def extract_features(df: pd.DataFrame) -> pd.DataFrame:
    """FeaturePipeline kullanarak feature cikarimi yapar."""
    if FEATURE_MATRIX_PATH.exists():
        print(f"Onceden hesaplanmis feature matrix yukleniyor: {FEATURE_MATRIX_PATH}")
        feature_df = pd.read_csv(FEATURE_MATRIX_PATH)
        feature_df["timestamp"] = pd.to_datetime(feature_df["timestamp"], utc=True, errors="coerce")
        print(f"  {len(feature_df):,} satir, {feature_df.shape[1]} sutun")
        return feature_df

    pipeline = FeaturePipeline()
    feature_df = pipeline.transform(df)
    feature_df.to_csv(FEATURE_MATRIX_PATH, index=False)
    print(f"Feature matrix kaydedildi: {FEATURE_MATRIX_PATH}")
    return feature_df


# ─────────────────────────────────────────────
# 3. TRAIN/TEST BOLUMLEME
# ─────────────────────────────────────────────

def split_data(feature_df: pd.DataFrame):
    """Zaman bazli train/test bolumleme (2023 → train, 2024 → test)."""
    X_train, X_test, y_train, y_test, _, _, feature_names = time_based_split(
        feature_df, train_end_year=2023, test_start_year=2024
    )

    # Numpy dosyalarini kaydet
    np.save(DATA_PROCESSED / "X_train.npy", X_train)
    np.save(DATA_PROCESSED / "X_test.npy", X_test)
    np.save(DATA_PROCESSED / "y_train.npy", y_train)
    np.save(DATA_PROCESSED / "y_test.npy", y_test)

    with open(DATA_PROCESSED / "feature_names.json", "w") as f:
        json.dump(feature_names, f, ensure_ascii=False)

    print(f"  Train/test split kaydedildi")
    return X_train, X_test, y_train, y_test, feature_names


# ─────────────────────────────────────────────
# 4. MODEL EGITIMI
# ─────────────────────────────────────────────

def build_baseline_models():
    """Egitilecek baseline modelleri tanimlar."""
    return {
        "Random Forest": RandomForestClassifier(
            n_estimators=200, max_depth=15, min_samples_split=5,
            min_samples_leaf=2, class_weight="balanced",
            n_jobs=-1, random_state=42,
        ),
        "XGBoost": XGBClassifier(
            n_estimators=200, max_depth=6, learning_rate=0.1,
            subsample=0.8, colsample_bytree=0.8, scale_pos_weight=1,
            n_jobs=-1, random_state=42, eval_metric="logloss", verbosity=0,
        ),
        "Logistic Regression": Pipeline([
            ("scaler", StandardScaler()),
            ("lr", LogisticRegression(
                C=1.0, max_iter=1000, class_weight="balanced",
                n_jobs=-1, random_state=42,
            )),
        ]),
    }


def train_baseline_models(models: dict, X_train, X_test, y_train, y_test,
                           feature_names: list) -> dict:
    """Baseline modelleri egitir, degerlendirir ve kaydeder."""
    print_section("Baseline Model Egitimi")
    trained = {}
    all_metrics = []
    y_prob_dict = {}

    for name, model in models.items():
        print(f"\n  {name} egitiliyor...")
        model.fit(X_train, y_train)
        trained[name] = model

        y_pred = model.predict(X_test)
        y_prob = model.predict_proba(X_test)[:, 1]
        y_prob_dict[name] = y_prob

        metrics = compute_metrics(y_test, y_pred, y_prob, model_name=name)
        all_metrics.append(metrics)

        plot_confusion_matrix(y_test, y_pred, model_name=name)

        # Feature importance (RF ve XGBoost icin)
        if hasattr(model, "feature_importances_"):
            plot_feature_importance(feature_names, model.feature_importances_,
                                    model_name=name, top_n=20)

        # Modeli kaydet
        safe_name = name.lower().replace(" ", "_")
        save_model(model, f"baseline_{safe_name}")

    plot_roc_curve(y_test, y_prob_dict)

    print_section("Baseline Sonuclar")
    print(metrics_table(all_metrics).to_string())

    return trained


# ─────────────────────────────────────────────
# 5. ADVERSARIAL SALDIRI
# ─────────────────────────────────────────────

def run_adversarial_attacks(trained_models: dict, X_test, y_test,
                             feature_names: list) -> dict:
    """Adversarial saldirilar calistirir ve sonuclari kaydeder."""
    print_section("Adversarial Saldirilari")

    runner = AttackRunner(budget=5)
    results = runner.run_all_attacks(trained_models, X_test, y_test, feature_names)

    # Adversarial ornekleri kaydet
    adv_ex = results.get("adversarial_examples", {})
    for key, X_adv in adv_ex.items():
        np.save(DATA_PROCESSED / f"X_adv_{key}.npy", X_adv)

    # Ozet tablosu
    summary_df = runner.summarize(results)
    summary_df.to_csv(DATA_PROCESSED / "attack_results.csv", index=False)

    return results


# ─────────────────────────────────────────────
# 6. SAVUNMA MODELLERI
# ─────────────────────────────────────────────

def train_defended_models(trained_models: dict, X_train, y_train,
                           feature_names: list) -> dict:
    """Savunmali (robust) modeller olusturur ve kaydeder."""
    print_section("Savunma Modelleri Egitimi")

    defended = build_defended_models(trained_models, X_train, y_train, feature_names)

    for name, model in defended.items():
        safe_name = name.lower().replace(" ", "_").replace("(", "").replace(")", "").replace("/", "_")
        save_model(model, f"defended_{safe_name}")

    return defended


# ─────────────────────────────────────────────
# ANA AKIS
# ─────────────────────────────────────────────

def main():
    ensure_dirs(DATA_RAW, DATA_PROCESSED, DATA_MODELS)

    # 1. Veri yukle
    print_section("1 / 6  Veri Yukleme")
    df = load_raw_data()

    # 2. Feature cikari
    print_section("2 / 6  Feature Cikarimi")
    feature_df = extract_features(df)

    # 3. Train/test bolumleme
    print_section("3 / 6  Train/Test Bolumleme")
    X_train, X_test, y_train, y_test, feature_names = split_data(feature_df)

    # 4. Baseline egitim
    print_section("4 / 6  Baseline Model Egitimi")
    models = build_baseline_models()
    trained = train_baseline_models(models, X_train, X_test, y_train, y_test, feature_names)

    # 5. Adversarial saldiri
    print_section("5 / 6  Adversarial Saldirilari")
    run_adversarial_attacks(trained, X_test, y_test, feature_names)

    # 6. Savunma modelleri
    print_section("6 / 6  Savunma Modelleri")
    train_defended_models(trained, X_train, y_train, feature_names)

    print_section("Egitim Tamamlandi!")
    print("  Kaydedilen modeller: data/models/")
    print("  Sunucuyu baslatmak icin: streamlit run app.py")


if __name__ == "__main__":
    main()
