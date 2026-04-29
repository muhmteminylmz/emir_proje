"""
utils.py - Yardımcı Fonksiyonlar
Phishing Adversarial Project
"""

import os
import json
import logging
import hashlib
import joblib
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib
matplotlib.use('Agg')
import seaborn as sns
from sklearn.metrics import (
    classification_report, confusion_matrix,
    roc_auc_score, f1_score, precision_score,
    recall_score, accuracy_score, roc_curve
)

# ─────────────────────────────────────────────
# LOGGING KURULUMU
# ─────────────────────────────────────────────

def setup_logger(name: str, log_file: str = None, level=logging.INFO) -> logging.Logger:
    """Proje genelinde kullanılacak logger."""
    formatter = logging.Formatter(
        '[%(asctime)s] %(levelname)s [%(name)s] %(message)s',
        datefmt='%Y-%m-%d %H:%M:%S'
    )
    logger = logging.getLogger(name)
    logger.setLevel(level)

    # Console handler
    ch = logging.StreamHandler()
    ch.setFormatter(formatter)
    logger.addHandler(ch)

    # File handler (opsiyonel)
    if log_file:
        fh = logging.FileHandler(log_file)
        fh.setFormatter(formatter)
        logger.addHandler(fh)

    return logger


logger = setup_logger("utils")

# ─────────────────────────────────────────────
# KLASÖR YÖNETİMİ
# ─────────────────────────────────────────────

def ensure_dirs(*paths):
    """Verilen klasör yollarını oluşturur."""
    for path in paths:
        Path(path).mkdir(parents=True, exist_ok=True)
        logger.debug(f"Klasör hazır: {path}")


PROJECT_ROOT = Path(__file__).parent.parent
DATA_RAW      = PROJECT_ROOT / "data" / "raw"
DATA_PROCESSED = PROJECT_ROOT / "data" / "processed"
DATA_MODELS   = PROJECT_ROOT / "data" / "models"
FIGURES_DIR   = PROJECT_ROOT / "figures"

ensure_dirs(DATA_RAW, DATA_PROCESSED, DATA_MODELS, FIGURES_DIR)

# ─────────────────────────────────────────────
# VERİ KAYDET / YÜKLE
# ─────────────────────────────────────────────

def save_dataframe(df: pd.DataFrame, filename: str, folder=DATA_PROCESSED):
    """DataFrame'i csv olarak kaydeder."""
    path = Path(folder) / filename
    df.to_csv(str(path) + ".csv", index=False)
    logger.info(f"DataFrame kaydedildi: {path} ({len(df)} satır)")


def load_dataframe(filename: str, folder=DATA_PROCESSED, fmt="csv") -> pd.DataFrame:
    path = Path(folder) / f"{filename}.{fmt}"
    if not path.exists():
        raise FileNotFoundError(f"Dosya bulunamadı: {path}")
    df = pd.read_csv(path)
    return df



def save_model(model, name: str):
    """Modeli joblib ile kaydeder."""
    path = DATA_MODELS / f"{name}.pkl"
    joblib.dump(model, path)
    logger.info(f"Model kaydedildi: {path}")


def load_model(name: str):
    """Kaydedilmiş modeli yükler."""
    path = DATA_MODELS / f"{name}.pkl"
    model = joblib.load(path)
    logger.info(f"Model yüklendi: {path}")
    return model


def save_json(data: dict, filename: str, folder=DATA_PROCESSED):
    """Dict'i JSON olarak kaydeder."""
    path = Path(folder) / f"{filename}.json"
    with open(path, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=2, ensure_ascii=False, default=str)
    logger.info(f"JSON kaydedildi: {path}")


def load_json(filename: str, folder=DATA_PROCESSED) -> dict:
    """JSON dosyasını yükler."""
    path = Path(folder) / f"{filename}.json"
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)

# ─────────────────────────────────────────────
# METRİK HESAPLAMA
# ─────────────────────────────────────────────

def compute_metrics(y_true, y_pred, y_prob=None, model_name="Model") -> dict:
    """
    Tüm sınıflandırma metriklerini hesaplar.
    
    Returns:
        dict: accuracy, precision, recall, f1, auc (varsa)
    """
    metrics = {
        "model": model_name,
        "accuracy":  round(accuracy_score(y_true, y_pred), 4),
        "precision": round(precision_score(y_true, y_pred, zero_division=0), 4),
        "recall":    round(recall_score(y_true, y_pred, zero_division=0), 4),
        "f1":        round(f1_score(y_true, y_pred, zero_division=0), 4),
    }
    if y_prob is not None:
        metrics["auc"] = round(roc_auc_score(y_true, y_prob), 4)
    
    logger.info(f"[{model_name}] ACC={metrics['accuracy']} F1={metrics['f1']}")
    return metrics


def metrics_table(results: list) -> pd.DataFrame:
    """
    Birden fazla model sonucunu tablo haline getirir.
    
    Args:
        results: compute_metrics'ten gelen dict listesi
    Returns:
        pd.DataFrame
    """
    df = pd.DataFrame(results)
    df = df.set_index("model")
    return df.round(4)

# ─────────────────────────────────────────────
# GÖRSELLEŞTİRME
# ─────────────────────────────────────────────

def plot_confusion_matrix(y_true, y_pred, model_name="Model", save=True):
    """Confusion matrix çizer ve kaydeder."""
    cm = confusion_matrix(y_true, y_pred)
    fig, ax = plt.subplots(figsize=(6, 5))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                xticklabels=["Legitimate", "Phishing"],
                yticklabels=["Legitimate", "Phishing"], ax=ax)
    ax.set_title(f"Confusion Matrix - {model_name}")
    ax.set_ylabel("Gerçek")
    ax.set_xlabel("Tahmin")
    plt.tight_layout()
    if save:
        path = FIGURES_DIR / f"cm_{model_name.lower().replace(' ', '_')}.png"
        plt.savefig(path, dpi=150)
        logger.info(f"Confusion matrix kaydedildi: {path}")
    plt.close()


def plot_roc_curve(y_true, y_prob_dict: dict, save=True):
    """
    Birden fazla model için ROC eğrisi çizer.
    
    Args:
        y_prob_dict: {"ModelAdı": y_prob_array} formatında dict
    """
    fig, ax = plt.subplots(figsize=(8, 6))
    for name, y_prob in y_prob_dict.items():
        fpr, tpr, _ = roc_curve(y_true, y_prob)
        auc = roc_auc_score(y_true, y_prob)
        ax.plot(fpr, tpr, label=f"{name} (AUC={auc:.3f})")
    ax.plot([0, 1], [0, 1], 'k--', label="Random")
    ax.set_xlabel("False Positive Rate")
    ax.set_ylabel("True Positive Rate")
    ax.set_title("ROC Curve Karşılaştırması")
    ax.legend()
    plt.tight_layout()
    if save:
        path = FIGURES_DIR / "roc_curves.png"
        plt.savefig(path, dpi=150)
        logger.info(f"ROC curve kaydedildi: {path}")
    plt.close()


def plot_metrics_comparison(results_before: list, results_after: list, 
                             title="Normal vs Adversarial Performans", save=True):
    """
    Adversarial saldırı öncesi ve sonrası metrik karşılaştırması.
    
    Args:
        results_before: normal test metrikleri listesi
        results_after:  adversarial test metrikleri listesi
    """
    df_before = pd.DataFrame(results_before).set_index("model")[["accuracy", "f1"]]
    df_after  = pd.DataFrame(results_after).set_index("model")[["accuracy", "f1"]]
    
    fig, axes = plt.subplots(1, 2, figsize=(14, 6))
    metric_names = ["accuracy", "f1"]
    colors = ["#2196F3", "#F44336"]

    for i, metric in enumerate(metric_names):
        x = np.arange(len(df_before))
        width = 0.35
        axes[i].bar(x - width/2, df_before[metric], width, label="Normal", color=colors[0], alpha=0.8)
        axes[i].bar(x + width/2, df_after[metric],  width, label="Adversarial", color=colors[1], alpha=0.8)
        axes[i].set_title(f"{metric.upper()} Karşılaştırması")
        axes[i].set_xticks(x)
        axes[i].set_xticklabels(df_before.index, rotation=15)
        axes[i].set_ylim(0, 1.1)
        axes[i].legend()
        axes[i].set_ylabel(metric.upper())

    fig.suptitle(title, fontsize=14, fontweight='bold')
    plt.tight_layout()
    if save:
        path = FIGURES_DIR / "adversarial_comparison.png"
        plt.savefig(path, dpi=150)
        logger.info(f"Karşılaştırma grafiği kaydedildi: {path}")
    plt.close()


def plot_feature_importance(feature_names, importances, model_name="Model", top_n=20, save=True):
    """Feature importance bar chart."""
    indices = np.argsort(importances)[::-1][:top_n]
    fig, ax = plt.subplots(figsize=(10, 6))
    ax.barh(
        [feature_names[i] for i in indices[::-1]],
        importances[indices[::-1]],
        color="#4CAF50", alpha=0.8
    )
    ax.set_title(f"Feature Importance - {model_name} (Top {top_n})")
    ax.set_xlabel("Importance")
    plt.tight_layout()
    if save:
        path = FIGURES_DIR / f"fi_{model_name.lower().replace(' ', '_')}.png"
        plt.savefig(path, dpi=150)
        logger.info(f"Feature importance kaydedildi: {path}")
    plt.close()

# ─────────────────────────────────────────────
# YARDIMCI FONKSIYONLAR
# ─────────────────────────────────────────────

def url_to_hash(url: str) -> str:
    """URL için MD5 hash üretir (tekrar kontrolü için)."""
    return hashlib.md5(url.encode()).hexdigest()


def print_section(title: str, char="═", width=60):
    """Notebook'larda bölüm başlıkları için."""
    line = char * width
    print(f"\n{line}")
    print(f"  {title}")
    print(f"{line}\n")


def timer(func):
    """Fonksiyon süresini ölçen decorator."""
    import time
    def wrapper(*args, **kwargs):
        start = time.time()
        result = func(*args, **kwargs)
        elapsed = time.time() - start
        logger.info(f"{func.__name__} tamamlandı: {elapsed:.2f}s")
        return result
    return wrapper


def summarize_dataset(df: pd.DataFrame, label_col="label"):
    """Dataset özeti yazdırır."""
    print_section("Dataset Özeti")
    print(f"Toplam örnek : {len(df):,}")
    print(f"Feature sayısı: {df.shape[1] - 1}")
    if label_col in df.columns:
        vc = df[label_col].value_counts()
        print(f"\nSınıf dağılımı:")
        for cls, cnt in vc.items():
            label = "Phishing" if cls == 1 else "Legitimate"
            print(f"  {label} ({cls}): {cnt:,} ({cnt/len(df)*100:.1f}%)")
    print(f"\nEksik değerler:\n{df.isnull().sum()[df.isnull().sum() > 0]}")
    print(f"\nVeri tipleri:\n{df.dtypes.value_counts()}")
