"""
defense.py - Adversarial Savunma Mekanizmaları
Adversarial Training, Feature Smoothing, Noise Injection, Ensemble Defense
"""

import warnings
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, ClassifierMixin
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.utils import resample
from tqdm import tqdm

warnings.filterwarnings("ignore")

# ─────────────────────────────────────────────
# SAVUNMA 1: ADVERSARIAL TRAINING
# ─────────────────────────────────────────────

class AdversarialTrainer:
    """
    Adversarial Training: Modeli hem normal hem adversarial örneklerle eğitir.
    
    Fikir: Eğer model adversarial örnekler görerek öğrenirse,
    test sırasında bu saldırılara daha dayanıklı olur.
    """

    def __init__(self, base_model, adversarial_ratio: float = 0.3):
        """
        Args:
            base_model:          Sklearn uyumlu base model
            adversarial_ratio:   Eğitim setinin ne kadarı adversarial olacak (0-1)
        """
        self.base_model         = base_model
        self.adversarial_ratio  = adversarial_ratio
        self.model_             = None
        self.is_fitted_         = False

    def fit(self, X_train: np.ndarray, y_train: np.ndarray,
            feature_names: List[str], 
            n_augmentation_rounds: int = 3) -> 'AdversarialTrainer':
        """
        Adversarial augmented training.
        
        Args:
            X_train:              Normal eğitim verisi
            y_train:              Etiketler
            feature_names:        Feature isimleri
            n_augmentation_rounds: Kaç tur adversarial örnek üretilsin
        """
        from src.adversarial import AdversarialAttacker
        
        print(f"🛡️  Adversarial Training başlıyor...")
        print(f"   Base model: {type(self.base_model).__name__}")
        print(f"   Adversarial ratio: {self.adversarial_ratio}")
        print(f"   Augmentation rounds: {n_augmentation_rounds}")
        
        attacker = AdversarialAttacker(manipulation_budget=3)
        
        X_augmented = [X_train]
        y_augmented = [y_train]
        
        # Her tur için adversarial örnekler üret
        for round_i in range(n_augmentation_rounds):
            # Phishing örneklerini adversarial olarak çevir
            X_adv = attacker.minimal_feature_manipulation(
                X_train, feature_names, y_train
            )
            # Gaussian gürültü ekle
            X_adv_noisy = attacker.gaussian_noise_attack(
                X_adv, feature_names, y_train, std_multiplier=0.5 * (round_i + 1)
            )
            
            # Sadece adversarial ratio kadar al
            n_adv = int(len(X_train) * self.adversarial_ratio)
            idx   = np.random.choice(len(X_adv_noisy), n_adv, replace=False)
            
            X_augmented.append(X_adv_noisy[idx])
            y_augmented.append(y_train[idx])  # Label değişmez! Hâlâ phishing
        
        # Birleştir ve karıştır
        X_final = np.vstack(X_augmented)
        y_final = np.concatenate(y_augmented)
        shuffle_idx = np.random.permutation(len(X_final))
        X_final = X_final[shuffle_idx]
        y_final = y_final[shuffle_idx]
        
        print(f"   Augmented dataset boyutu: {len(X_final)} (orijinal: {len(X_train)})")
        
        # Modeli eğit
        import copy
        self.model_ = copy.deepcopy(self.base_model)
        self.model_.fit(X_final, y_final)
        self.is_fitted_ = True
        
        print(f"✅ Adversarial training tamamlandı")
        return self

    def predict(self, X):
        return self.model_.predict(X)

    def predict_proba(self, X):
        return self.model_.predict_proba(X)

    def score(self, X, y):
        from sklearn.metrics import accuracy_score
        return accuracy_score(y, self.predict(X))


# ─────────────────────────────────────────────
# SAVUNMA 2: FEATURE SMOOTHING
# ─────────────────────────────────────────────

class FeatureSmoothing:
    """
    Feature Smoothing: Extreme değerleri yumuşatır.
    
    Adversarial saldırılar genellikle feature değerlerini
    extreme noktalara iterek çalışır. Smoothing bunu önler.
    """

    def __init__(self, method: str = "percentile", lower: float = 5.0, upper: float = 95.0):
        """
        Args:
            method:  "percentile" veya "zscore"
            lower:   Alt percentile clip
            upper:   Üst percentile clip
        """
        self.method  = method
        self.lower   = lower
        self.upper   = upper
        self.bounds_ = None  # Fit sırasında öğrenilir

    def fit(self, X: np.ndarray) -> 'FeatureSmoothing':
        """Eğitim verisi üzerinden smoothing sınırlarını öğren."""
        if self.method == "percentile":
            self.bounds_ = {
                "low":  np.percentile(X, self.lower, axis=0),
                "high": np.percentile(X, self.upper, axis=0),
            }
        elif self.method == "zscore":
            self.bounds_ = {
                "mean": X.mean(axis=0),
                "std":  X.std(axis=0) + 1e-8,
                "z_threshold": 3.0,
            }
        print(f"✅ Feature Smoothing fit edildi ({self.method})")
        return self

    def transform(self, X: np.ndarray) -> np.ndarray:
        """Feature değerlerini clip ederek smoothing uygular."""
        if self.bounds_ is None:
            raise ValueError("Önce fit() çağrılmalı!")
        
        X_smooth = X.copy().astype(float)
        
        if self.method == "percentile":
            for j in range(X.shape[1]):
                X_smooth[:, j] = np.clip(X_smooth[:, j], 
                                          self.bounds_["low"][j], 
                                          self.bounds_["high"][j])
        
        elif self.method == "zscore":
            z = (X_smooth - self.bounds_["mean"]) / self.bounds_["std"]
            threshold = self.bounds_["z_threshold"]
            # Extreme z-score → mean'e çek
            extreme = np.abs(z) > threshold
            X_smooth[extreme] = (
                np.broadcast_to(self.bounds_["mean"], X_smooth.shape)[extreme]
            )
        
        return X_smooth

    def fit_transform(self, X: np.ndarray) -> np.ndarray:
        return self.fit(X).transform(X)


# ─────────────────────────────────────────────
# SAVUNMA 3: ROBUST MODEL WRAPPER
# ─────────────────────────────────────────────

class RobustClassifier(BaseEstimator, ClassifierMixin):
    """
    Tüm savunma mekanizmalarını bir araya getiren wrapper.
    Sklearn API ile uyumlu.
    """

    def __init__(self, base_model, 
                 use_smoothing: bool = True,
                 use_adversarial_training: bool = True,
                 adversarial_ratio: float = 0.3,
                 smoothing_method: str = "percentile"):
        self.base_model               = base_model
        self.use_smoothing            = use_smoothing
        self.use_adversarial_training = use_adversarial_training
        self.adversarial_ratio        = adversarial_ratio
        self.smoothing_method         = smoothing_method
        
        self.smoother_    = None
        self.adv_trainer_ = None
        self.classes_     = np.array([0, 1])

    def fit(self, X: np.ndarray, y: np.ndarray, 
            feature_names: List[str] = None) -> 'RobustClassifier':
        """Tüm savunma mekanizmalarıyla modeli eğitir."""
        print(f"\n🛡️  RobustClassifier eğitiliyor...")
        print(f"   Smoothing: {self.use_smoothing}")
        print(f"   Adversarial Training: {self.use_adversarial_training}")
        
        X_processed = X.copy()
        
        # 1. Feature Smoothing
        if self.use_smoothing:
            self.smoother_ = FeatureSmoothing(method=self.smoothing_method)
            X_processed    = self.smoother_.fit_transform(X_processed)
        
        # 2. Adversarial Training
        if self.use_adversarial_training and feature_names is not None:
            self.adv_trainer_ = AdversarialTrainer(
                self.base_model, 
                adversarial_ratio=self.adversarial_ratio
            )
            self.adv_trainer_.fit(X_processed, y, feature_names)
        else:
            import copy
            self.final_model_ = copy.deepcopy(self.base_model)
            self.final_model_.fit(X_processed, y)
        
        print(f"✅ RobustClassifier hazır")
        return self

    def _preprocess(self, X: np.ndarray) -> np.ndarray:
        """Tahmin öncesi ön işlem."""
        X_proc = X.copy()
        if self.use_smoothing and self.smoother_ is not None:
            X_proc = self.smoother_.transform(X_proc)
        return X_proc

    def predict(self, X: np.ndarray) -> np.ndarray:
        X_proc = self._preprocess(X)
        if self.adv_trainer_ is not None:
            return self.adv_trainer_.predict(X_proc)
        return self.final_model_.predict(X_proc)

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        X_proc = self._preprocess(X)
        if self.adv_trainer_ is not None:
            return self.adv_trainer_.predict_proba(X_proc)
        return self.final_model_.predict_proba(X_proc)

    def score(self, X, y):
        from sklearn.metrics import accuracy_score
        return accuracy_score(y, self.predict(X))


# ─────────────────────────────────────────────
# SAVUNMA 4: ENSEMBLE DEFENSE
# ─────────────────────────────────────────────

class EnsembleDefense:
    """
    Ensemble tabanlı savunma.
    Birden fazla model oylama yaparak saldırıya direnir.
    Saldırgan tek modeli kandırabilir ama ensemble'ı kandırmak çok daha zor.
    """

    def __init__(self, models: Dict, voting: str = "soft"):
        """
        Args:
            models:  {"model_adı": model} dict (predict_proba desteklemeli)
            voting:  "soft" (olasılık ortalaması) veya "hard" (çoğunluk oylaması)
        """
        self.models = models
        self.voting = voting

    def predict(self, X: np.ndarray) -> np.ndarray:
        """Ensemble tahmin."""
        if self.voting == "soft":
            probs = self.predict_proba(X)
            return (probs[:, 1] >= 0.5).astype(int)
        else:
            predictions = np.array([m.predict(X) for m in self.models.values()])
            return (predictions.mean(axis=0) >= 0.5).astype(int)

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        """Soft voting ile olasılık tahmini."""
        all_probs = []
        for name, model in self.models.items():
            try:
                prob = model.predict_proba(X)
                all_probs.append(prob)
            except Exception as e:
                print(f"  {name} predict_proba hatası: {e}")
        
        if not all_probs:
            return np.zeros((len(X), 2))
        
        avg_probs = np.mean(all_probs, axis=0)
        return avg_probs

    def score(self, X, y):
        from sklearn.metrics import accuracy_score
        return accuracy_score(y, self.predict(X))


# ─────────────────────────────────────────────
# SAVUNMA DEĞERLENDİRME
# ─────────────────────────────────────────────

class DefenseEvaluator:
    """Savunma mekanizmalarının etkinliğini ölçer."""

    def compare_before_after(self, 
                               baseline_models: Dict,
                               defended_models: Dict,
                               X_normal: np.ndarray,
                               X_adv: np.ndarray,
                               y_true: np.ndarray) -> pd.DataFrame:
        """
        Savunma öncesi ve sonrası karşılaştırma tablosu.
        
        Returns:
            pd.DataFrame: Detaylı karşılaştırma
        """
        from sklearn.metrics import accuracy_score, f1_score
        
        rows = []
        
        for model_name in baseline_models.keys():
            baseline = baseline_models[model_name]
            defended = defended_models.get(model_name)
            
            # Baseline - normal
            y_pred_base_normal = baseline.predict(X_normal)
            acc_base_normal    = accuracy_score(y_true, y_pred_base_normal)
            f1_base_normal     = f1_score(y_true, y_pred_base_normal, zero_division=0)
            
            # Baseline - adversarial
            y_pred_base_adv    = baseline.predict(X_adv)
            acc_base_adv       = accuracy_score(y_true, y_pred_base_adv)
            f1_base_adv        = f1_score(y_true, y_pred_base_adv, zero_division=0)
            
            row = {
                "model":              model_name,
                "baseline_normal_acc": round(acc_base_normal, 4),
                "baseline_adv_acc":   round(acc_base_adv, 4),
                "baseline_drop":      round(acc_base_normal - acc_base_adv, 4),
            }
            
            if defended is not None:
                # Defended - normal
                y_pred_def_normal = defended.predict(X_normal)
                acc_def_normal    = accuracy_score(y_true, y_pred_def_normal)
                
                # Defended - adversarial
                y_pred_def_adv    = defended.predict(X_adv)
                acc_def_adv       = accuracy_score(y_true, y_pred_def_adv)
                
                row.update({
                    "defended_normal_acc":  round(acc_def_normal, 4),
                    "defended_adv_acc":     round(acc_def_adv, 4),
                    "defended_drop":        round(acc_def_normal - acc_def_adv, 4),
                    "defense_improvement":  round(acc_def_adv - acc_base_adv, 4),
                })
            
            rows.append(row)
        
        df = pd.DataFrame(rows).set_index("model")
        
        print("\n" + "="*70)
        print("  SAVUNMA KARŞILAŞTIRMA TABLOSU")
        print("="*70)
        print(df.to_string())
        
        return df

    def robustness_summary(self, results_df: pd.DataFrame) -> None:
        """Özet robustness raporu yazdırır."""
        print("\n📊 ROBUSTNESS ÖZETİ")
        print("-"*50)
        
        if "baseline_drop" in results_df.columns and "defended_drop" in results_df.columns:
            for model in results_df.index:
                b_drop = results_df.loc[model, "baseline_drop"]
                d_drop = results_df.loc[model, "defended_drop"]
                imp    = results_df.loc[model, "defense_improvement"]
                print(f"  {model}:")
                print(f"    Saldırı öncesi drop: {b_drop:.4f}")
                print(f"    Savunma sonrası drop: {d_drop:.4f}")
                print(f"    İyileşme: +{imp:.4f} ({imp*100:.1f}%)")
                print()


# ─────────────────────────────────────────────
# TAM SAVUNMA PIPELINE
# ─────────────────────────────────────────────

def build_defended_models(baseline_models: Dict,
                           X_train: np.ndarray,
                           y_train: np.ndarray,
                           feature_names: List[str]) -> Dict:
    """
    Her baseline model için savunmalı versiyonunu oluşturur.
    
    Args:
        baseline_models: {"rf": rf_model, "xgb": xgb_model, ...}
        X_train:         Eğitim feature matrix
        y_train:         Eğitim etiketleri
        feature_names:   Feature isimleri listesi
    Returns:
        dict: {"rf_robust": ..., "xgb_robust": ..., "ensemble": ...}
    """
    import copy
    defended = {}
    
    print("\n" + "="*60)
    print("  SAVUNMA MODELLERİ EĞİTİLİYOR")
    print("="*60)
    
    for model_name, model in baseline_models.items():
        print(f"\n🛡️  {model_name} için robust model eğitiliyor...")
        
        robust = RobustClassifier(
            base_model=copy.deepcopy(model),
            use_smoothing=True,
            use_adversarial_training=True,
            adversarial_ratio=0.25,
        )
        robust.fit(X_train, y_train, feature_names=feature_names)
        defended[f"{model_name}_robust"] = robust
    
    # Ensemble defense (orijinal modeller ile)
    print("\n🛡️  Ensemble defense oluşturuluyor...")
    ensemble = EnsembleDefense(baseline_models, voting="soft")
    defended["ensemble"] = ensemble
    
    print("\n✅ Tüm savunma modelleri hazır")
    return defended


if __name__ == "__main__":
    # Test
    from sklearn.ensemble import RandomForestClassifier
    import numpy as np
    
    np.random.seed(42)
    X = np.random.rand(200, 15)
    y = np.array([1]*100 + [0]*100)
    feature_names = [f"feature_{i}" for i in range(15)]
    
    rf = RandomForestClassifier(n_estimators=50, random_state=42)
    rf.fit(X, y)
    
    robust_rf = RobustClassifier(rf, use_smoothing=True, use_adversarial_training=False)
    robust_rf.fit(X, y, feature_names=feature_names)
    
    from sklearn.metrics import accuracy_score
    print(f"RF Accuracy: {accuracy_score(y, rf.predict(X)):.4f}")
    print(f"Robust RF Accuracy: {accuracy_score(y, robust_rf.predict(X)):.4f}")
