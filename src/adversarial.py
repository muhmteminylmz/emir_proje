"""
adversarial.py - Adversarial Saldırı Modülü
Feature-level manipülasyon ile ML modellerini kandırma
Tabular/URL veri için özelleştirilmiş adversarial örnekler
"""

import copy
import random
import warnings
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
from tqdm import tqdm

warnings.filterwarnings("ignore")

# ─────────────────────────────────────────────
# SABİTLER
# ─────────────────────────────────────────────

# Phishing → Meşru görünüme çekmeye çalışılan feature manipülasyonları
FEATURE_MANIPULATIONS = {
    # URL uzunluğunu kısalt (kısa URL'ler daha güvenilir görünür)
    "url_length":           {"delta": -15, "min": 10, "max": 2000},
    # Tire sayısını azalt
    "num_hyphens":          {"delta": -2, "min": 0, "max": 20},
    # @ işaretini kaldır
    "num_at":               {"delta": -1, "min": 0, "max": 5},
    # Digit oranını azalt
    "digit_ratio":          {"delta": -0.05, "min": 0.0, "max": 1.0},
    # Entropy'yi düşür
    "url_entropy":          {"delta": -0.3, "min": 0.0, "max": 6.0},
    "domain_entropy":       {"delta": -0.2, "min": 0.0, "max": 6.0},
    # Şüpheli keywords sayısını azalt
    "suspicious_keywords":  {"delta": -1, "min": 0, "max": 20},
    # Domain yaşını artır (sahte olarak)
    "domain_age_days":      {"delta": +500, "min": 0, "max": 5000},
    "domain_age_log":       {"delta": +1.0, "min": 0.0, "max": 10.0},
    # SSL var gibi göster
    "has_ssl":              {"set_to": 1},
    # HTTPS gibi göster
    "is_https":             {"set_to": 1},
    # Levenshtein'ı artır (markadan uzaklaştır)
    "min_brand_levenshtein": {"delta": +2, "min": 0, "max": 999},
    # Homoglyph'i gizle
    "has_homoglyph":        {"set_to": 0},
    "homoglyph_count":      {"set_to": 0},
    # Brand in domain'i gizle
    "brand_in_non_brand_domain": {"set_to": 0},
    # Domain uzunluğunu normalize et
    "domain_length":        {"delta": -5, "min": 3, "max": 100},
    # Subdomain sayısını azalt
    "num_subdomains":       {"delta": -1, "min": 0, "max": 10},
}

# Meşru → Phishing'e çeken manipülasyonlar (ters yön - savunma testi için)
REVERSE_MANIPULATIONS = {
    "url_length":           {"delta": +20, "min": 10, "max": 2000},
    "num_hyphens":          {"delta": +2, "min": 0, "max": 20},
    "digit_ratio":          {"delta": +0.05, "min": 0.0, "max": 1.0},
    "url_entropy":          {"delta": +0.3, "min": 0.0, "max": 6.0},
    "suspicious_keywords":  {"delta": +2, "min": 0, "max": 20},
    "has_ssl":              {"set_to": 0},
    "is_https":             {"set_to": 0},
    "domain_age_days":      {"delta": -300, "min": 0, "max": 5000},
}

# ─────────────────────────────────────────────
# TEMEL ADVERSARIAL SINIFI
# ─────────────────────────────────────────────

class AdversarialAttacker:
    """
    Feature-level adversarial saldırı uygulayan sınıf.
    
    Phishing örneklerini meşru görünümlü hale getirmeye çalışır.
    Modelin karar sınırını kandırmak için minimal feature değişimi yapar.
    """

    def __init__(self, manipulation_budget: int = 5, noise_std: float = 0.01):
        """
        Args:
            manipulation_budget: Kaç feature değiştirilebilir (maksimum)
            noise_std:           Gaussian gürültü standart sapması
        """
        self.budget    = manipulation_budget
        self.noise_std = noise_std

    # ─────────────────────────────────────────
    # TEMEL SALDIRI YÖNTEMLERİ
    # ─────────────────────────────────────────

    def minimal_feature_manipulation(self, X: np.ndarray, 
                                      feature_names: List[str],
                                      y: np.ndarray = None) -> np.ndarray:
        """
        Yöntem 1: Minimal Feature Manipülasyonu
        Sadece phishing örneklerinde belirli feature'ları değiştirir.
        Gerçek hayat senaryosu: Saldırgan URL'yi biraz değiştirir.
        
        Args:
            X:             Feature matrix (n_samples, n_features)
            feature_names: Feature isim listesi
            y:             Label dizisi (sadece phishing örnekleri hedeflenir)
        Returns:
            X_adv: Manipüle edilmiş feature matrix
        """
        X_adv = X.copy().astype(float)
        
        # Hangi örnekleri manipüle edeceğiz?
        if y is not None:
            target_indices = np.where(y == 1)[0]  # Sadece phishing
        else:
            target_indices = np.arange(len(X))
        
        feature_map = {name: i for i, name in enumerate(feature_names)}
        
        for idx in target_indices:
            # Budget kadar feature seç ve manipüle et
            applied = 0
            shuffled_features = list(FEATURE_MANIPULATIONS.items())
            random.shuffle(shuffled_features)
            
            for feat_name, rule in shuffled_features:
                if applied >= self.budget:
                    break
                if feat_name not in feature_map:
                    continue
                
                feat_idx = feature_map[feat_name]
                current  = X_adv[idx, feat_idx]
                
                if "set_to" in rule:
                    X_adv[idx, feat_idx] = rule["set_to"]
                    applied += 1
                elif "delta" in rule:
                    new_val = current + rule["delta"]
                    new_val = np.clip(new_val, rule["min"], rule["max"])
                    X_adv[idx, feat_idx] = new_val
                    applied += 1
        
        return X_adv

    def gaussian_noise_attack(self, X: np.ndarray, 
                               feature_names: List[str],
                               y: np.ndarray = None,
                               std_multiplier: float = 1.0) -> np.ndarray:
        """
        Yöntem 2: Gaussian Gürültü Saldırısı
        Her feature'a küçük bir gürültü ekler.
        Saldırgan modelin hangi feature'lara baktığını bilmez.
        
        Args:
            X:              Feature matrix
            feature_names:  Feature isimleri
            y:              Label dizisi
            std_multiplier: Gürültü şiddeti çarpanı
        Returns:
            X_adv: Gürültülü feature matrix
        """
        X_adv = X.copy().astype(float)
        
        # Feature standart sapmalarını hesapla
        feature_stds = X.std(axis=0)
        
        # Sadece phishing örneklerine gürültü ekle
        if y is not None:
            mask = (y == 1)
        else:
            mask = np.ones(len(X), dtype=bool)
        
        noise = np.random.normal(
            0, 
            feature_stds * self.noise_std * std_multiplier, 
            X_adv[mask].shape
        )
        X_adv[mask] += noise
        
        # Clip: negatif olamaz (sayısal tutarlılık)
        X_adv = np.clip(X_adv, 0, None)
        
        return X_adv

    def gradient_free_attack(self, X: np.ndarray,
                              feature_names: List[str],
                              model,
                              y: np.ndarray = None,
                              n_iterations: int = 10,
                              step_size: float = 0.1) -> np.ndarray:
        """
        Yöntem 3: Gradient-Free (Black-Box) Saldırı
        Modelin gradient bilgisine ihtiyaç duymaz.
        Random pertürbasyon ile model çıktısını izler.
        Tabular veri için FGSM alternatifi.
        
        Args:
            X:            Feature matrix
            model:        Sklearn uyumlu model (predict_proba desteklemeli)
            y:            Label dizisi
            n_iterations: Kaç iterasyon deneyecek
            step_size:    Her adımda ne kadar değişim
        Returns:
            X_adv: En başarılı adversarial örnekler
        """
        X_adv = X.copy().astype(float)
        
        if y is not None:
            target_indices = np.where(y == 1)[0]
        else:
            target_indices = np.arange(len(X))
        
        print(f"   Gradient-free saldırı: {len(target_indices)} hedef, {n_iterations} iterasyon")
        
        for idx in tqdm(target_indices, desc="   Black-box attack"):
            best_x   = X_adv[idx].copy()
            try:
                best_prob = model.predict_proba(best_x.reshape(1, -1))[0][1]
            except Exception:
                continue
            
            for _ in range(n_iterations):
                # Random pertürbasyon
                perturbation = np.random.normal(0, step_size, best_x.shape)
                candidate    = np.clip(best_x + perturbation, 0, None)
                
                try:
                    prob = model.predict_proba(candidate.reshape(1, -1))[0][1]
                except Exception:
                    continue
                
                # Phishing olasılığı düştüyse kabul et (saldırı başarılı)
                if prob < best_prob:
                    best_prob = prob
                    best_x    = candidate
            
            X_adv[idx] = best_x
        
        return X_adv

    def combined_attack(self, X: np.ndarray,
                         feature_names: List[str],
                         y: np.ndarray = None,
                         model=None) -> np.ndarray:
        """
        Yöntem 4: Kombine Saldırı
        Önce minimal manipulation, sonra gaussian noise.
        En güçlü saldırı kombinasyonu.
        """
        print("   [1/2] Minimal feature manipülasyonu uygulanıyor...")
        X_adv = self.minimal_feature_manipulation(X, feature_names, y)
        
        print("   [2/2] Gaussian gürültü ekleniyor...")
        X_adv = self.gaussian_noise_attack(X_adv, feature_names, y, std_multiplier=0.5)
        
        return X_adv


# ─────────────────────────────────────────────
# ADVERSARIAL DEĞERLENDİRME
# ─────────────────────────────────────────────

class AdversarialEvaluator:
    """
    Adversarial saldırı öncesi ve sonrası model performansını ölçer.
    """

    def evaluate(self, model, X_normal: np.ndarray, X_adv: np.ndarray,
                  y_true: np.ndarray, model_name: str = "Model") -> Dict:
        """
        Adversarial saldırı etkisini ölçer.
        
        Returns:
            dict: normal ve adversarial metrikler + düşüş miktarları
        """
        from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score
        
        # Normal tahminler
        y_pred_normal = model.predict(X_normal)
        acc_normal    = accuracy_score(y_true, y_pred_normal)
        f1_normal     = f1_score(y_true, y_pred_normal, zero_division=0)
        
        # Adversarial tahminler
        y_pred_adv    = model.predict(X_adv)
        acc_adv       = accuracy_score(y_true, y_pred_adv)
        f1_adv        = f1_score(y_true, y_pred_adv, zero_division=0)
        
        # Saldırı başarı oranı
        phishing_idx  = np.where(y_true == 1)[0]
        if len(phishing_idx) > 0:
            normal_correct  = (y_pred_normal[phishing_idx] == 1).sum()
            adv_correct     = (y_pred_adv[phishing_idx] == 1).sum()
            attack_success  = (normal_correct - adv_correct) / max(normal_correct, 1)
        else:
            attack_success  = 0.0
        
        result = {
            "model":              model_name,
            # Normal
            "acc_normal":         round(acc_normal, 4),
            "f1_normal":          round(f1_normal, 4),
            # Adversarial
            "acc_adversarial":    round(acc_adv, 4),
            "f1_adversarial":     round(f1_adv, 4),
            # Düşüş
            "acc_drop":           round(acc_normal - acc_adv, 4),
            "f1_drop":            round(f1_normal - f1_adv, 4),
            # Saldırı başarısı
            "attack_success_rate": round(attack_success, 4),
            # Robustness skoru (1 - düşüş)
            "robustness_score":   round(1 - (acc_normal - acc_adv), 4),
        }
        
        print(f"\n{'='*50}")
        print(f"  Adversarial Değerlendirme: {model_name}")
        print(f"{'='*50}")
        print(f"  Normal Accuracy    : {acc_normal:.4f}")
        print(f"  Adversarial Accuracy: {acc_adv:.4f}")
        print(f"  Accuracy Düşüşü    : {acc_normal - acc_adv:.4f} ({(acc_normal-acc_adv)/acc_normal*100:.1f}%)")
        print(f"  F1 Düşüşü          : {f1_normal - f1_adv:.4f}")
        print(f"  Saldırı Başarısı   : {attack_success:.2%}")
        print(f"  Robustness Skoru   : {1 - (acc_normal - acc_adv):.4f}")
        
        return result

    def compare_models(self, models: Dict, X_normal: np.ndarray, 
                        X_adv: np.ndarray, y_true: np.ndarray) -> pd.DataFrame:
        """
        Birden fazla modelin adversarial dayanıklılığını karşılaştırır.
        
        Args:
            models: {"model_adı": model_nesnesi} dict
        Returns:
            Karşılaştırma DataFrame
        """
        results = []
        for name, model in models.items():
            result = self.evaluate(model, X_normal, X_adv, y_true, name)
            results.append(result)
        
        df = pd.DataFrame(results).set_index("model")
        return df

    def feature_sensitivity_analysis(self, model, X: np.ndarray, 
                                      feature_names: List[str],
                                      y: np.ndarray,
                                      n_samples: int = 100) -> pd.DataFrame:
        """
        Her feature için tek tek saldırı yaparak
        hangi feature manipülasyonunun en çok etki ettiğini bulur.
        
        Returns:
            pd.DataFrame: feature → accuracy_drop sıralı liste
        """
        print("\n🔬 Feature sensitivity analizi başlıyor...")
        
        # Sadece phishing örnekleri
        phish_idx = np.where(y == 1)[0][:n_samples]
        X_phish   = X[phish_idx]
        y_phish   = y[phish_idx]
        
        from sklearn.metrics import accuracy_score
        baseline = accuracy_score(y_phish, model.predict(X_phish))
        
        sensitivity = []
        for feat_name, rule in FEATURE_MANIPULATIONS.items():
            if feat_name not in feature_names:
                continue
            
            feat_idx = feature_names.index(feat_name)
            X_modified = X_phish.copy().astype(float)
            
            if "set_to" in rule:
                X_modified[:, feat_idx] = rule["set_to"]
            elif "delta" in rule:
                X_modified[:, feat_idx] = np.clip(
                    X_modified[:, feat_idx] + rule["delta"],
                    rule["min"], rule["max"]
                )
            
            modified_acc = accuracy_score(y_phish, model.predict(X_modified))
            drop         = baseline - modified_acc
            
            sensitivity.append({
                "feature":      feat_name,
                "baseline_acc": round(baseline, 4),
                "modified_acc": round(modified_acc, 4),
                "acc_drop":     round(drop, 4),
                "impact":       "HIGH" if drop > 0.1 else ("MEDIUM" if drop > 0.05 else "LOW")
            })
        
        df = pd.DataFrame(sensitivity).sort_values("acc_drop", ascending=False)
        print(f"✅ Sensitivity analizi tamamlandı. En etkili feature: {df.iloc[0]['feature']}")
        return df


# ─────────────────────────────────────────────
# ATTACK RUNNER - Tüm saldırıları çalıştır
# ─────────────────────────────────────────────

class AttackRunner:
    """Tüm adversarial saldırıları organize eden sınıf."""

    def __init__(self, budget: int = 5):
        self.attacker   = AdversarialAttacker(manipulation_budget=budget)
        self.evaluator  = AdversarialEvaluator()

    def run_all_attacks(self, models: Dict, X_test: np.ndarray,
                         y_test: np.ndarray, feature_names: List[str]) -> Dict:
        """
        Tüm saldırı türlerini çalıştırır ve sonuçları döner.
        
        Args:
            models:        {"model_adı": model} dict
            X_test:        Test feature matrix
            y_test:        Test labels
            feature_names: Feature isimleri
        Returns:
            dict: Her saldırı türü için sonuçlar
        """
        all_results = {}
        
        print("\n" + "="*60)
        print("  ADVERSARIAL SALDIRILARI BAŞLIYOR")
        print("="*60)
        
        # Saldırı 1: Minimal Manipulation
        print("\n🔴 Saldırı 1: Minimal Feature Manipülasyonu")
        X_adv1 = self.attacker.minimal_feature_manipulation(
            X_test, feature_names, y_test
        )
        results1 = self.evaluator.compare_models(models, X_test, X_adv1, y_test)
        all_results["minimal_manipulation"] = results1
        
        # Saldırı 2: Gaussian Noise
        print("\n🟠 Saldırı 2: Gaussian Gürültü Saldırısı")
        X_adv2 = self.attacker.gaussian_noise_attack(
            X_test, feature_names, y_test, std_multiplier=2.0
        )
        results2 = self.evaluator.compare_models(models, X_test, X_adv2, y_test)
        all_results["gaussian_noise"] = results2
        
        # Saldırı 3: Kombine
        print("\n🔴 Saldırı 3: Kombine Saldırı (En Güçlü)")
        X_adv3 = self.attacker.combined_attack(X_test, feature_names, y_test)
        results3 = self.evaluator.compare_models(models, X_test, X_adv3, y_test)
        all_results["combined"] = results3
        
        # Adversarial örnekleri sakla
        all_results["adversarial_examples"] = {
            "minimal":  X_adv1,
            "gaussian": X_adv2,
            "combined": X_adv3,
        }
        all_results["feature_names"] = feature_names
        
        print("\n✅ Tüm saldırılar tamamlandı!")
        return all_results

    def summarize(self, all_results: Dict) -> pd.DataFrame:
        """Tüm saldırı sonuçlarını özet tablo olarak döner."""
        rows = []
        for attack_name, result_df in all_results.items():
            if attack_name in ["adversarial_examples", "feature_names"]:
                continue
            for model_name, row in result_df.iterrows():
                rows.append({
                    "attack":         attack_name,
                    "model":          model_name,
                    "acc_drop":       row.get("acc_drop", 0),
                    "f1_drop":        row.get("f1_drop", 0),
                    "attack_success": row.get("attack_success_rate", 0),
                    "robustness":     row.get("robustness_score", 0),
                })
        return pd.DataFrame(rows)


if __name__ == "__main__":
    # Basit test
    np.random.seed(42)
    n_samples, n_features = 100, 20
    X = np.random.rand(n_samples, n_features)
    y = np.array([1]*50 + [0]*50)
    feature_names = [list(FEATURE_MANIPULATIONS.keys())[i % len(FEATURE_MANIPULATIONS)] 
                     for i in range(n_features)]
    
    attacker = AdversarialAttacker(manipulation_budget=5)
    X_adv = attacker.minimal_feature_manipulation(X, feature_names, y)
    
    print(f"Orijinal X[0]: {X[0, :5]}")
    print(f"Adversarial  : {X_adv[0, :5]}")
    print(f"Fark         : {(X != X_adv).sum()} değer değişti")
