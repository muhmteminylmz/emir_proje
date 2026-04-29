"""
data_collector.py - Veri Toplama Modülü
PhishTank API + Kaggle CSV birleştirme
WHOIS / SSL metadata çekme
"""

import os
import re
import ssl
import time
import socket
import random
import hashlib
import requests
import warnings
import subprocess
from datetime import datetime, timezone
from pathlib import Path
from typing import Optional, Tuple

import numpy as np
import pandas as pd
import whois
import tldextract
from tqdm import tqdm

warnings.filterwarnings("ignore")

# Proje kökü
PROJECT_ROOT = Path(__file__).parent.parent
DATA_RAW     = PROJECT_ROOT / "data" / "raw"
DATA_RAW.mkdir(parents=True, exist_ok=True)

# ─────────────────────────────────────────────
# SABİTLER
# ─────────────────────────────────────────────

PHISHTANK_API   = "https://data.phishtank.com/data/online-valid.csv"
LEGITIMATE_URLS = [
    "https://www.google.com", "https://www.youtube.com", "https://www.facebook.com",
    "https://www.amazon.com", "https://www.wikipedia.org", "https://www.twitter.com",
    "https://www.instagram.com", "https://www.linkedin.com", "https://www.reddit.com",
    "https://www.netflix.com", "https://www.microsoft.com", "https://www.apple.com",
    "https://www.github.com", "https://www.stackoverflow.com", "https://www.medium.com",
    "https://www.nytimes.com", "https://www.bbc.com", "https://www.cnn.com",
    "https://www.paypal.com", "https://www.ebay.com", "https://www.walmart.com",
    "https://www.dropbox.com", "https://www.spotify.com", "https://www.twitch.tv",
    "https://www.adobe.com", "https://www.oracle.com", "https://www.ibm.com",
    "https://www.salesforce.com", "https://www.zoom.us", "https://www.slack.com",
    "https://www.trello.com", "https://www.notion.so", "https://www.canva.com",
    "https://www.cloudflare.com", "https://www.digitalocean.com", "https://www.heroku.com",
    # HTTP kullanan meşru siteler - modelin HTTP=phishing bias'ını önlemek için gerekli
    "http://www.bbc.co.uk", "http://www.reuters.com", "http://www.archive.org",
    "http://www.gnu.org", "http://www.debian.org", "http://www.apache.org",
    "http://www.mit.edu", "http://www.harvard.edu", "http://www.cornell.edu",
    "http://www.w3.org", "http://www.ietf.org", "http://www.python.org",
]

REQUEST_HEADERS = {
    "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36"
}

# ─────────────────────────────────────────────
# PHISHTANK VERİ TOPLAMA
# ─────────────────────────────────────────────

class PhishTankCollector:
    """PhishTank'tan phishing URL'leri toplar."""

    def __init__(self, api_key: str = None, max_samples: int = 5000):
        self.api_key   = api_key
        self.max_samples = max_samples

    def fetch(self) -> pd.DataFrame:
        """
        PhishTank CSV'sini indirir ve parse eder.
        API key yoksa demo dataset döner.
        """
        print("📥 PhishTank verisi çekiliyor...")
        try:
            url = PHISHTANK_API
            if self.api_key:
                url = f"https://data.phishtank.com/data/{self.api_key}/online-valid.csv"
            
            resp = requests.get(url, headers=REQUEST_HEADERS, timeout=30)
            resp.raise_for_status()
            
            from io import StringIO
            df = pd.read_csv(StringIO(resp.text))
            df = df[["url", "submission_time", "verified", "online"]].copy()
            df = df[df["verified"] == "yes"].copy()
            df["label"] = 1
            df["source"] = "phishtank"
            df = df.rename(columns={"url": "url", "submission_time": "timestamp"})
            df["timestamp"] = pd.to_datetime(df["timestamp"], utc=True, errors="coerce")
            df = df.dropna(subset=["timestamp"])
            df = df.head(self.max_samples)
            print(f"✅ PhishTank: {len(df)} phishing URL alındı")
            return df[["url", "timestamp", "label", "source"]]
        
        except Exception as e:
            print(f"⚠️  PhishTank API hatası ({e}), demo veri kullanılıyor...")
            return self._generate_demo_phishing()

    def _generate_demo_phishing(self) -> pd.DataFrame:
        """API erişimi yoksa demo phishing URL'leri üretir."""
        patterns = [
            "paypa1-{}.login-secure.com/verify",
            "faceb00k-{}.secure-login.net/auth",
            "amazon-security-{}.xyz/update",
            "apple-id-{}.phish-verify.com/signin",
            "microsofft-{}.login-page.net/account",
            "instagram-{}.support-help.com/verify",
            "netflix-{}.update-billing.net/payment",
            "bankofamerica-{}.secure-portal.xyz/login",
            "chase-{}.account-verify.net/confirm",
            "wellsfargo-{}.secure-update.com/banking",
        ]
        urls, timestamps = [], []
        np.random.seed(42)

        # 2023 ve 2024 için dengeli üretim
        # Gerçek phishing siteleri de HTTPS kullanabiliyor (Let's Encrypt ücretsiz sertifika)
        # Eğitim verisinde sadece http:// kullanmak modele HTTP=phishing bias'ı yaratır
        schemes = ["http://", "https://"]
        for year in [2023, 2024]:
            for i in range(500):
                pattern = random.choice(patterns)
                scheme = random.choice(schemes)
                url = f"{scheme}{pattern.format(random.randint(1000, 9999))}"
                urls.append(url)
                month = random.randint(1, 12)
                day   = random.randint(1, 28)
                ts    = pd.Timestamp(year=year, month=month, day=day, tz="UTC")
                timestamps.append(ts)
        
        df = pd.DataFrame({
            "url": urls,
            "timestamp": timestamps,
            "label": 1,
            "source": "demo_phishtank"
        })
        print(f"✅ Demo PhishTank: {len(df)} phishing URL üretildi")
        return df


# ─────────────────────────────────────────────
# KAGGLE VERİ YÜKLEME
# ─────────────────────────────────────────────

class KaggleLoader:
    """
    Kaggle Phishing Dataset yükler.
    Beklenen sütunlar: url, label (0/1 veya phishing/legitimate)
    """

    KNOWN_DATASETS = {
        "web_page_phishing": {
            "url_col": "url",
            "label_col": "status",
            "phishing_val": "phishing",
        },
        "phishing_site_urls": {
            "url_col": "URL",
            "label_col": "Label",
            "phishing_val": "bad",
        },
        "dataset_phishing": {
            "url_col": "url",
            "label_col": "status",
            "phishing_val": "phishing",
        },
    }

    def load(self, csv_path: str) -> pd.DataFrame:
        """CSV'yi yükler ve standart formata çevirir."""
        print(f"📂 Kaggle dataset yükleniyor: {csv_path}")
        df = pd.read_csv(csv_path)
        print(f"   Sütunlar: {list(df.columns)}")
        print(f"   Boyut: {df.shape}")
        
        # Otomatik sütun tespiti
        url_col   = self._find_col(df, ["url", "URL", "link", "domain"])
        label_col = self._find_col(df, ["label", "Label", "status", "Status", "class", "Class", "type"])
        
        if not url_col or not label_col:
            raise ValueError(f"URL veya label sütunu bulunamadı. Sütunlar: {list(df.columns)}")
        
        df = df[[url_col, label_col]].copy()
        df.columns = ["url", "label_raw"]
        
        # Label normalizasyonu
        df["label"] = df["label_raw"].apply(self._normalize_label)
        df = df.dropna(subset=["label"])
        df["label"] = df["label"].astype(int)
        
        # Timestamp yoksa random ata (2023-2024 arası)
        df["timestamp"] = self._assign_timestamps(len(df))
        df["source"] = "kaggle"
        
        result = df[["url", "timestamp", "label", "source"]]
        print(f"✅ Kaggle: {len(result)} URL yüklendi "
              f"(Phishing: {result['label'].sum()}, Legitimate: {(result['label']==0).sum()})")
        return result

    def _find_col(self, df: pd.DataFrame, candidates: list) -> Optional[str]:
        for c in candidates:
            if c in df.columns:
                return c
            for col in df.columns:
                if col.lower() == c.lower():
                    return col
        return None

    def _normalize_label(self, val) -> Optional[int]:
        if isinstance(val, (int, float)):
            return int(val) if val in [0, 1] else None
        val = str(val).strip().lower()
        if val in ["phishing", "bad", "1", "malicious", "spam", "phish"]:
            return 1
        if val in ["legitimate", "good", "0", "benign", "ham", "safe"]:
            return 0
        return None

    def _assign_timestamps(self, n: int) -> pd.Series:
        """Veri için 2023-2024 arası random timestamp atar."""
        start = pd.Timestamp("2023-01-01", tz="UTC")
        end   = pd.Timestamp("2024-12-31", tz="UTC")
        delta = (end - start).days
        random_days = np.random.randint(0, delta, size=n)
        timestamps = [start + pd.Timedelta(days=int(d)) for d in random_days]
        return pd.Series(timestamps)


# ─────────────────────────────────────────────
# VERİ BİRLEŞTİRME
# ─────────────────────────────────────────────

class DatasetMerger:
    """PhishTank ve Kaggle verilerini birleştirir."""

    def merge(self, phishtank_df: pd.DataFrame, kaggle_df: pd.DataFrame,
              max_total: int = 10000) -> pd.DataFrame:
        """
        İki kaynağı birleştir, tekrarlananları kaldır, dengeli hale getir.
        """
        print("\n🔗 Dataset'ler birleştiriliyor...")
        combined = pd.concat([phishtank_df, kaggle_df], ignore_index=True)
        
        # URL normalize
        combined["url"] = combined["url"].str.strip().str.lower()
        
        # Duplicate kaldır
        before = len(combined)
        combined = combined.drop_duplicates(subset=["url"])
        print(f"   Duplicate kaldırıldı: {before - len(combined)}")
        
        # Geçersiz URL kaldır
        combined = combined[combined["url"].str.startswith(("http://", "https://"))]
        
        # Sınıf dengesi
        combined = self._balance_classes(combined, max_total)
        
        # Sırala
        combined = combined.sort_values("timestamp").reset_index(drop=True)
        
        print(f"✅ Birleşik dataset: {len(combined)} URL")
        print(f"   Phishing:    {combined['label'].sum():,}")
        print(f"   Legitimate:  {(combined['label']==0).sum():,}")
        return combined

    def _balance_classes(self, df: pd.DataFrame, max_total: int) -> pd.DataFrame:
        """Her sınıftan eşit sayıda örnek alır."""
        n_each = min(max_total // 2, df['label'].value_counts().min())
        phishing   = df[df['label'] == 1].sample(n=n_each, random_state=42)
        legitimate = df[df['label'] == 0].sample(n=n_each, random_state=42)
        return pd.concat([phishing, legitimate], ignore_index=True)

    def add_legitimate_urls(self, df: pd.DataFrame) -> pd.DataFrame:
        """Bilinen meşru siteleri ekler (Kaggle'da legitimate az ise)."""
        legit_count = (df['label'] == 0).sum()
        phish_count = (df['label'] == 1).sum()
        
        if legit_count >= phish_count * 0.8:
            return df  # Zaten dengeli
        
        needed = phish_count - legit_count
        print(f"⚖️  {needed} meşru URL ekleniyor...")
        
        extra_rows = []
        for url in LEGITIMATE_URLS[:needed]:
            extra_rows.append({
                "url": url.lower(),
                "timestamp": pd.Timestamp("2023-06-15", tz="UTC"),
                "label": 0,
                "source": "known_legitimate"
            })
        
        extra_df = pd.DataFrame(extra_rows)
        return pd.concat([df, extra_df], ignore_index=True)


# ─────────────────────────────────────────────
# WHOIS / SSL METADATA
# ─────────────────────────────────────────────

class MetadataCollector:
    """Domain WHOIS ve SSL bilgilerini çeker."""

    def __init__(self, timeout: int = 5, max_workers: int = 4):
        self.timeout     = timeout
        self.max_workers = max_workers

    def collect_all(self, df: pd.DataFrame, 
                    sample_size: int = None) -> pd.DataFrame:
        """
        DataFrame'deki her URL için WHOIS + SSL metadata çeker.
        
        Args:
            df: url sütunu olan DataFrame
            sample_size: None ise tüm dataset, int ise örneklem
        Returns:
            metadata sütunları eklenmiş DataFrame
        """
        if sample_size:
            df = df.head(sample_size).copy()
        
        print(f"\n🔍 Metadata toplanıyor ({len(df)} URL)...")
        print("   (Bu işlem uzun sürebilir, lütfen bekleyin)")
        
        metadata_list = []
        for _, row in tqdm(df.iterrows(), total=len(df), desc="Metadata"):
            meta = self.get_domain_metadata(row["url"])
            metadata_list.append(meta)
        
        meta_df = pd.DataFrame(metadata_list)
        result  = pd.concat([df.reset_index(drop=True), meta_df], axis=1)
        print(f"✅ Metadata tamamlandı")
        return result

    def get_domain_metadata(self, url: str) -> dict:
        """
        Tek URL için metadata çeker.
        
        Returns:
            dict: domain_age_days, has_ssl, ssl_issuer, 
                  registrar, whois_available, domain_registered_days
        """
        meta = {
            "domain_age_days": -1,
            "has_ssl": 0,
            "ssl_issuer": "unknown",
            "ssl_issuer_is_free": 0,
            "registrar": "unknown",
            "whois_available": 0,
            "registration_period_days": -1,
            "country": "unknown",
        }
        
        try:
            ext    = tldextract.extract(url)
            domain = f"{ext.domain}.{ext.suffix}"
            
            # WHOIS
            whois_meta = self._get_whois(domain)
            meta.update(whois_meta)
            
            # SSL
            ssl_meta = self._get_ssl(domain)
            meta.update(ssl_meta)
        
        except Exception:
            pass
        
        return meta

    def _get_whois(self, domain: str) -> dict:
        """WHOIS bilgisini çeker."""
        result = {
            "domain_age_days": -1,
            "registrar": "unknown",
            "whois_available": 0,
            "registration_period_days": -1,
            "country": "unknown",
        }
        try:
            w = whois.whois(domain)
            result["whois_available"] = 1
            
            # Domain yaşı
            creation = w.creation_date
            if isinstance(creation, list):
                creation = creation[0]
            if creation:
                if creation.tzinfo is None:
                    creation = creation.replace(tzinfo=timezone.utc)
                now = datetime.now(timezone.utc)
                result["domain_age_days"] = max(0, (now - creation).days)
            
            # Kayıt süresi
            expiry = w.expiration_date
            if isinstance(expiry, list):
                expiry = expiry[0]
            if expiry and creation:
                result["registration_period_days"] = max(0, (expiry - creation).days)
            
            # Registrar
            if w.registrar:
                result["registrar"] = str(w.registrar)[:50]
            
            # Ülke
            if hasattr(w, "country") and w.country:
                result["country"] = str(w.country)
        
        except Exception:
            pass
        return result

    def _get_ssl(self, domain: str) -> dict:
        """SSL sertifika bilgisini çeker."""
        result = {
            "has_ssl": 0,
            "ssl_issuer": "unknown",
            "ssl_issuer_is_free": 0,
        }
        try:
            ctx = ssl.create_default_context()
            conn = ctx.wrap_socket(
                socket.socket(socket.AF_INET),
                server_hostname=domain
            )
            conn.settimeout(self.timeout)
            conn.connect((domain, 443))
            cert = conn.getpeercert()
            conn.close()
            
            result["has_ssl"] = 1
            
            issuer = dict(x[0] for x in cert.get("issuer", []))
            org    = issuer.get("organizationName", "unknown")
            result["ssl_issuer"] = str(org)[:50]
            
            # Let's Encrypt, ZeroSSL gibi ücretsiz CA'lar phishing'de yaygın
            free_cas = ["let's encrypt", "zerossl", "sectigo", "comodo free"]
            result["ssl_issuer_is_free"] = int(
                any(ca in org.lower() for ca in free_cas)
            )
        except Exception:
            pass
        return result


# ─────────────────────────────────────────────
# ANA PIPELINE
# ─────────────────────────────────────────────

def run_data_collection(
    kaggle_csv: str = None,
    phishtank_api_key: str = None,
    max_samples: int = 5000,
    collect_metadata: bool = False,
    metadata_sample: int = 500,
    output_name: str = "raw_dataset"
) -> pd.DataFrame:
    """
    Tüm veri toplama pipeline'ını çalıştırır.
    
    Args:
        kaggle_csv:        Kaggle CSV dosya yolu (opsiyonel)
        phishtank_api_key: PhishTank API key (opsiyonel)
        max_samples:       Maksimum örnek sayısı
        collect_metadata:  WHOIS/SSL metadata toplanacak mı?
        metadata_sample:   Kaç URL için metadata çekilsin?
        output_name:       Çıktı dosya adı
    
    Returns:
        pd.DataFrame: Birleşik dataset
    """
    print("=" * 60)
    print("  VERİ TOPLAMA PIPELINE BAŞLIYOR")
    print("=" * 60)
    
    # 1. PhishTank
    pt_collector = PhishTankCollector(
        api_key=phishtank_api_key,
        max_samples=max_samples // 2
    )
    phishtank_df = pt_collector.fetch()
    
    # 2. Kaggle
    if kaggle_csv and Path(kaggle_csv).exists():
        kaggle_loader = KaggleLoader()
        kaggle_df = kaggle_loader.load(kaggle_csv)
    else:
        print("ℹ️  Kaggle CSV bulunamadı, sadece PhishTank kullanılıyor")
        # Meşru URL'leri kendi ekleyelim
        legit_rows = []
        for url in LEGITIMATE_URLS:
            legit_rows.append({
                "url": url.lower(),
                "timestamp": pd.Timestamp(
                    year=random.choice([2023, 2024]),
                    month=random.randint(1, 12),
                    day=random.randint(1, 28),
                    tz="UTC"
                ),
                "label": 0,
                "source": "known_legitimate"
            })
        kaggle_df = pd.DataFrame(legit_rows)
    
    # 3. Birleştir
    merger = DatasetMerger()
    combined = merger.merge(phishtank_df, kaggle_df, max_total=max_samples)
    
    # 4. Metadata (opsiyonel, zaman alır)
    if collect_metadata:
        meta_collector = MetadataCollector()
        combined = meta_collector.collect_all(combined, sample_size=metadata_sample)
    else:
        # Dummy metadata sütunları ekle
        # has_ssl: HTTPS ile başlayan URL'ler SSL kullanıyor demektir.
        # Şemasız URL'ler (örn: "example.com") HTTP olarak kabul edilir, has_ssl=0 alır.
        # Bu, feature_extractor.py'nin şemasız URL'leri http:// olarak normalize etmesiyle tutarlıdır.
        combined["domain_age_days"]        = -1
        combined["has_ssl"]                = combined["url"].str.startswith("https://").astype(int)
        combined["ssl_issuer"]             = "unknown"
        combined["ssl_issuer_is_free"]     = 0
        combined["registrar"]              = "unknown"
        combined["whois_available"]        = 0
        combined["registration_period_days"] = -1
        combined["country"]               = "unknown"
        print("ℹ️  Metadata atlandı (collect_metadata=False), has_ssl URL şemasından çıkarıldı")
    
    # 5. Kaydet
    out_path = DATA_RAW / f"{output_name}.csv"
    combined.to_csv(out_path, index=False)
    print(f"\n💾 Dataset kaydedildi: {out_path}")
    print(f"   Toplam: {len(combined):,} URL")
    
    return combined


if __name__ == "__main__":
    df = run_data_collection(
        max_samples=2000,
        collect_metadata=False,
        output_name="raw_dataset"
    )
    print(df.head())
    print(df.dtypes)
