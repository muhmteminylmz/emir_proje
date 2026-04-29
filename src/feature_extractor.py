"""
feature_extractor.py - Feature Cikarma Modulu
URL Features + Domain Metadata + Adversarial Features (Levenshtein, Homoglyph)
"""

import re
import math
import string
from urllib.parse import urlparse, parse_qs
from typing import Dict, List, Optional
from pathlib import Path

import numpy as np
import pandas as pd
import tldextract
from tqdm import tqdm

KNOWN_BRANDS = [
    # Global markalar
    "google", "youtube", "facebook", "amazon", "wikipedia", "twitter",
    "instagram", "linkedin", "reddit", "netflix", "microsoft", "apple",
    "github", "stackoverflow", "paypal", "ebay", "walmart", "dropbox",
    "spotify", "twitch", "adobe", "oracle", "ibm", "salesforce", "zoom",
    "slack", "trello", "notion", "canva", "cloudflare", "bankofamerica",
    "chase", "wellsfargo", "citibank", "hsbc", "barclays", "americanexpress",
    "visa", "mastercard", "whatsapp", "telegram", "tiktok", "snapchat",
    "yahoo", "bing", "duckduckgo", "pinterest", "tumblr", "quora",
    "medium", "wordpress", "blogger", "wix", "squarespace", "shopify",
    "stripe", "square", "coinbase", "binance", "kraken", "opensea",
    "uber", "lyft", "airbnb", "booking", "tripadvisor", "expedia",
    "aliexpress", "alibaba", "etsy", "rakuten", "wish",
    # URL kısaltıcılar - MEŞRU
    "bitly", "bit", "tinyurl", "tiny", "goo", "ow", "t",
    "rebrand", "short", "buff", "ift", "dlvr",
    # Türkçe haber ve medya
    "onedio", "hurriyet", "sabah", "milliyet", "cumhuriyet",
    "haberturk", "ntv", "cnnturk", "trt", "sozcu",
    "mynet", "ensonhaber", "takvim", "posta", "aksam",
    "bloomberght", "dunya", "ekonomist", "forbes",
    # Türkçe e-ticaret
    "trendyol", "hepsiburada", "gittigidiyor", "pttavm",
    "sahibinden", "arabam", "ikinciyeni", "dolap",
    "yemeksepeti", "getir", "trendyolmarket", "migros",
    "amazon", "morhipo", "boyner", "lcwaikiki", "defacto",
    # Türkçe bankalar
    "akbank", "garanti", "isbank", "ziraatbank", "vakifbank",
    "enpara", "finansbank", "qnb", "denizbank", "yapikredi",
    "halkbank", "teb", "ing", "odea", "albaraka", "kuveytturk",
    # Türkçe telekom
    "turkcell", "vodafone", "turktelekom", "turk", "turknet",
    "superonline", "millenicom", "kablonet",
    # Türkçe devlet
    "gov", "turkiye", "egov", "edevlet", "meb", "saglik",
    "tubitak", "btk", "spk", "bddk", "tcmb",
    # Türkçe diğer
    "eksi", "uludagsozluk", "instela", "twitter",
    "biletix", "passo", "ticketmaster",
]

HOMOGLYPH_MAP = {
    'a': ['@', '4'],
    'e': ['3'],
    'i': ['1', 'l', '!'],
    'o': ['0'],
    's': ['5', '$'],
    'l': ['1', 'I'],
    'g': ['9', 'q'],
    'b': ['6'],
    't': ['+'],
}

SUSPICIOUS_TLDS = {
    "xyz", "tk", "ml", "ga", "cf", "gq", "pw", "top", "club",
    "online", "site", "website", "space", "fun", "live", "store",
    "shop", "loan", "click", "link", "email", "zip", "mov",
}

TRUSTED_TLDS = {"com", "org", "edu", "gov", "net", "co"}

SUSPICIOUS_KEYWORDS = [
    "login", "signin", "verify", "secure", "account", "update",
    "confirm", "banking", "password", "credential", "paypal", "ebay",
    "amazon", "apple", "microsoft", "support", "help", "service",
    "validation", "authenticate", "billing", "payment", "suspended",
    "unusual", "activity", "click", "here", "free", "win", "prize",
]


class URLFeatureExtractor:
    def extract(self, url: str) -> Dict:
        features = {}
        try:
            parsed = urlparse(url)
            ext    = tldextract.extract(url)
            features["url_length"]        = len(url)
            features["domain_length"]     = len(ext.domain)
            features["path_length"]       = len(parsed.path)
            features["query_length"]      = len(parsed.query)
            features["subdomain_length"]  = len(ext.subdomain)
            features["num_subdomains"]    = len(ext.subdomain.split(".")) if ext.subdomain else 0
            features["num_dots"]          = url.count(".")
            features["num_hyphens"]       = url.count("-")
            features["num_underscores"]   = url.count("_")
            features["num_slashes"]       = url.count("/")
            features["num_question"]      = url.count("?")
            features["num_equals"]        = url.count("=")
            features["num_at"]            = url.count("@")
            features["num_ampersand"]     = url.count("&")
            features["num_exclamation"]   = url.count("!")
            features["num_tilde"]         = url.count("~")
            features["num_percent"]       = url.count("%")
            features["num_hash"]          = url.count("#")
            features["num_plus"]          = url.count("+")
            digits    = sum(c.isdigit() for c in url)
            alpha     = sum(c.isalpha() for c in url)
            special   = len(url) - digits - alpha - url.count(" ")
            url_len   = max(len(url), 1)
            features["digit_ratio"]       = digits / url_len
            features["alpha_ratio"]       = alpha / url_len
            features["special_ratio"]     = special / url_len
            features["url_entropy"]       = self._entropy(url)
            features["domain_entropy"]    = self._entropy(ext.domain)
            features["path_entropy"]      = self._entropy(parsed.path)
            features["is_https"]          = int(parsed.scheme == "https")
            features["has_port"]          = int(bool(parsed.port))
            features["port"]              = parsed.port if parsed.port else 0
            tld = ext.suffix.lower() if ext.suffix else ""
            features["tld_suspicious"]    = int(tld in SUSPICIOUS_TLDS)
            features["tld_trusted"]       = int(tld in TRUSTED_TLDS)
            features["tld_length"]        = len(tld)
            features["is_ip"]             = int(bool(
                re.match(r"^\d{1,3}(\.\d{1,3}){3}$", parsed.hostname or "")
            ))
            url_lower = url.lower()
            features["suspicious_keywords"] = sum(
                1 for kw in SUSPICIOUS_KEYWORDS if kw in url_lower
            )
            features["has_login_keyword"]  = int("login" in url_lower or "signin" in url_lower)
            features["has_verify_keyword"] = int("verify" in url_lower or "confirm" in url_lower)
            features["has_secure_keyword"] = int("secure" in url_lower)
            features["has_paypal_keyword"] = int("paypal" in url_lower)
            features["has_account_keyword"] = int("account" in url_lower)
            path_parts = [p for p in parsed.path.split("/") if p]
            features["path_depth"]        = len(path_parts)
            features["num_query_params"]  = len(parse_qs(parsed.query))
            features["has_double_slash"]  = int("//" in parsed.path)
            features["domain_has_digit"]  = int(any(c.isdigit() for c in ext.domain))
            features["domain_has_hyphen"] = int("-" in ext.domain)
            features["long_url"]          = int(len(url) > 75)
            features["very_long_url"]     = int(len(url) > 100)
            features["short_url"]         = int(len(url) < 20)
        except Exception:
            features = self._zero_features()
        return features

    def _entropy(self, s: str) -> float:
        if not s:
            return 0.0
        freq = {}
        for c in s:
            freq[c] = freq.get(c, 0) + 1
        n = len(s)
        return -sum((f/n) * math.log2(f/n) for f in freq.values())

    def _zero_features(self) -> Dict:
        return {f: 0 for f in self._get_feature_names()}

    def _get_feature_names(self) -> List[str]:
        dummy = self.extract("http://example.com/path?q=1")
        return list(dummy.keys())


class AdversarialFeatureExtractor:
    def extract(self, url: str) -> Dict:
        features = {}
        try:
            ext    = tldextract.extract(url)
            domain = ext.domain.lower()
            lev_scores = self._levenshtein_all_brands(domain)
            features["min_brand_levenshtein"]    = lev_scores["min_dist"]
            features["min_brand_levenshtein_norm"] = lev_scores["min_norm"]
            features["closest_brand"]            = lev_scores["closest"]
            features["is_near_brand"]            = int(lev_scores["min_dist"] <= 2 and
                                                       lev_scores["min_dist"] > 0)
            features["is_exact_brand"]           = int(lev_scores["min_dist"] == 0)
            features["has_homoglyph"]            = self._detect_homoglyph(domain)
            features["homoglyph_count"]          = self._count_homoglyphs(domain)
            brand_in_domain = self._brand_in_domain(domain)
            features["brand_in_non_brand_domain"] = brand_in_domain
            features["digit_substitution"]       = self._has_digit_substitution(domain)
            features["repeated_chars"]           = self._count_repeated_chars(domain)
            features["domain_confusion_score"]   = self._confusion_score(domain, lev_scores)
        except Exception:
            features = {
                "min_brand_levenshtein": 999,
                "min_brand_levenshtein_norm": 1.0,
                "closest_brand": "none",
                "is_near_brand": 0,
                "is_exact_brand": 0,
                "has_homoglyph": 0,
                "homoglyph_count": 0,
                "brand_in_non_brand_domain": 0,
                "digit_substitution": 0,
                "repeated_chars": 0,
                "domain_confusion_score": 0.0,
            }
        return features

    def _levenshtein(self, s1: str, s2: str) -> int:
        if len(s1) < len(s2):
            return self._levenshtein(s2, s1)
        if len(s2) == 0:
            return len(s1)
        prev_row = range(len(s2) + 1)
        for i, c1 in enumerate(s1):
            curr_row = [i + 1]
            for j, c2 in enumerate(s2):
                insertions  = prev_row[j + 1] + 1
                deletions   = curr_row[j] + 1
                substitutions = prev_row[j] + (c1 != c2)
                curr_row.append(min(insertions, deletions, substitutions))
            prev_row = curr_row
        return prev_row[-1]

    def _levenshtein_all_brands(self, domain: str) -> Dict:
        min_dist    = float('inf')
        closest     = "none"
        for brand in KNOWN_BRANDS:
            dist = self._levenshtein(domain, brand)
            if dist < min_dist:
                min_dist = dist
                closest  = brand
        max_len    = max(len(domain), 1)
        min_norm   = min_dist / max_len
        return {
            "min_dist": int(min_dist) if min_dist != float('inf') else 999,
            "min_norm": round(min_norm, 4),
            "closest":  closest,
        }

    def _detect_homoglyph(self, domain: str) -> int:
        for char, glyphs in HOMOGLYPH_MAP.items():
            for g in glyphs:
                if g in domain and char not in domain:
                    return 1
        return 0

    def _count_homoglyphs(self, domain: str) -> int:
        count = 0
        for char, glyphs in HOMOGLYPH_MAP.items():
            for g in glyphs:
                if g in domain:
                    count += domain.count(g)
        return count

    def _brand_in_domain(self, domain: str) -> int:
        for brand in KNOWN_BRANDS:
            if brand in domain and domain != brand:
                return 1
        return 0

    def _has_digit_substitution(self, domain: str) -> int:
        substitutions = [
            ("0", "o"), ("1", "l"), ("1", "i"),
            ("3", "e"), ("4", "a"), ("5", "s"),
        ]
        for digit, letter in substitutions:
            if digit in domain:
                modified = domain.replace(digit, letter)
                if any(brand in modified for brand in KNOWN_BRANDS):
                    return 1
        return 0

    def _count_repeated_chars(self, domain: str) -> int:
        count = 0
        for i in range(1, len(domain)):
            if domain[i] == domain[i-1]:
                count += 1
        return count

    def _confusion_score(self, domain: str, lev_scores: Dict) -> float:
        score = 0.0
        if lev_scores["min_dist"] == 1:
            score += 0.5
        elif lev_scores["min_dist"] == 2:
            score += 0.3
        score += min(0.3, self._count_homoglyphs(domain) * 0.1)
        # Brand domain içinde geçiyorsa yüksek skor
        if self._brand_in_domain(domain):
            score += 0.4
        # Tire sayısı fazlaysa şüpheli
        hyphen_count = domain.count("-")
        if hyphen_count >= 3:
            score += 0.4
        elif hyphen_count == 2:
            score += 0.2
        elif hyphen_count == 1:
            score += 0.1
        # Domain çok uzunsa şüpheli
        if len(domain) > 20:
            score += 0.2
        elif len(domain) > 15:
            score += 0.1
        return round(min(score, 1.0), 4)


class MetadataFeatureProcessor:
    def process(self, df: pd.DataFrame) -> pd.DataFrame:
        df = df.copy()
        if "domain_age_days" in df.columns:
            df["domain_age_days"] = df["domain_age_days"].fillna(-1)
            df["domain_very_new"]  = (df["domain_age_days"].between(0, 30)).astype(int)
            df["domain_new"]       = (df["domain_age_days"].between(0, 90)).astype(int)
            df["domain_recent"]    = (df["domain_age_days"].between(0, 365)).astype(int)
            df["domain_age_log"]   = np.log1p(df["domain_age_days"].clip(lower=0))
        if "has_ssl" in df.columns:
            df["has_ssl"] = df["has_ssl"].fillna(0).astype(int)
        if "ssl_issuer_is_free" in df.columns:
            df["ssl_issuer_is_free"] = df["ssl_issuer_is_free"].fillna(0).astype(int)
        if "registrar" in df.columns:
            df["registrar_known"] = df["registrar"].apply(
                lambda x: int(x not in ["unknown", "", None])
            )
        if "registration_period_days" in df.columns:
            df["registration_period_days"] = df["registration_period_days"].fillna(-1)
            df["short_registration"] = (
                df["registration_period_days"].between(0, 365)
            ).astype(int)
        if "whois_available" in df.columns:
            df["whois_available"] = df["whois_available"].fillna(0).astype(int)
        return df


class FeaturePipeline:
    """Tum feature extractor'lari birlestiren ana sinif."""

    def __init__(self):
        self.url_extractor  = URLFeatureExtractor()
        self.adv_extractor  = AdversarialFeatureExtractor()
        self.meta_processor = MetadataFeatureProcessor()

    def transform(self, df: pd.DataFrame, verbose: bool = True) -> pd.DataFrame:
        print("\n Feature extraction basliyor...")

        # 1. URL Features
        print("   [1/4] URL features cikariliyor...")
        url_features = []
        for url in tqdm(df["url"], desc="URL Features", disable=not verbose):
            url_features.append(self.url_extractor.extract(str(url)))
        url_df = pd.DataFrame(url_features)

        # 2. Adversarial Features
        print("   [2/4] Adversarial features cikariliyor...")
        adv_features = []
        for url in tqdm(df["url"], desc="Adversarial Features", disable=not verbose):
            adv_features.append(self.adv_extractor.extract(str(url)))
        adv_df = pd.DataFrame(adv_features)

        # 3. Metadata features isle
        print("   [3/4] Metadata features isleniyor...")
        meta_cols = [
            "domain_age_days", "has_ssl", "ssl_issuer_is_free",
            "registrar", "whois_available", "registration_period_days"
        ]
        available_meta = [c for c in meta_cols if c in df.columns]
        if available_meta:
            meta_df = self.meta_processor.process(df[available_meta + ["url"]].copy())
            meta_numeric = meta_df.select_dtypes(include=[np.number])
        else:
            meta_numeric = pd.DataFrame(index=df.index)

        # 4. Cialdini Psikolojik Feature'lar
        print("   [4/4] Cialdini psikolojik features cikariliyor...")
        from src.cialdini import CialdiniAnalyzer
        cialdini = CialdiniAnalyzer()
        cld_features = []
        for url in tqdm(df["url"], desc="Cialdini Features", disable=not verbose):
            scores = cialdini.score_url(str(url))
            cld_features.append(scores)
        cld_df = pd.DataFrame(cld_features)

        # 5. Birlestir
        feature_df = pd.concat([
            df[["url", "label"]].reset_index(drop=True),
            url_df.reset_index(drop=True),
            adv_df.reset_index(drop=True),
            meta_numeric.reset_index(drop=True),
            cld_df.reset_index(drop=True)
        ], axis=1)

        if "timestamp" in df.columns:
            feature_df["timestamp"] = df["timestamp"].values

        drop_cols = ["closest_brand"]
        feature_df = feature_df.drop(columns=[c for c in drop_cols if c in feature_df.columns])

        feature_cols = [c for c in feature_df.columns if c not in ["url", "label", "timestamp"]]
        feature_df[feature_cols] = feature_df[feature_cols].fillna(0)

        print(f"Feature extraction tamamlandi: {len(feature_df)} satir, {len(feature_cols)} feature")
        return feature_df

    def get_feature_names(self) -> List[str]:
        dummy_url = "http://paypa1-secure.login-now.xyz/verify?user=1"
        dummy_df  = pd.DataFrame({"url": [dummy_url], "label": [1]})
        result = self.transform(dummy_df, verbose=False)
        return [c for c in result.columns if c not in ["url", "label", "timestamp"]]


def time_based_split(df: pd.DataFrame,
                     train_end_year: int = 2023,
                     test_start_year: int = 2024) -> tuple:
    if "timestamp" not in df.columns:
        raise ValueError("DataFrame'de 'timestamp' sutunu bulunamadi!")

    df = df.copy()
    df["timestamp"] = pd.to_datetime(df["timestamp"], utc=True)
    cutoff = pd.Timestamp(f"{train_end_year + 1}-01-01", tz="UTC")
    train_df = df[df["timestamp"] < cutoff].copy()
    test_df  = df[df["timestamp"] >= cutoff].copy()

    if len(test_df) == 0:
        print("Test seti bos! Timestamp dagilimi kontrol ediliyor...")
        print(df["timestamp"].dt.year.value_counts())
        split_idx = int(len(df) * 0.8)
        train_df  = df.iloc[:split_idx]
        test_df   = df.iloc[split_idx:]
        print("   Fallback: son %20 test seti olarak kullanildi")

    exclude = ["url", "label", "timestamp"]
    feature_cols = [c for c in df.columns if c not in exclude]

    X_train = train_df[feature_cols].values
    X_test  = test_df[feature_cols].values
    y_train = train_df["label"].values
    y_test  = test_df["label"].values

    print(f"\nZaman Bazli Split:")
    print(f"   TRAIN: {len(train_df):,} ornek (<= {train_end_year})")
    print(f"   TEST:  {len(test_df):,} ornek (>= {test_start_year})")
    print(f"   Feature sayisi: {len(feature_cols)}")

    return X_train, X_test, y_train, y_test, train_df, test_df, feature_cols


if __name__ == "__main__":
    pipeline = FeaturePipeline()
    test_urls = [
        "http://paypa1-login.secure-account.xyz/verify?user=admin",
        "https://www.google.com/search?q=test",
    ]
    test_df = pd.DataFrame({
        "url": test_urls,
        "label": [1, 0],
        "timestamp": pd.date_range("2023-01-01", periods=2, freq="6ME", tz="UTC")
    })
    features = pipeline.transform(test_df)
    print(features.T)
