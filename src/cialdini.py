"""
cialdini.py - Psikolojik Manipülasyon Tespiti
Cialdini'nin 6 İkna İlkesi ile URL analizi
Türkçe genişletilmiş lexicon - v3
"""

import re
from urllib.parse import urlparse
from typing import Dict, List

CIALDINI_LEXICON = {
    "authority": [
        # İngilizce genel
        "official", "secure", "verified", "authorized", "legitimate",
        "government", "gov", "bank", "admin", "support", "helpdesk",
        "service", "portal", "headquarters", "certified", "trusted",
        "auth", "authentication", "identity", "id", "credential",
        "ssl", "security", "protected", "safe", "genuine",
        "real", "original", "main", "primary", "central", "global",
        "international", "national", "federal", "state", "department",
        "corp", "corporate", "inc",
        # Global markalar
        "paypal", "amazon", "google", "microsoft", "apple", "netflix",
        "instagram", "facebook", "twitter", "linkedin", "ebay",
        "whatsapp", "telegram", "tiktok", "snapchat", "youtube",
        "spotify", "uber", "airbnb", "booking", "visa", "mastercard",
        "americanexpress", "chase", "wellsfargo", "citibank", "hsbc",
        "barclays", "irs", "fbi", "interpol", "police", "court", "legal",
        # Türk bankaları
        "garanti", "garantibbva", "ziraat", "ziraatbank", "akbank",
        "isbank", "isbankas", "halkbank", "vakifbank", "vakif",
        "finansbank", "qnbfinansbank", "qnb", "enpara", "yapikrdi",
        "yapikredi", "ykb", "teb", "ingbank", "icbc", "odeabank",
        "odea", "burgan", "sekerbank", "alternatifbank", "fibabanka",
        "turkish", "turkbank", "anadolubank", "turkiyefinans",
        "kuveytturk", "albarakaturk", "albaraka", "ptt", "pttbank",
        # Türk telecom
        "turkcell", "vodafone", "turktelekom", "turk", "telekom",
        "superonline", "ttnet", "bipsiparis",
        # Türk e-ticaret
        "trendyol", "hepsiburada", "n11", "gittigidiyor", "ciceksepeti",
        "morhipo", "boyner", "lcwaikiki", "defacto", "koton", "mavi",
        "zara", "h&m", "mediamarkt", "teknosa", "vatan", "bimeks",
        # Türk market/gıda
        "bim", "migros", "carrefour", "a101", "sok", "file", "koctas",
        "ikea", "decathlon", "intersport",
        # Türk haber/medya
        "hurriyet", "milliyet", "sabah", "sozcu", "cumhuriyet",
        "haberturk", "ntv", "cnnturk", "trt", "show", "kanal",
        # Türk gayrimenkul/ilan
        "sahibinden", "emlakjet", "zingat", "hepsiemlak",
        # Türk seyahat
        "thyao", "thy", "turkishairlines", "pegasus", "sunexpress",
        "anadolujet", "obilet", "enuygun", "tatilsepeti",
        # Türk fintech/ödeme
        "papara", "ininal", "param", "iyzico", "paybull", "paytr",
        "sipay", "nkolay", "hayat", "fintech",
        # Türk devlet kurumları
        "turkiye", "tcmb", "spk", "bddk", "sgk", "gib", "nvi",
        "egm", "icisleri", "maliye", "hazine", "merkez",
    ],
    "scarcity": [
        # İngilizce
        "urgent", "limited", "expire", "expiring", "expired", "deadline",
        "last", "final", "hurry", "soon", "today", "now", "immediate",
        "critical", "important", "alert", "warning", "notice",
        "act", "quickly", "fast", "asap", "immediately", "hours",
        "minutes", "ending", "closing", "only", "remaining", "left",
        "few", "rare", "exclusive", "limited-time", "time-sensitive",
        "priority", "emergency", "breaking", "instant", "quick",
        "rapid", "sudden", "temporary", "24h", "48h", "72h",
        "countdown", "running-out", "scarce",
        # Türkçe
        "acil", "hemen", "simdi", "bugun", "son", "bitis", "sure",
        "dakika", "saat", "kritik", "onemli", "uyari", "ivedi",
        "derhal", "vakit", "geciktirme", "beklemeden", "aninda",
        "sureli", "sinirli", "kampanya", "firsat", "indirim",
        "sona-eriyor", "son-gun", "son-saat", "son-dakika",
    ],
    "fear": [
        # İngilizce
        "suspended", "blocked", "locked", "disabled", "terminated",
        "compromised", "hacked", "breach", "violation", "illegal",
        "unauthorized", "suspicious", "fraud", "stolen", "infected",
        "virus", "malware", "danger", "risk", "threat", "attack",
        "penalty", "fine", "lawsuit", "arrested", "banned", "closed",
        "deactivated", "restricted", "flagged", "reported", "detected",
        "unusual", "abnormal", "unrecognized", "failed", "error",
        "problem", "issue", "trouble", "concern", "leaked", "exposed",
        "vulnerable", "weak", "unsafe", "insecure", "warning",
        "caution", "harmful", "malicious", "phishing", "scam",
        # Türkçe
        "askiya", "alindi", "engellendi", "kapatildi", "donduruldu",
        "iptal", "iptali", "tehlike", "tehdit", "risk", "guvenlik",
        "ihlal", "yetkisiz", "dolandirici", "dolandiricilik",
        "sahte", "kisisel", "gizli", "sifre-calindi", "hesap-ele",
        "gecersiz", "hatali", "basarisiz", "onaylanmadi", "reddedildi",
        "ceza", "para-cezasi", "yasal-islem", "sikayete",
        "bloke", "kisitlama", "uyari", "alarm",
    ],
    "reciprocity": [
        # İngilizce
        "free", "gift", "bonus", "reward", "prize", "win", "winner",
        "congratulations", "selected", "chosen", "lucky", "special",
        "exclusive", "offer", "deal", "discount", "cashback", "refund",
        "claim", "redeem", "collect", "earn", "benefit", "promo",
        "giveaway", "freebie", "complimentary", "gratis", "no-cost",
        "treat", "surprise", "jackpot", "lottery", "sweepstakes",
        "raffle", "coupon", "voucher", "code", "savings", "sale",
        "clearance", "bargain", "cheap", "rebate", "compensation",
        "payout", "payment", "cash", "money",
        # Türkçe
        "ucretsiz", "bedava", "hediye", "ikramiye", "odul", "kazandin",
        "kazandi", "tebrikler", "secildiniz", "secildi", "san",
        "ozel", "kampanya", "indirim", "geri-odeme", "iade",
        "puan", "bonus", "para-iadesi", "nakit", "para-kazan",
        "promosyon", "kupon", "kod", "firsati", "avantaj",
    ],
    "social_proof": [
        # İngilizce
        "million", "users", "customers", "people", "everyone", "community",
        "popular", "trending", "viral", "famous", "top", "best",
        "rated", "reviewed", "trusted-by", "recommended", "endorsed",
        "verified-by", "approved", "certified-by", "joined", "members",
        "subscribers", "followers", "fans", "reviews", "testimonials",
        "feedback", "stars", "likes", "shares", "votes", "ranking",
        "most-popular", "highly-rated", "award-winning", "featured",
        # Türkçe
        "milyon", "kullanici", "musteri", "herkes", "topluluk",
        "populer", "trend", "unlu", "en-iyi", "tavsiye", "onaylandi",
        "guvenilir", "katildi", "uye", "takipci", "yorum", "puan",
        "begeni", "paylasim", "oy", "siralama", "odul-kazandi",
    ],
    "commitment": [
        # İngilizce
        "confirm", "verify", "validate", "update", "complete", "finish",
        "activate", "register", "signup", "login", "signin", "access",
        "continue", "proceed", "submit", "enter", "provide", "fill",
        "required", "mandatory", "must", "need", "account", "profile",
        "agree", "accept", "acknowledge", "consent", "authorize",
        "approve", "enable", "unlock", "restore", "recover", "reset",
        "change", "modify", "edit", "review", "check", "manage",
        "setup", "configure", "install", "download", "upload",
        "connect", "link", "sync", "integrate", "migrate", "transfer",
        # Türkçe
        "giris", "uye", "kayit", "onayla", "dogrula", "guncelle",
        "tamamla", "aktivasyon", "sifre", "hesap", "profil",
        "basvur", "basvuru", "onay", "kabul", "sozlesme", "sartlar",
        "zorunlu", "gerekli", "doldurun", "girin", "tiklayın",
        "devam", "ilerle", "yukle", "indir", "baglan", "yenile",
        "degistir", "duzelt", "kontrol", "yonet", "ayarla",
    ]
}


class CialdiniAnalyzer:
    def __init__(self):
        self.lexicon = CIALDINI_LEXICON
        self.principles = list(CIALDINI_LEXICON.keys())

    def tokenize_url(self, url: str) -> List[str]:
        url = re.sub(r'https?://', '', url.lower())
        url = re.sub(r'[/\-_\.@?=&%+#]', ' ', url)
        url = re.sub(r'(\d+)([a-z])', r'\1 \2', url)
        url = re.sub(r'([a-z])(\d+)', r'\1 \2', url)
        tokens = [t for t in url.split() if len(t) > 2]
        return tokens

    def score_url(self, url: str) -> Dict[str, float]:
        tokens = self.tokenize_url(url)
        total_tokens = max(len(tokens), 1)
        scores = {}
        for principle, keywords in self.lexicon.items():
            matches = sum(1 for token in tokens if token in keywords)
            scores[f"cld_{principle}"] = round(min(matches / total_tokens * 3, 1.0), 4)
        scores["cld_total_score"] = round(sum(scores.values()) / len(scores), 4)
        principle_scores = {k: v for k, v in scores.items() if k != "cld_total_score"}
        dominant = max(principle_scores, key=principle_scores.get)
        scores["cld_dominant_principle"] = self.principles.index(
            dominant.replace("cld_", "")
        )
        return scores

    def explain(self, url: str) -> Dict:
        scores = self.score_url(url)
        tokens = self.tokenize_url(url)
        principle_names = {
            "cld_authority": "Otorite",
            "cld_scarcity": "Kıtlık / Aciliyet",
            "cld_fear": "Korku",
            "cld_reciprocity": "Karşılıklılık / Ödül",
            "cld_social_proof": "Sosyal Kanıt",
            "cld_commitment": "Bağlılık / Taahhüt"
        }
        active = {
            principle_names[k]: v
            for k, v in scores.items()
            if k in principle_names and v > 0
        }
        matched_words = {}
        for principle, keywords in self.lexicon.items():
            found = [t for t in tokens if t in keywords]
            if found:
                matched_words[principle] = found
        return {
            "scores": scores,
            "active_principles": active,
            "matched_words": matched_words,
            "total_score": scores["cld_total_score"],
            "is_manipulative": scores["cld_total_score"] > 0.1
        }


if __name__ == "__main__":
    analyzer = CialdiniAnalyzer()
    test_urls = [
        "http://garanti-giris-guvenlik.xyz/dogrula",
        "https://ziraat-hesap-askiya-alindi.com/login",
        "http://akbank-ucretsiz-bonus-kazan.xyz/claim",
        "https://turkcell-ozel-kampanya-simdi.com/kazan",
        "http://paypal-secure-login.xyz/verify?user=1234",
        "https://www.google.com",
    ]
    for url in test_urls:
        result = analyzer.explain(url)
        print(f"\nURL: {url}")
        print(f"Toplam Skor: {result['total_score']}")
        print(f"Aktif İlkeler: {result['active_principles']}")
        print(f"Eşleşen Kelimeler: {result['matched_words']}")
