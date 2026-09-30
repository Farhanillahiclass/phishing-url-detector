"""
url_features.py
----------------
Turns a raw URL string into a fixed vector of NUMERIC, URL-only features.

Why this file exists (read this before touching train.py or app.py):
    The single most common mistake in this kind of project is training on one
    set of engineered features (e.g. columns already sitting in a CSV) and then
    feeding the deployed app a raw URL that gets turned into features a
    DIFFERENT way. The model then sees numbers it was never trained on and
    predictions become meaningless.

    To avoid that, this module is the ONLY place feature extraction happens.
    train.py calls extract_features() on every URL in the training data, and
    app.py calls the exact same function on whatever the user types in. There
    is no other feature-extraction code anywhere else in this project.

Design constraints (from the capstone brief):
    - Free / open-source, no paid APIs.
    - Must work instantly inside a Streamlit app -> NO live network calls
      (no WHOIS, no DNS lookups, no "is this page really live" checks).
      Every feature below is computed purely from the URL STRING.
    - Every feature must be explainable in one sentence for the viva.
"""

import re
import logging
from urllib.parse import urlparse

logger = logging.getLogger(__name__)

MAX_INPUT_LENGTH = 2048  # guards against pathological input (e.g. a pasted file)


# A short, well-known list of URL-shortening services (structural signal, no network call)
SHORTENING_SERVICES = re.compile(
    r"(bit\.ly|goo\.gl|shorte\.st|go2l\.ink|x\.co|ow\.ly|tinyurl|tr\.im|is\.gd|"
    r"cli\.gs|yfrog\.com|migre\.me|ff\.im|tiny\.cc|url4\.eu|twit\.ac|su\.pr|"
    r"twurl\.nl|snipurl\.com|short\.to|budurl\.com|ping\.fm|post\.ly|just\.as|"
    r"bkite\.com|snipr\.com|fic\.kr|loopt\.us|doiop\.com|short\.ie|kl\.am|"
    r"wp\.me|rubyurl\.com|om\.ly|to\.ly|t\.co|lnkd\.in)", re.IGNORECASE)

# TLDs that show up disproportionately often in phishing corpora (structural signal only)
SUSPICIOUS_TLDS = {"zip", "xyz", "top", "gq", "tk", "ml", "ga", "cf", "click",
                    "work", "support", "country", "loan", "men", "date"}

# Words that show up in phishing URLs trying to look official
PHISH_HINT_WORDS = ["secure", "account", "update", "login", "verify", "signin",
                     "banking", "confirm", "webscr", "ebayisapi", "wp", "password",
                     "suspend", "urgent", "billing"]

IP_RE = re.compile(
    r"^(?:(?:25[0-5]|2[0-4]\d|[01]?\d?\d)\.){3}(?:25[0-5]|2[0-4]\d|[01]?\d?\d)$")

FEATURE_NAMES = [
    "length_url", "length_hostname", "ip", "nb_dots", "nb_hyphens", "nb_at",
    "nb_qm", "nb_and", "nb_eq", "nb_underscore", "nb_percent", "nb_slash",
    "nb_www", "nb_com", "https_token", "ratio_digits_url", "ratio_digits_host",
    "nb_subdomains", "prefix_suffix", "shortening_service", "suspicious_tld",
    "port", "tld_in_path", "abnormal_subdomain", "length_words_raw",
    "shortest_word_raw", "longest_word_raw", "avg_word_raw", "char_repeat",
    "phish_hints", "nb_redirection",
]


def _safe_hostname(url):
    try:
        p = urlparse(url if "://" in url else "http://" + url)
        return p.hostname or ""
    except Exception:
        return ""


def _tokenize(url_no_scheme):
    return [w for w in re.split(r"[/\-_.?=&%]", url_no_scheme) if w]


def extract_features(url: str) -> dict:
    """Return an ordered dict of numeric features for a single raw URL string.
    Matches the order of FEATURE_NAMES exactly.

    Raises:
        ValueError: if `url` is not a non-empty string, or exceeds
            MAX_INPUT_LENGTH characters (guards against pathological input).
    """
    if not isinstance(url, str):
        raise ValueError(f"url must be a string, got {type(url).__name__}")
    url = url.strip()
    if not url:
        raise ValueError("url must not be empty")
    if len(url) > MAX_INPUT_LENGTH:
        raise ValueError(f"url exceeds MAX_INPUT_LENGTH ({MAX_INPUT_LENGTH} characters)")

    if not re.match(r"^[a-zA-Z][a-zA-Z0-9+.-]*://", url):
        # No scheme given (e.g. "example.com/login") -> assume http for parsing
        url_for_parse = "http://" + url
    else:
        url_for_parse = url

    parsed = urlparse(url_for_parse)
    hostname = parsed.hostname or ""
    path = parsed.path or ""
    full = url  # keep the ORIGINAL text (including any odd scheme) for char counts

    digits_url = sum(c.isdigit() for c in full)
    digits_host = sum(c.isdigit() for c in hostname)

    labels = hostname.split(".") if hostname else []
    # crude "registered domain" = second-level label before the TLD
    nb_subdomains = max(len(labels) - 2, 0) if len(labels) > 2 else 0

    tokens = _tokenize(full.split("://")[-1])
    word_lengths = [len(t) for t in tokens] if tokens else [0]

    # longest run of an identical character back to back (e.g. "aaaa" or "----")
    char_repeat = 1
    run = 1
    for i in range(1, len(full)):
        if full[i] == full[i - 1]:
            run += 1
            char_repeat = max(char_repeat, run)
        else:
            run = 1

    tld = labels[-1].lower() if labels else ""

    feats = {
        "length_url": len(full),
        "length_hostname": len(hostname),
        "ip": 1 if IP_RE.match(hostname) else 0,
        "nb_dots": full.count("."),
        "nb_hyphens": full.count("-"),
        "nb_at": full.count("@"),
        "nb_qm": full.count("?"),
        "nb_and": full.count("&"),
        "nb_eq": full.count("="),
        "nb_underscore": full.count("_"),
        "nb_percent": full.count("%"),
        "nb_slash": full.count("/"),
        "nb_www": 1 if "www" in hostname.lower() else 0,
        "nb_com": full.lower().count(".com"),
        # https_token: 1 = SUSPICIOUS ("https" appears somewhere it shouldn't,
        # e.g. embedded in the hostname text rather than as the real scheme)
        "https_token": 1 if ("https" in (hostname + path).lower()
                              and parsed.scheme != "https") else 0,
        "ratio_digits_url": round(digits_url / len(full), 4) if full else 0,
        "ratio_digits_host": round(digits_host / len(hostname), 4) if hostname else 0,
        "nb_subdomains": nb_subdomains,
        "prefix_suffix": 1 if "-" in (labels[-2] if len(labels) >= 2 else "") else 0,
        "shortening_service": 1 if SHORTENING_SERVICES.search(full) else 0,
        "suspicious_tld": 1 if tld in SUSPICIOUS_TLDS else 0,
        "port": 1 if parsed.port else 0,
        "tld_in_path": 1 if tld and tld in path.lower() else 0,
        "abnormal_subdomain": 1 if nb_subdomains >= 3 else 0,
        "length_words_raw": len(tokens),
        "shortest_word_raw": min(word_lengths),
        "longest_word_raw": max(word_lengths),
        "avg_word_raw": round(sum(word_lengths) / len(word_lengths), 3),
        "char_repeat": char_repeat,
        "phish_hints": sum(full.lower().count(w) for w in PHISH_HINT_WORDS),
        "nb_redirection": max(full.count("//") - 1, 0),
    }
    return feats


def extract_features_vector(url: str):
    """Same as extract_features but returns a plain list in FEATURE_NAMES order
    (what the model actually consumes)."""
    d = extract_features(url)
    return [d[name] for name in FEATURE_NAMES]


if __name__ == "__main__":
    # quick manual sanity check
    tests = [
        "http://www.crestonwood.com/router.php",
        "http://paypal-secure-login.xyz/verify/update?id=93af93",
        "https://192.168.1.5/wp-admin/login.php",
        "https://www.google.com/search?q=hello",
    ]
    for u in tests:
        print(u)
        print(extract_features(u))
        print()
