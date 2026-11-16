"""Turn a URL string into a fixed list of simple numeric features.

Training and live prediction both call extract_features(). The function
reads the URL text only. It does not open, visit, or download the address.

Scheme rule:
- A missing scheme is treated as https:// (example.com becomes https://example.com).
- A leading // is treated as https:// (protocol-relative URL).
- Only http and https are accepted.
- A leading www. label is removed. Every legitimate URL in the PhiUSIIL file
  starts with www, so keeping it would let the model treat that prefix as
  proof of safety. www.x.com and x.com are scored as the same host.
"""

from urllib.parse import urlparse, urlunparse
import ipaddress


FEATURE_NAMES = [
    "url_length",
    "dot_count",
    "hyphen_count",
    "digit_count",
    "at_count",
    "subdomain_count",
    "uses_https",
    "is_ip_address",
    "suspicious_keyword_count",
    "question_mark_count",
    "equals_count",
    "path_length",
]

SUSPICIOUS_KEYWORDS = [
    "login",
    "verify",
    "update",
    "secure",
    "account",
    "confirm",
    "password",
]

# The website refuses very long pastes. Training does not use this limit.
API_MAX_URL_LENGTH = 2048

FEATURE_INFO = [
    {
        "name": "url_length",
        "label": "URL length",
        "description": "Total number of characters in the normalized URL.",
        "why": "Some phishing links are very long because they hide the real destination or add extra tracking text.",
    },
    {
        "name": "dot_count",
        "label": "Number of dots",
        "description": "How many '.' characters appear in the full URL string.",
        "why": "Extra dots can come from extra host labels or from a long, messy link.",
    },
    {
        "name": "hyphen_count",
        "label": "Number of hyphens",
        "description": "How many '-' characters appear in the full URL string.",
        "why": "Hyphens are sometimes used to imitate a real brand name inside a hostname or path.",
    },
    {
        "name": "digit_count",
        "label": "Number of digits",
        "description": "How many numeric characters appear in the URL.",
        "why": "Automatically generated phishing links often contain many digits.",
    },
    {
        "name": "at_count",
        "label": "Number of @ symbols",
        "description": "How many '@' characters appear in the URL.",
        "why": "An @ symbol can make the real hostname harder to notice, because the browser uses the host after the @.",
    },
    {
        "name": "subdomain_count",
        "label": "Number of subdomains",
        "description": (
            "Extra hostname labels beyond the last two labels, after a leading www is removed. "
            "example.com and www.example.com both count as 0. login.example.com counts as 1. "
            "This is a simple approximation and does not understand suffixes such as co.uk. "
            "IP addresses count as 0. "
            "www is removed because every legitimate row in this dataset starts with www."
        ),
        "why": "Extra subdomains are sometimes used to imitate a trusted brand.",
    },
    {
        "name": "uses_https",
        "label": "HTTPS usage",
        "description": "1 if the scheme is https, otherwise 0. A missing scheme is treated as https before this check.",
        "why": "The scheme is one signal. HTTPS encrypts traffic, but phishing websites can use HTTPS too, so it is not proof of safety.",
    },
    {
        "name": "is_ip_address",
        "label": "IP address usage",
        "description": "1 if the hostname is an IPv4 or IPv6 address according to Python's ipaddress module, otherwise 0.",
        "why": "Most public websites use a domain name. A raw IP address can be suspicious, though it is not always malicious.",
    },
    {
        "name": "suspicious_keyword_count",
        "label": "Suspicious keywords",
        "description": (
            "Total substring count of these words, matched without regard to case: "
            + ", ".join(SUSPICIOUS_KEYWORDS)
            + ". A word inside a longer word can also match. These are weak indicators, not proof."
        ),
        "why": "Words such as login, verify, and password appear in many phishing lures. They also appear on real login pages.",
    },
    {
        "name": "question_mark_count",
        "label": "Number of question marks",
        "description": "How many '?' characters appear in the URL.",
        "why": "A question mark usually starts a query string, which can carry extra parameters.",
    },
    {
        "name": "equals_count",
        "label": "Number of equals signs",
        "description": "How many '=' characters appear in the URL.",
        "why": "Equals signs usually join a query parameter to its value.",
    },
    {
        "name": "path_length",
        "label": "URL path length",
        "description": "Number of characters in the URL path returned by urllib.parse, including the leading slash when one is present.",
        "why": "A very long path can be used to bury the real destination of a link.",
    },
]


def normalize_url(url, max_length=None):
    """Return a cleaned URL string, or raise ValueError if it cannot be used."""
    if url is None:
        raise ValueError("Please enter a URL.")
    if not isinstance(url, str):
        raise ValueError("The url field must be a string.")

    text = url.strip()
    if text == "" or text.lower() == "nan":
        raise ValueError("Please enter a URL.")
    if any(character.isspace() for character in text):
        raise ValueError("The URL contains spaces. Remove the spaces and try again.")
    if max_length is not None and len(text) > max_length:
        raise ValueError(
            "This URL is too long to analyze. Please submit a URL under "
            + str(max_length)
            + " characters."
        )

    if text.startswith("//"):
        text = "https:" + text

    parsed = urlparse(text)
    if parsed.scheme == "" and "://" not in text:
        text = "https://" + text
        parsed = urlparse(text)

    scheme = parsed.scheme.lower()
    if scheme not in ("http", "https"):
        raise ValueError("Only HTTP and HTTPS URLs can be analyzed.")
    if parsed.hostname is None or parsed.hostname == "":
        raise ValueError("This URL does not have a valid hostname.")
    return strip_leading_www(text)


def strip_leading_www(url):
    """Drop a leading www. label. www.example.com and example.com become the same host."""
    parsed = urlparse(url)
    host = parsed.hostname or ""
    if not host.lower().startswith("www."):
        return url
    netloc = parsed.netloc
    prefix, _, hostport = netloc.rpartition("@")
    if prefix:
        prefix = prefix + "@"
    if not hostport.lower().startswith("www."):
        return url
    return urlunparse(parsed._replace(netloc=prefix + hostport[4:]))


def hostname_is_ip(hostname):
    """Return True when the hostname is an IP address, not merely text that contains digits."""
    if hostname is None:
        return False
    candidate = hostname.strip().strip("[]")
    try:
        ipaddress.ip_address(candidate)
        return True
    except ValueError:
        return False


def count_subdomains(hostname, is_ip):
    """Count extra hostname labels using a simple two-label approximation.

    The last two dot-separated labels are treated as the domain and its suffix.
    Labels before those two are counted as subdomains.
    """
    if is_ip or hostname is None or hostname == "":
        return 0
    labels = [part for part in hostname.split(".") if part != ""]
    if len(labels) <= 2:
        return 0
    return len(labels) - 2


def extract_features(url, max_length=None):
    """Return one feature dictionary. Keys always follow FEATURE_NAMES."""
    normalized = normalize_url(url, max_length=max_length)
    parsed = urlparse(normalized)
    hostname = parsed.hostname or ""
    lowered = normalized.lower()
    is_ip = hostname_is_ip(hostname)

    keyword_total = 0
    for keyword in SUSPICIOUS_KEYWORDS:
        keyword_total += lowered.count(keyword)

    values = {
        "url_length": len(normalized),
        "dot_count": normalized.count("."),
        "hyphen_count": normalized.count("-"),
        "digit_count": sum(1 for character in normalized if character.isdigit()),
        "at_count": normalized.count("@"),
        "subdomain_count": count_subdomains(hostname, is_ip),
        "uses_https": 1 if parsed.scheme.lower() == "https" else 0,
        "is_ip_address": 1 if is_ip else 0,
        "suspicious_keyword_count": keyword_total,
        "question_mark_count": normalized.count("?"),
        "equals_count": normalized.count("="),
        "path_length": len(parsed.path or ""),
    }
    return {name: int(values[name]) for name in FEATURE_NAMES}


def feature_vector(url, max_length=None):
    """Return the feature values in the same order used during training."""
    features = extract_features(url, max_length=max_length)
    return [features[name] for name in FEATURE_NAMES]
