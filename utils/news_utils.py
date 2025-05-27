import re
from urllib.parse import urlparse

def is_url(url: str) -> bool:
    url_regex = re.compile(
        r"^(http(s)?://)?(?:[a-zA-Z0-9-]+\.)+[a-zA-Z]{2,}|(?:\d{1,3}\.){3}\d{1,3}(?::\d+)?(?:/.*)?$"
    )
    if not url_regex.match(url):
        return False

    parsed = urlparse(url if "://" in url else f"https://{url}")
    return bool(parsed.netloc and parsed.scheme)


