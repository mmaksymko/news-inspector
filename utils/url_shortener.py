import hashlib

url_dict = {}
base_url = "http://short.ly/"

def encode_url(url):
    short_code = hashlib.sha256(url.encode()).hexdigest()[:6]
    url_dict[short_code] = url
    return base_url + short_code

def decode_url(short_url):
    short_code = short_url.replace(base_url, "")
    return url_dict.pop(short_code, None)