import re
from telegram import Message
from utils.log_utils import log_io
import utils.scraper_utils as su
from newspaper import Article
from telegram.constants import MessageEntityType, MessageOriginType


EXCLUDED_URL_PREFIXES = tuple(
    proto + domain + ('/' if not domain.endswith('/') else '')
    for domain in ['t.me', 'facebook.com', 'instagram.com']
    for proto in ('https://', 'http://', 'https://www.', 'https://www.', '')
)

@log_io()
def create_from_url(url: str) -> str:
    try:
        article = Article(url, language='uk')
        article.download()
        article.parse()
        article.text = su.remove_junk(article.text)
        article.source_url = article.source_url.removeprefix('https://').removeprefix('http://').removeprefix('www.')
        article.meta_data['source'] = 'url'

        # print('Authors: ' + ', '.join(article.authors))
        # print('Top image: ' + article.top_image)
        # print('Is media news: ' + article.is_media_news())
        # print("Publish date: " + article.publish_date.strftime('%Y-%m-%d %H:%M:%S') if article.publish_date else "No publish date")
        return article
    except Exception as e:
        return None

@log_io()
def create_from_message(message: Message) -> Article:
    def parse_message(text: str):
        split = text.split('\n', 1)
        if len(split) > 1:
            content, body = split[0], split[1]
        elif len(text) > 128:
            content, body = None, text
        else:
            content, body = text, None

        return (content.strip() if content else None, su.remove_junk(body.strip()) if body else None)
        
    article = Article('', language='uk')
    article.title, article.text = parse_message(message.text or message.caption)
    article.source_url = get_redirect_chat(message) or message.chat.username
    article.meta_data['source'] = 'telegram'
    # article.publish_date = extract from message if forwarded else None
    # article.top_image = extract first from message else None
    # maybe create separate method that processes if media news or not for both types depenfing on source
    return article

def get_redirect_chat(message: Message) -> str:
    if message.forward_origin:
        if message.forward_origin.type == MessageOriginType.CHANNEL:
            return message.forward_origin.chat.username
        if message.forward_origin.type in (MessageOriginType.USER, MessageOriginType.HIDDEN_USER, MessageOriginType.CHAT):
            return message.forward_origin.sender_user.username
    return None

def is_self_link(url: str, channel_name: str) -> bool:
    pattern = fr"^(?:\+.*|@?{re.escape(channel_name)}(?:/.*)?)$"
    normalized_url = re.sub(r"^(https?://)?(t\.me/|telegram\.me/)", "", url)
    return bool(re.match(pattern, normalized_url))

def get_urls(message: Message) -> list[str]:
    def get_text_urls(src) -> list[str]:
        return [entity.url for entity in src if entity.type == MessageEntityType.TEXT_LINK]
    
    entities = list(filter(lambda entity: entity.type == MessageEntityType.URL, message.entities))
    text_urls = get_text_urls(message.entities) + get_text_urls(message.caption_entities)
    if chat:=get_redirect_chat(message):
        text_urls = list(filter(lambda url: not is_self_link(url, chat), text_urls))
    
    oset = to_oset([message.text[entity.offset:entity.offset+entity.length] for entity in entities]) + text_urls
    return [element for element in oset if not element.startswith(EXCLUDED_URL_PREFIXES)]

def to_oset(items: list) -> set:
    return list(dict.fromkeys(items).keys())