import csv
import os
import sys
import orjson as json
from typing import Dict, List
import csv
import utils.string_utils as su

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'news')))

def process_string(str: str):
    str = su.remove_emojis(str)
    str = su.process_escape_sequences(str)
    str = su.normalize_quotes(str)
    str = str.replace('\n', ' ')
    return str.strip('\n ')

def parse_messages(messages, title_only: bool = True):
    headlines = []
    texts = []
    for message in messages:
        text = message.get('text', '')
        media_type = message.get('media_type', '')
        if not text or media_type:
            continue
        txt = None
        for token in text:
            if isinstance(token, dict):
                type = token.get('type', '')
                if type and type == 'bold':
                    content = process_string(token.get('text', ''))
                    if content:
                        headlines.append(content)
                        break
            elif not title_only and not txt:
                content = process_string(token)
                if content:
                    txt = content
        else:
            if not title_only and txt:
                texts.append(txt)
    if title_only:
        return headlines
    else:
        return headlines, texts
    

channel = '<CHANNEL>'
with open(f'tg\{channel}.json', 'rb') as file:
    data = json.loads(file.read())

messages: List[Dict[str, str]] = data.get('messages', [])

headlines = parse_messages(messages)

with open(f'./scrape/training_data/{channel}.csv', 'w', newline='', encoding='utf-8') as csvfile:
    writer = csv.writer(csvfile, quoting=csv.QUOTE_ALL)
    for headline in headlines:
        writer.writerow([headline])