import re
import string

PUNCTUATION = string.punctuation + '‒'

def normalize_quotes(text, opening='«', closing='»'):
    result = []
    is_opening = True
    for char in text:
        if char in ('"', '″'):
            if is_opening:
                result.append(opening)
            else:
                result.append(closing)
            is_opening = not is_opening
        else:
            result.append(char)
    string = ''.join(result)
    string = re.sub(r'[“‹„«]', opening, string)    
    string = re.sub(r'[”›»]', closing, string)    
    return string

def process_escape_sequences(text):
    text = re.sub(r'\r\n', r'\n', text)
    text = re.sub(r'\xa0|\t|\u2009', ' ', text)
    text = re.sub(r'(^[.,!?]\s*)|\u200b|\u200e|/?noindex', '', text)
    text = text.replace('\\n', '\n')
    text = re.sub(r'\n+', '\n', text)
    text = re.sub(r'(\s)+', r'\1', text)
    text = re.sub(r'\s+([.!?:;])', r'\1', text)
    text = re.sub(r'«\s*', r'«', text)
    text = re.sub(r'\s*»', r'»', text)
    return text

def join_paragraphs(text):
    text = re.sub(r'\n', r' ', text)    
    return text

def strip_punctuation_and_spaces(text):
    return text.translate(str.maketrans('', '', PUNCTUATION)).strip()

def remove_symbol(text, mark):
    return text.replace(mark, '')

EMOJIS_REGEX = re.compile("["
        u"\U0001F600-\U0001F64F"  # emoticons
        u"\U0001F300-\U0001F5FF"  # symbols & pictographs
        u"\U0001F680-\U0001F6FF"  # transport & map symbols
        u"\U0001F1E0-\U0001F1FF"  # flags (iOS)
        u"\U00002500-\U00002BEF"  # chinese char
        u"\U00002702-\U000027B0"
        u"\U000024C2-\U0001F251"
        u"\U0001f926-\U0001f937"
        u"\U00010000-\U0010ffff"
        u"\u2640-\u2642" 
        u"\u2600-\u2B55"
        u"\u200d"
        u"\u23cf"
        u"\u23e9"
        u"\u231a"
        u"\ufe0f"  # dingbats
        u"\u3030"
        "‼"
    "]+", re.UNICODE)

def remove_emojis(data: str) -> str:
    return re.sub(EMOJIS_REGEX, '', data)