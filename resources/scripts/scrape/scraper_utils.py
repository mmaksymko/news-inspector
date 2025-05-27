from collections import deque
import re
from string import punctuation

from bs4 import BeautifulSoup, Comment, NavigableString


SOCIALS = ['сторінки', 'канали', 'telegram', 'телеграм', 'viber', 'вайбер', 'instagram', 'інстаграм', 'facebook', 'фейсбук', 'twitter', 'твіттер', 'твітер', 'youtube', 'ютуб', 'google news']
SOCIALS_STRING = '|'.join(SOCIALS)

ADS_REGEXEN = [
    fr'(?:(?:підпи(суй|шіть|ши)|приєд(нуй|най)|читай)(?:те)?с?[ья]?)\s*(новини)?\s*([«"]?\s*[^\.!,?:]*\s*[»"]?)?\s*(?:на(с|шому)\s*)?(?:у|в|до|на)\s*(?:на(?:ш|с|шого|ших|ші|шому)?\s*({SOCIALS_STRING}))?',
    fr'(слідкуйте за нашою сторінкою)|(в нас)\s(у|в|на)\s*({SOCIALS_STRING})',
    fr'^(дізнавай(те)?с[ья])|(читай(те)?)\s*.+(на|у|в)/+({SOCIALS_STRING})і?',
    fr'(ми у соцiальних мережах)|(наші соцмережі)|(наш ({SOCIALS_STRING})\b)',
    r'^((ми [уво] (соц)?мережах)|(cлідкуйте за нами у соцмережах)|(слідкуйте за останніми новинами)|(долучайтеся, читайте та дивіться новини)|(усі найцікавіші новини)|(підписатись на новини)(підписатися на оновлення)|(слідкуй за нами)|(підписатися на оновлення))',
    r'^(поширити|поділитис[ья])\s*(([ув]\s*соцмережах)|(новиною)|(це)|(цим))$',
    r'використ(ання(м)?|овує(мо)?)[^!?\.]*cookies',
    r'\[email\sprotected\]',
    r'^((related posts)|(share this post[!?\.]?)|(login)|(you must be logged in to post a comment)|(comments)|(comments are closed)|(more in)|(search)|(no media source currently available))$',
    r'підтримай(те)? (нас|про[еє]кт)',
    r'^((долучайтеся до спільноти)|(слідкуйте за новинами)|(раніше на))',
    r'^\s*(підписа|поділи)тис[ья]\s*$',
    r'підпи(суйс|ш(ис|ітьс))[ья][^!?\.]*розсилку',
    r'[ву]сі права захищен(о|і)', 
    r'^((повідомити про помилку)|([cс]ообщить об опечатке))',
    r'©',
    r'^(топ-)?\d*\s*перегляд(и|ів)$',
    r'^політика конфіденційності$',
    r'^\s*(схожі|останні|[ву]сі|популярні|топ)?-?\s*новини\s*(партнерів)?\s*(\:)?\s*$',
    r'^((фото)|(відео)|(редактор)\s*\:)',
    r"^(\s*(новини)|(за темою)|(теми)|(тема тижня)|(близькі теми)|(матеріали по темі)|(статті на цю ж тему)|(пов[’']язані теми|статті)|(ще на цю тему)|(не пропустіть)|(актуальн[ео])|(це теж цікаво)|(головне сьогодні)|(важливе останнє)|(ще в розділі)|(вам також може сподобатися)|(що читати далі)|(почитати ще)|((схожі|свіжі) (записи|новини|матеріали|публікації))|(зараз читають)\s*)$",
    r"^((коментар(і)?)|(блог(и)?)|соціум|(категорія)|(обговорення)|(соціум)|(головне)|(докладно)|(найпопулярніше)|(гороскоп)|(популярне)|(аналітика)|(реклама\\s*партнерів?)|(скопійовано)|(навігація записів)|(рекомендуємо)|(контакти)|(ще з сайту)|(події)|(партнери)|(інші новини)|([гґ]аджети)|(важливе)|(peredplata)|(особистий кабінет)|(архів)|(вибір редакції)|(вам буде цікаво)|((ще)?\\s*більше новин)|(пов[`'’]язаний запис)|(ви пропустили)|(для вас)|(пошук)|(хмара тегів)|(популярні записи)|(топ за (тиж)? день)|(вас зацікавить)|(каталог змі)|(вибір редакції)|(здоров[`'’]я)|(культура)|(господарка)|(для чоловіків)|(назад)|(далі)|(Cтрічка новин)|(інформація)|(календар)|(швидкі новини)|(медіа гарячих новин)|(більше новин)|(розбивка на сторінки)|(категорії)|(популярне зараз)[\:\\.]?)$",
    r'ctrl\s*\+\s*enter',
    r'^((написати|залишити)\s*(відповідь|коментар))|(про|с)?(коментувати)|(копіювати)$',
    r'^((\d{2,4}[-\/]\d{2,4}[-\/]\d{2,4})(,\s*[воу]?)?\s*(\d{2}[-:]\d{2})?)$',
    r"^(понеділок|вівторок|середа|четвер|п'ятниця|субота|неділя)?[,во]?\s*\d*\:?\d*,?\s*\d*\s*(((січ|бер(ез)?|кв(іт)?|тр(ав)?|чер(в)?|лип|сер(п)?|вер(ес)?|жов(т)?|гр(уд)?)\.?(ня|ень)?)|(лют(ого|ий)?)|(лис(т)?(опад)?(а)?)|сьогодні|вчора),?\s*\d*\s*[,во]?\s*\d*\:?\d*$",
    r'(read?\s*in?\s*english)|(читать?\s*(на русском)|(по-русски))|(читати?\s*українською)',
    r'^((увійти за допомогою)|((ще)?\s*немає облікового запису)|(відновити пароль)|(пароль прийде на пошту)|(блокування реклами ввімкнено)|(перемістити в кошик)|(редагувати)(\?\.))$',
    r'^((підписатись на сповіщення про важливі новини від .+\?) | (дякуємо! тепер редактори знають.)|(ознайомтеся з іншими популярними матеріалами\:)|(Хочу отримувати\:)|(для згоди, будь ласка, натисніть «прийняти».)|(будь ласка, оберіть мову для подальшої роботи на сайті))$'
]

TAGS_TO_REMOVE = ['iframe', 'style', 'form', 'noscript', 'script', 'aside', 'figure']
STYLES_TO_REMOVE = ['display: none', 'opacity: 0']
NAMES_TO_REMOVE = [r'^all-cat$', r'(promo$|promo_block|^promo)', r'^nav$', r'^ad$', r'comment(s)?[-_]count', r'fb[\-:]comments', r'^seo$',
                r'^(?!newsFull_|header-wrapper$|article-content-header$|article-content-secondary-header$).*header.*', r'^show-all$', 'author', 
                'search', 'footer', '^aside$', 'subscribe', 'post-more', 'views', 'news-last', 'mc-war-memorial', 'reply', 'related', 'popup', 
                'date', 'donate', 'cookies', 'links', 'breadcrumb', 'disqus', 'google-news', 'modal-dialog', 'sign-modal', 'consultation', 
                'tooltip', 'comline', 'read_also', 'meta-comment', 'fb-root']
IDS_TO_REMOVE = ['SinoptikInformer', 'related_news'] + NAMES_TO_REMOVE
CLASSES_TO_REMOVE = ['tags-posts', 'copy_box', 'addthis', 'saboxplugin', 'offcanvas-menu', 'region-page-bottom', 'region-bottom', 'widget', 
                     'bottom-region', 'post-date', 'ribbon', r'(\bsecondary[-_\s]?(content)?\b)|(publication-card)'] + NAMES_TO_REMOVE

def clean_up(source: BeautifulSoup, deep_clean = True):    
    useless = source.find_all(TAGS_TO_REMOVE)
    useless += [element for element in source.find_all(style=True) if any(style in element.get('style', '') for style in STYLES_TO_REMOVE)]
    if deep_clean:
        useless += [element for id_ in IDS_TO_REMOVE for element in source.find_all(id=re.compile(id_, re.IGNORECASE))]
        useless += [element for class_ in CLASSES_TO_REMOVE for element in source.find_all(class_=re.compile(class_, re.IGNORECASE))]
    
    comments = source.find_all(string=lambda text: isinstance(text, Comment))
    for comment in comments:
        comment.extract()

    for element in useless:
        if element.name != 'body':
            element.decompose()
            

def has_anchor_parent(element):
    while element.parent:
        element = element.parent
        if element.name == 'a':
            return True
    return False

def contains_non_anchor(el):
    if has_anchor_parent(el):
        return False
    
    text = el.find(string=True, recursive=False)
    
    if el.name not in ('a') and (not el.find_all(recursive=False) or (text and text.strip())):
        return True
    
    for child in el.find_all(recursive=False):
        if contains_non_anchor(child):
            return True    
    return False

def extract_text_from_element(el):
    allowed_tags = ['span', 'em', 'strong', 'a', 'b', 'i', 'noindex', 'strong', 'weak', 's', 'u']
    return ' '.join([
        str(content).strip() if (hasattr(content, 'name') and content.name not in allowed_tags) 
        else content.get_text(strip=True, separator='\n')
        for content in el.contents 
        if isinstance(content, str) or (hasattr(content, 'name') and content.name in allowed_tags)
    ]).strip()
    
def find_paragraphs(page):
    strategies = [
        (True, None),
        (False, None),
        (True, ['p', 'h2', 'h3', 'h4', 'h5', 'li', 'div', 'em', 'strong']),
        (False, ['p', 'h2', 'h3', 'h4', 'h5', 'li', 'div', 'em', 'strong'])
    ]
    
    for until_comments, tags in strategies:
        if tags:
            paragraphs_src = find_paragraphs_source(page, tags, until_comments)
        else:
            paragraphs_src = find_paragraphs_source(page, until_comments=until_comments)
        if len(paragraphs_src) > 1:
            return paragraphs_src
    
    return None

def remove_low_gravity_elements(page):
    list_items = page.find_all()
    
    for item in list_items:
        if calculate_gravity_score(item) < 0:
            print(item, calculate_gravity_score(item))
            item.extract()
        

def normalize_tags(source: BeautifulSoup, tags_to_remove = ['em', 'b', 'i', 's', 'u', 'strong', 'weak']):
    for tag in tags_to_remove:
        for match in source.find_all(tag):
            match.unwrap()
    return source

def find_paragraphs_source(page: BeautifulSoup, text_elements =  ['p', 'h2', 'h3', 'h4', 'h5', 'li', 'blockquote', 'em', 'strong', 'noindex'], until_comments=True):
    h1 = page.find('h1')
    src = h1 if h1 else page
    
    with_anchor_parents = {el.get_text() for el in src.find_all_next() if has_anchor_parent(el)}
    queue = deque([el for el in src.find_all_next() if calculate_gravity_score(el) > 0])
    if not queue:
        queue = deque([el for el in src.find_all() if calculate_gravity_score(el) > 0])
    
    paragraphs = []
    while queue:
        tag = queue.popleft()
        if until_comments and any('comment' in cls.lower() for cls in tag.get('class', [])+[tag.get('id', '')]):
            break
        if tag.name in text_elements and not tag.find_parent(['header', 'footer', 'button']) and contains_non_anchor(tag) and\
            tag not in paragraphs and not (lambda a: a and 'monobank.ua' in a['href'])(tag.find('a', href=True)) and\
                not (tag.name == 'li' and tag.find_all() and all(child.name in ['span', 'a', 'img'] for child in tag.find_all())) and\
                    tag.get_text() not in with_anchor_parents:
            paragraphs.append(tag)
        queue.extend(tag.find_all(recursive=False))
    
    if all(el.name in {'li', 'h1', 'h2', 'h3', 'h4', 'h5', 'h6', 'em', 'strong'} for el in paragraphs):
        return []
    
    return paragraphs
    
def get_content(source):
    valid_elements = []
    counter = 1
    elements = [el for el in source if not any(re.search(regex, el.get_text(), re.IGNORECASE) for regex in ADS_REGEXEN)]
    read_also_regex = r'(читайте також)|(читайте также)|(також читайте)'
    for el in elements:
        if re.search(read_also_regex, el.get_text(), re.IGNORECASE):
            el.string = re.split(read_also_regex, el.get_text(), flags=re.IGNORECASE)[0]

    for el in elements:
        text = extract_text_from_element(el)
        if re.search(r'^[^:]+•', text, re.DOTALL):
            continue
        if not bool(re.search(rf'[^\s{re.escape(punctuation)}]', text)):
            continue
        if el.name == 'li':
            parent = el.parent
            if parent.name == 'ol':
                valid_elements.append(f"{counter}. {text}")
                counter += 1
            else:
                valid_elements.append(f"• {text}")
                counter = 1
        else:
            counter = 1
            if el.name == 'blockquote':
                text = f"«{text}»"
            else:
                valid_elements.append(text)
    return valid_elements

def has_anchor_before_text(element):
    text_found = False
    
    for child in element.children:
        if isinstance(child, NavigableString):
            if child.strip():
                text_found = True
        elif child.name == 'a':
            if not text_found:
                return True
        elif has_anchor_before_text(child):
            return True
            
    return False

def calculate_gravity_score(element):
    score = 0    
        
    if has_anchor_before_text(element):   
        score -= 50

    text_content = element.get_text(strip=True)
    text_length = len(text_content)
    
    if text_length > 0:
        score += min(50, text_length * 0.05)

    if element.name in ['p', 'article', 'main', 'section']:
        score += 15

    links = element.find_all('a')
    if len(links) > 0:
        link_density = len(' '.join([a.get_text(strip=True) for a in links])) / (text_length + 1)
        score -= link_density * 50

    num_images = len(element.find_all('img'))
    num_scripts = len(element.find_all('script'))
    score -= (num_images * 5) + (num_scripts * 10)
    
    depth = 0
    parent = element
    while parent:
        depth += 1
        parent = parent.parent
    score += 10 - depth

    if element.find_parent(['article', 'main']):
        score += 15
    if score < -50:
        score = -50
    elif score > 100:
        score = 100
    return score