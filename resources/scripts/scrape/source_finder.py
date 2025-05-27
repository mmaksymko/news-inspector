import re
# from profession_finder import find_professions
import scrape.news_scraper as ns
from string_utils import remove_symbol, strip_punctuation_and_spaces


import spacy
from spacy.tokens import Doc, Span, Token
from spacy.language import Language
# from nlp import SpacyModelSingleton

nlp = spacy.load("uk_core_news_lg")

def contains_latin_characters(s):
    return bool(re.search('[a-zA-Z]', s))

def get_property(token: Token, property_name: str):
    return token.morph.get(property_name)[0] if len(token.morph.get(property_name)) != 0 else None

@Language.component("set_custom_entities")
def set_custom_entities(doc: Doc):
    new_ents = list(doc.ents)
    
    doc_length = len(doc)
    for i in range(doc_length):
        token = doc[i]
        if token.text in ("США", "Іран") and not any(ent.start <= token.i < ent.end for ent in doc.ents):
            span = Span(doc, token.i, token.i + 1, label="LOC")
            new_ents.append(span)
        elif not token.ent_type_:
            if contains_latin_characters(token.text):
                foreign = True
            elif token.text.isupper() and len(token.text) > 1:
                foreign = False
            else:
                continue
            span = Span(doc, token.i, token.i + 1, label="ORG")
            foreign = "Yes" if foreign else "No"
            morph_str = f"{str(token.morph) + '|' if token.morph else ''}Foreign={foreign}"

            if i + 1 < len(doc):
                next_token = doc[i + 1]
                gender = get_property(next_token, 'Gender')
                if gender:
                    morph_str += f"|Gender={gender}"
            token.set_morph(morph_str)
            
    doc.ents = new_ents
    return doc

nlp.add_pipe("set_custom_entities", after="ner")


def remove_repeated_words(text):
    pattern = re.compile(r'\b(\w+)\s+\1\b', re.IGNORECASE)
    return pattern.sub(r'\1', text)

def title_if_not_uppercase(text: str):
    return text if text.isupper() else text.title()

def entity_dict(string: str, label: str, case: str, foreign: bool, plurality: bool, gender: str):
    return {
        'name': string,
        'label': label,
        'case': case,
        'foreign': foreign,
        'plurality': plurality,
        'gender': gender
    }

def find_entities(doc: Doc) -> list[dict[str, str, str | bool, bool, str | bool]]:
    entities = []
    for ent in doc.ents:
        tokens = [token for token in ent]
        label = ent.label_
        
        abbr = get_property(tokens[0], 'Abbr')
        foreign = bool(get_property(tokens[0], 'Foreign'))
        first_word_case = get_property(tokens[0], 'Case')
        plurality = True if get_property(tokens[0], 'Number') == 'Plur' else False
        first_word_gender = get_property(tokens[0], 'Gender')

        if len(tokens) == 1:
            if ent.text.isupper():
                entity = ent.text
            else:
                entity = ent.lemma_.title()
        else:
            entity = ent.text
        
            if match := re.search('[a-zA-Z]', entity) and re.search('[а-яА-яієєґІЄЇҐ]', entity):
                index = match.start()
                words_cnt = entity[:index].strip().count(' ') + 1
                first_entity = ' '.join([token.lemma_ for token in tokens[:words_cnt]]).strip()

                plurality = True if get_property(tokens[0], 'Number') else False
                first_word_gender = get_property(tokens[0], 'Gender')

                entities.append(entity_dict(title_if_not_uppercase(first_entity), 'ORG', first_word_case, False, plurality, first_word_gender))
                entity = ' '.join([token.text for token in tokens[words_cnt:]])
                label = "ORG"
                foreign = True
                plurality = False
            
            if first_word_case not in ('Acc', 'Nom') and not abbr and not foreign == '':
                if label == "LOC":
                    if tokens[0].pos_ == 'ADJ':
                        if first_word_gender == 'Neut':
                            first_word = tokens[0].lemma_.rstrip('ий') + 'e'
                        elif first_word_gender == 'Fem':
                            first_word = tokens[0].lemma_.rstrip('ий') + 'а'
                        else:
                            first_word = tokens[0].lemma_
                        entity = ' '.join([first_word] + [token.lemma_ for token in tokens[1:]])
                elif first_word_case == 'Gen':
                    if tokens[0].pos_ in ('NOUN', 'PROPN'):
                        entity = ' '.join([tokens[0].lemma_] + [token.text for token in tokens[1:]])
                    elif tokens[0].pos_ == 'ADJ':
                        entity = ' '.join([token.lemma_ if token.pos_ != 'PROPN' else token.text for token in tokens])
        entities.append(entity_dict(title_if_not_uppercase(entity), label, first_word_case, foreign, plurality, first_word_gender))

    return entities


def generate_endings(ending_type: str, pre_endings: list[str] = ['и', 'ля', 'а', 'сти']) -> list[str]:
    return [ending + PAST_ENDING[ending_type] for ending in pre_endings]

CLAUSES = ['йдеться', 'за словами', 'за даними', 'згідно з', 'відповідно до', 'за версією', 'пише', 'каже', 'інформує', 'з посиланням на', 'посилаючись на']
DECLENSION_CLAUSES = ['зазнач', 'розмі', 'визн', 'заяв', 'підтверд', 'повідом', 'звітув', 'інформув', 'сказ', 'пис', 'відзнач', 'прокоментув', 'поясн']

PAST_ENDING = {
    'masuline': 'в',
    'feminine': 'ла',
    'neutral': 'ло',
    'plural': 'ли',
    'infinitive': 'ти',
    'present': '(?:є|е)',
    'passive': 'но'  
}

ENDINGS = {
    'masculine_past': generate_endings('masuline'),
    'feminine_past': generate_endings('feminine'),
    'neutral_past': generate_endings('neutral'),
    'plural_past': generate_endings('plural'),
    'infinitive': generate_endings('infinitive'),
    'present': generate_endings('present', ['ля', 'жу', "щує"]),
    'passive': generate_endings('passive', ['ле', 'а', 'ще'])
}

DECLENSION_ENDINGS = [ending for endings_list in ENDINGS.values() for ending in endings_list]

clauses = '|'.join(CLAUSES)
declension_clauses = '|'.join(DECLENSION_CLAUSES)
declension_endings = '|'.join(DECLENSION_ENDINGS)

prefix_match = r'[^\.\?!\n]*?'
clause_pattern = fr'(?:\b(?:(?:{clauses})|({declension_clauses})({declension_endings}))\b)'
suffix_match = r'[^\.\?!,–\-\n]*?'  # Match anything after the clauses/declension but before sentence-ending punctuation

optional_preceeding_quotation = r'(?:(\"[^\"]+?\")\s*[\.,!?]?\s*–?\s*)?'
optional_colon_quotation = r'(?:: (\".+\"))?'

procedding_argument = r'(?:, що ([^\.\?!\n]*))?'
end_pattern = r'[\.\?!,–\-\n]'

non_greedy_until_next_clause = fr'(?=\b(?:(?:{clauses})|({declension_clauses})({declension_endings}))\b|{end_pattern}))'

pattern = fr'{optional_preceeding_quotation}({prefix_match}{clause_pattern}{suffix_match})\b(?:(?=[\.\?!,–\-:]){optional_colon_quotation}{procedding_argument}{non_greedy_until_next_clause}'
external_source_pattern = r'(?:джерело)\s*:\s*([^\.,\n]+)'
exact_phrase_pattern = r'(?:дослівно(?: )(\w+)?): ([^\.\n]+)'

anonimity_pattern = r'(на\s?(власні|власне|анонімне|анонімні)?\s?(джерело|джерела))([^\.,!?]*)'

print(pattern)

def normalize_hyphens(text: str):
    return text.replace(' - ', '-')

def extract_entities_from_subtree(token: Token, entity_type: str):
    return ' '.join([tok.text for tok in token.subtree if tok.ent_type_ == entity_type])

def full_profession(doc: Doc, substring: str):
    substring_tokens = substring.split()
    
    doc_text = ' '.join([token.text for token in doc])
    normalized_doc_text = normalize_hyphens(doc_text).lower()
    normalized_substring = normalize_hyphens(substring).lower()
    
    start_i = normalized_doc_text.find(normalized_substring)
    
    if start_i != -1:
        end_i = start_i + len(normalized_substring)
        
        # Find the token index corresponding to the end of the substring
        token_index = None
        for i, token in enumerate(doc):
            if token.idx >= end_i:
                token_index = i
                break        
        
        if token_index is not None:
            result = []
            location = False
            person = False
            organization = False
            for i in range(len(substring_tokens), 0, -1):
                for token in doc[token_index - i].subtree:
                    if not location and token.ent_type_ == 'LOC':
                        location = extract_entities_from_subtree(token, 'LOC')
                    elif not person and token.ent_type_ == 'PER':
                        person = extract_entities_from_subtree(token, 'PER')
                    elif not person and token.ent_type_ == 'ORG':
                        organization = extract_entities_from_subtree(token, 'ORG')
                    elif not (location or person) and token not in result:
                        result.append(remove_repeated_words(token.text))
                    else:
                        break
            profession = ' '.join(result).strip()
            return profession 
    return []

def normalize_quotes(text: str) -> str:
    text = re.compile(r'[«»‚‛“”„‟‹›⹂]').sub('"', text)
    if not text:
        return text
    doc = nlp(text)
    
    text_chars = list(text)
    
    for entity in doc.ents:
        if entity.label_ == 'LOC' or entity.lemma_ == 'іран':
             continue
        start, end = entity.start_char, entity.end_char
        
        if start > 0 and text_chars[start - 1] == '"':
            text_chars[start - 1] = '«'
        
        if end < len(text_chars) and text_chars[end] == '"':
            text_chars[end] = '»'
    
    return ''.join(text_chars)        

def get_entities(part: str):
    def get_full_subject(doc: Doc):
        for token in doc:
            if token.dep_ in ('nsubj', 'csubj'):
                return {
                    "name": ' '.join([leaf.text for leaf in token.subtree if not (leaf.text.lower() == 'агентство' or leaf.dep_.startswith('flat:') or leaf.ent_type_ == 'LOC')]),
                    "animacy": get_property(token, 'Animacy')
                }
        return None
    def replace_shortened_entity_with_full_subject(subject: dict[str, str], entities):
        def process_entity(entity, subject_name, entities, value):
            for i in range(len(entity)):
                if entity[i] in subject_name:
                    if value != "people" and len(entities["people"]) == 2:
                        entities["people"] = [person for person in entities["people"] if person not in subject_name]
                    entity[i] = subject_name
                    return True
                if subject_name in entity[i]:
                    return True
            return False
        
        subject_name = subject["name"]
        for value in entities.keys():
            entity = entities[value]
            if process_entity(entity, subject_name, entities, value):
                return
        else:
            if subject["animacy"] == "Anim":
                entities["professions"] = [subject["name"]]
            else:
                entities["organizations"] = [subject["name"]]
            
    doc = nlp(part)
            
    # professions = find_professions(part)
    # professions = [full_profession(doc, prof) for prof in professions]

    entities = find_entities(doc)
    entities: dict[str, list] = {
        # "professions": professions,
        "organizations" : [entity["name"] for entity in entities if entity["label"] == 'ORG'],
        "locations" : [entity["name"] for entity in entities if entity["label"] == 'LOC'],
        "people" :  [entity["name"] for entity in entities if entity["label"] == 'PER'],
        "anonymous": [],
        "quotes": []
    }
        
    subject = get_full_subject(doc)
    if subject:
        replace_shortened_entity_with_full_subject(subject, entities)
    if not (entities["locations"] or entities["organizations"] or entities["people"]):
        # or entities["professions"]):
        if subject:
            entities["organizations"] = [subject["name"]]
        else:
            entities["anonymous"].append(part)
    return entities

def get_groups(match: re.Match): 
    return [strip_punctuation_and_spaces(match.group(i)) if match.group(i) else None for i in range(1, len(match.groups()) + 1)]

def create_entity(professions: list[str] = [], organizations: list[str] = [], locations: list[str] = [], persons: list[str] = [], anonymous: list[str] = [], quotes: list[str] = []):
    return {
        # "professions": professions,
        "organizations": organizations,
        "locations": locations,
        "people": persons,
        "anonymous": anonymous,
        "quotes": quotes
    }

def get_first_letter(word):
    return next((char.upper() for char in word if char.isalpha()), '')

def find_sources(article: str):
    def unique_anonymous(sources):
        docs = [nlp(entity) for entity in sources]
        lemmatized_texts = [{token.lemma_ for token in doc if token.pos_ in ('NOUN', 'ADJ', 'ADV', 'VERB')} for doc in docs]
        return [sources[i] for i, lemmas in enumerate(lemmatized_texts) if not any(set(lemmas).issubset(set(other)) for j, other in enumerate(lemmatized_texts) if i != j)]
    def get_first_or_none(list):
        return list[0] if list else None
    def process_acronyms(value):
        name, acronym = None, None
        if len(value) == 2:
            org1 = min(value[0], value[1], key=len)
            org2 = max(value[0], value[1], key=len)
            if is_acronym(org1, org2):
                name = org2
                acronym = org1
        elif len(value) == 1:
            name = value[0]
            acronym = ''.join([get_first_letter(word).upper() for word in value[0].split()]) if len(name.split()) > 1 else None
        return {"name": name, "acronym": acronym}
        
    article = normalize_quotes(article)
    article += '' if article.endswith(('.', '!', '?')) else '.'
    matches = re.finditer(pattern, article, re.IGNORECASE)
    anonimity_matches = re.finditer(anonimity_pattern, article, re.IGNORECASE)
    exact_matches = re.finditer(exact_phrase_pattern, article, re.IGNORECASE)
    external_source_matches = re.finditer(external_source_pattern, article, re.IGNORECASE)
    
    sources = {
        "personalized": [],
        "anonymous": []
    }
    
    for match in matches:
        groups = get_groups(match)

        part = groups[1]
        entities = get_entities(part)
        entities["anonymous"] = unique_anonymous(entities["anonymous"])
                
        if groups[0]:
            # print(groups[0], strip_punctuation_and_spaces(groups[0]))
            entities["quotes"].append(strip_punctuation_and_spaces(groups[0]))
            # print(groups[0])
        if groups[5]:
            # print(groups[5], strip_punctuation_and_spaces(groups[5]))
            entities["quotes"].append(strip_punctuation_and_spaces(groups[5]))
        sources["personalized"].append(entities)
        
    anonymous = unique_anonymous([match.group(3)+match.group(4) for match in anonimity_matches if match.group(3) and match.group(4)])
    sources["anonymous"] = anonymous  

    for match in exact_matches:
        groups = get_groups(match)
        sources["personalized"].append(create_entity(organizations=[groups[0]], quotes=[groups[1]]))
    
    for match in external_source_matches:
        groups = get_groups(match)
        sources["personalized"].append(create_entity(organizations=[groups[0]]))
    
    for source in sources["personalized"]:
        for key, value in source.items():
            if key == 'organizations':
                source[key] = process_acronyms(value)
            else:
                source[key] = get_first_or_none(value)
                source[key] = source[key][0].upper() + source[key][1:] if source[key] and source[key][0].islower() else source[key]
            
            new_key = 'anonymous' if key == 'anonymous' else key[:-1] if key.endswith('s') else 'person' if key == 'people' else key
            if new_key != key:
                source[new_key] = source.pop(key)
    
    return sources
    
    
def get_sources_from_headline(headline: str):
    index = max(headline.rfind(' – '), headline.rfind(' - '))
    sources = []
    if index != -1:
        sources.append(headline[index+3:].strip())
    
    result = []
    for source in sources:
        result.append(get_entities(source))
    return result
    
def is_acronym(lhs, rhs):
    return lhs == ''.join([word[0].capitalize() for word in rhs.split()])

# print(ns.scrape("https://www.pravda.com.ua/news/2022/10/7/7362961/"))

# for article in articles:
#     sources = find_sources(article)
#     for source in sources["personalized"]:
#         src = ' '.join([item for item in [source["profession"], source["organization"]["name"], source["location"], source["person"], source["anonymous"]] if item])
#         quoted_src = ': '.join([src, source["quote"]]) if source.get("quote") else src
#         print(quoted_src)
#     print('Anonymous:', sources["anonymous"])
#     print('-----------------------------------')
    
