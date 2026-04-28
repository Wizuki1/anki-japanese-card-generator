from sudachipy import dictionary
import spacy
from jmdict_extractor import LocalDictionary
from vector_sim import get_best_meaning
import re

tokenizer_obj = dictionary.Dictionary().create()
nlp = spacy.load("ja_ginza")
mode = tokenizer_obj.SplitMode.B
my_dict = LocalDictionary("EN-JP JMdict")
japanese_pattern = re.compile(r'[\u3040-\u30FF\u4E00-\u9FFF]')

def tokenize(text: str):
    words_in_context: dict[str, list[str]] = {}
    seen_words: set = set()

    doc = nlp(text)
    for sentence in doc.sents:
        tokens = tokenizer_obj.tokenize(sentence.text, mode)
        words_in_context[sentence.text] = []
        for token in tokens:
            pos = token.part_of_speech()
            word = token.dictionary_form()
            if pos[0] in ('名詞', '動詞', '形容詞', '副詞', '代名詞', '固有名詞', '形状詞') and word not in seen_words \
                    and len(re.findall(japanese_pattern, word)) == len(word):
                words_in_context[sentence.text].append(word)
                seen_words.add(word)
        if not words_in_context[sentence.text]:
            del words_in_context[sentence.text]
    return words_in_context


def filter_empty_fields(text):
    filtered_lines = []

    for line in text.strip().split('\n'):
        # Проверяем, есть ли в строке двоеточие
        if ':' in line:
            # Разделяем на ключ и значение
            key, value = line.split(':', 1)
            # Если после двоеточия есть что-то кроме пробелов — оставляем
            if value.strip():
                filtered_lines.append(line)
        else:
            # Если двоеточия нет, но строка не пустая — тоже оставляем
            if line.strip():
                filtered_lines.append(line)

    return '\n'.join(filtered_lines)


def get_surface(sentence: str, needed_word: str):
    tokens = tokenizer_obj.tokenize(sentence, mode)

    for token in tokens:
        word = token.dictionary_form()
        surface = token.surface()
        if word == needed_word:
            return surface


def get_japanese_pos(raw_tag):
    exact_map = {
        'n': '名詞',
        'pn': '代名詞',
        'adv': '副詞',
        'vt': '他動詞',  # Переходный
        'vi': '自動詞',  # Непереходный
        'adj-na': 'な形容詞',
        'adj-i': 'い形容詞',
        'v1': '一段動詞',
    }

    if raw_tag.startswith('v5'):
        return '五段動詞'
    if raw_tag.startswith('v2'):
        return '二段動詞 (文語)'  # Указываем, что это архаика (бунго)
    if raw_tag.startswith('vs'):
        return 'サ変動詞'

    return exact_map.get(raw_tag, raw_tag)


def process_word_types(raw_string):
    tags = raw_string.split('・')
    translated = [get_japanese_pos(tag) for tag in tags if get_japanese_pos(tag) != tag]
    if not translated:
        return ''
    return ', '.join(translated)

def get_prompts(preprocessed_text: tuple[str]):
    prompts = []

    for word, sentence in preprocessed_text:
        word_info = my_dict.search(word)
        if not word_info:
            prompts.append(
f'''Word: {word}
Context sentence: "{sentence}"'''
                )
        elif len(word_info) == 1:
            word_type = process_word_types(word_info[0]['word_type'])
            prompts.append(filter_empty_fields(
f'''Word: {word}
Reading: {word_info[0]['reading']}
Word type: {word_type}
Meanings: {word_info[0]['meanings']}
Context sentence: "{sentence}"'''
                ))
        else:
            word_info = get_best_meaning(get_surface(sentence, word), sentence, word_info)
            word_type = process_word_types(word_info['word_type'])
            prompts.append(filter_empty_fields(
f'''Word: {word}
Word type: {word_type}
Reading: {word_info['reading']}
Meanings: {word_info['meanings']}
Context sentence: "{sentence}"'''
                ))
    return prompts