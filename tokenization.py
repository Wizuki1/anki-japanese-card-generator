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
    "Tokenizes the text"
    
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
    """
    Cleans up generated prompt strings by removing lines that have empty values after a colon
    """

    filtered_lines = []

    for line in text.strip().split('\n'):
        if ':' in line:
            key, value = line.split(':', 1)
            if value.strip():
                filtered_lines.append(line)
        else:
            if line.strip():
                filtered_lines.append(line)

    return '\n'.join(filtered_lines)


def get_surface(sentence: str, needed_word: str):
    """
    Tokenizes the provided sentence and returns thet surface form (the actual inflected form in context)
    of the first occurrence of the specified dictionary form
    """

    tokens = tokenizer_obj.tokenize(sentence, mode)

    for token in tokens:
        word = token.dictionary_form()
        surface = token.surface()
        if word == needed_word:
            return surface


def get_japanese_pos(raw_tag):
    """
    Maps POS tags from a source format (e.g., “n”, “v1”, “v5k”)
    to their Japanese label equivalents, handling a few special cases for verb classes
    """

    exact_map = {
        'n': '名詞',
        'pn': '代名詞',
        'adv': '副詞',
        'vt': '他動詞',
        'vi': '自動詞',
        'adj-na': 'な形容詞',
        'adj-i': 'い形容詞',
        'v1': '一段動詞',
    }

    if raw_tag.startswith('v5'):
        return '五段動詞'
    if raw_tag.startswith('v2'):
        return '二段動詞 (文語)'
    if raw_tag.startswith('vs'):
        return 'サ変動詞'

    return exact_map.get(raw_tag, raw_tag)


def process_word_types(raw_string):
    """
    Splits a combined POS string (“n・adj-i”) into individual tags,
    converts each to the Japanese POS label, and returns a comma-separated string of the translated tags
    """

    tags = raw_string.split('・')
    translated = [get_japanese_pos(tag) for tag in tags if get_japanese_pos(tag) != tag]
    if not translated:
        return ''
    return ', '.join(translated)

def get_prompts(preprocessed_text: tuple[str]):
    """
    Builds a formatted prompt string describing the word, its reading, word type, meanings, and context sentence
    """

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
