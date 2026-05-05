from tokenization import tokenize
import json
import urllib.request

def invoke(action, **params):
    """
    Sends a request to a locally running AnkiConnect server with the specified action and parameters
    """

    payload = json.dumps({"action": action, "version": 6, "params": params}).encode('utf-8')
    response = json.load(urllib.request.urlopen(urllib.request.Request('http://localhost:8765', payload)))
    if response['error'] is not None:
        raise Exception(response['error'])
    return response['result']


def get_having_cards(deck_names: list[str], deck_front_side_names: list[str]):
    """
    For each deck name, finds all cards in decks whose names start with that term
    """

    all_cards = set()
    for i in range(len(deck_names)):
        card_ids = invoke('findCards', query=f'deck:"{deck_names[i]}*"')
        cards_info = invoke('cardsInfo', cards=card_ids)

        for card in cards_info:
            try:
                front_content = card['fields'][deck_front_side_names[i]]['value']
                all_cards.add(front_content)
            except Exception as e:
                pass

    return all_cards

def valid_words(sentence: str, deck_names: list[str], deck_front_side_names: list[str]):
    """
    For each sentence, filters for words that are not already
    present on any card’s front field and returns a list of missing words that would result in new cards
    """

    tokenized = tokenize(sentence)
    all_cards = get_having_cards(deck_names, deck_front_side_names)
    result = {}
    for k, v in tokenized.items():
        missing_words = [w for w in v if w not in all_cards]
        if missing_words:
            result[k] = missing_words

    return result

def get_deck_name_and_front_card_name():
    """
    Queries Anki for all deck names, dropping the last two entries, which are system decks.

    For each deck, finds one card, inspects its first field, and assumes this is the front side’s field name
    """
    decks = invoke("deckNames")[:-2]
    cards_front_names = []
    for deck in decks:
        card_ids = invoke('findCards', query=f'deck:"{deck}*"')
        card_info = invoke('cardsInfo', cards=[card_ids[0]])[0]
        front_field = list(card_info['fields'].keys())[0]
        cards_front_names.append(front_field)
    return zip(decks, cards_front_names)
