import json
from pathlib import Path
import re

class LocalDictionary:
    def __init__(self, dict_folder_path):
        """
        Initialises the dictionary by loading all term_bank JSON files from the folder into a Hash Map
        """
        self.lookup_table = {}
        self.load_yomitan_dictionary(dict_folder_path)

    def load_yomitan_dictionary(self, folder_path):
        """
        Retrieves information from the dictionary and presents it in a use-friendly format
        """
        folder = Path(folder_path)
        for file_path in folder.glob("term_bank_*.json"):
            with open(file_path, 'r', encoding='utf-8') as f:
                terms = json.load(f)

                for term_data in terms:
                    word = term_data[0]
                    reading = term_data[1]
                    meanings = term_data[5]

                    if word not in self.lookup_table:
                        self.lookup_table[word] = []

                    text = re.sub(r'^(?:.*\n){2}', '', meanings[0])
                    text = re.sub(r'〘.*?〙', '', text)

                    text = text.replace('\n', ' ')

                    text = re.sub(r'\s+', ' ', text).strip()
                    text = text.replace(';', ',')

                    self.lookup_table[word].append({
                        "reading": reading,
                        "word_type": re.search(r'〘(.*?)〙', meanings[0]).group(1),
                        "meanings": text[:-1],
                    })

    def search(self, word):
        return self.lookup_table.get(word, None)
