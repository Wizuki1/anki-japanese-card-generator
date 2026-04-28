from valid_words import valid_words
from tokenization import get_prompts
from sort_all import give_n_best
from qwen_generating import get_answer
import re
import requests
import base64
import os
import ollama
import torch
from image_creating import image_generating, unload_sdxl
from time import sleep
from sentence_TTS import create_audio
import gc
import shutil
import uuid

ANKI_URL = "http://localhost:8765"

def add_anki_card(word_data: list[dict], img_path: str, word_audio_path: str, sentence_audio_path: str, deck_name: str, img: bool, voice: bool):
    """Creates and uploads flashcards to a specified Anki deck using the AnkiConnect API"""

    def to_base64(path):
        with open(path, "rb") as f:
            return base64.b64encode(f.read()).decode('utf-8')

    unique_id = str(uuid.uuid4())

    img_name = f"{unique_id}.jpg"
    w_audio_name = f"{unique_id}_word.mp3"
    s_audio_name = f"{unique_id}_sent.mp3"

    front_html = f"""
    <div style="text-align: center; padding-top: 40px;">
        <span style="font-size: 80px; font-weight: bold; color: #3A6DF8;">{word_data['word']}</span>
    </div>
    """

    back_html = f"""
    <div style="text-align: center; font-family: 'Segoe UI', Tahoma, Geneva, sans-serif; color: #333;">

        <!-- Слово с фуриганой сверху -->
        <div style="margin-bottom: 25px;">
            <ruby style="font-size: 60px; font-weight: bold; color: #3A6DF8;">
                {word_data['word']}
                <rt style="font-size: 24px; color: #4485EE; font-weight: normal;">{word_data['reading']}</rt>
            </ruby>
        </div>

        <!-- Перевод слова -->
        <div style="font-size: 28px; color: #e74c3c; font-weight: bold; margin-bottom: 25px;">
            {', '.join(word_data['selected_meanings'])}
        </div>

        <!-- Блок с примером (выделен серым фоном со скругленными углами) -->
        <div style="background-color: #f8f9fa; padding: 20px; border-radius: 15px; display: inline-block; margin-bottom: 25px; box-shadow: 0 4px 6px rgba(0,0,0,0.05); text-align: left; max-width: 80%;">
            <div style="font-size: 22px; color: #2c3e50; margin-bottom: 10px;">
                {word_data['example_jp']}
            </div>
            <div style="font-size: 18px; color: #7f8c8d; font-style: italic;">
                {word_data['example_translation']}
            </div>
        </div>

        <br>

        <!-- Картинка (скругленные углы и небольшая тень) -->
        <div style="margin-bottom: 20px;">
            <img src='{img_name}' style="max-width: 300px; height: auto; border-radius: 15px; box-shadow: 0 4px 10px rgba(0,0,0,0.15);">
        </div>

        <!-- Озвучка -->
        <div style="margin-top: 15px;">
            [sound:{w_audio_name}] [sound:{s_audio_name}]
        </div>

    </div>
    """

    if img and voice:
        payload = {
            "action": "addNote",
            "version": 6,
            "params": {
                "note": {
                    "deckName": deck_name,
                    "modelName": "Basic",
                    "fields": {
                        "Word": front_html,
                        "Meaning": back_html
                    },
                    "picture": [{
                        "data": to_base64(img_path),
                        "filename": img_name,
                        "fields": ["Back"]
                    }],
                    "audio": [
                        {
                            "data": to_base64(word_audio_path),
                            "filename": w_audio_name,
                            "fields": ["Back"]
                        },
                        {
                            "data": to_base64(sentence_audio_path),
                            "filename": s_audio_name,
                            "fields": ["Back"]
                        }
                    ]
                }
            }
        }
    elif img:
        payload = {
            "action": "addNote",
            "version": 6,
            "params": {
                "note": {
                    "deckName": deck_name,
                    "modelName": "Basic",
                    "fields": {
                        "Word": front_html,
                        "Meaning": back_html
                    },
                    "picture": [{
                        "data": to_base64(img_path),
                        "filename": img_name,
                        "fields": ["Back"]
                    }]
                }
            }
        }
    elif voice:
        payload = {
            "action": "addNote",
            "version": 6,
            "params": {
                "note": {
                    "deckName": deck_name,
                    "modelName": "Basic",
                    "fields": {
                        "Word": front_html,
                        "Meaning": back_html
                    },
                    "audio": [
                        {
                            "data": to_base64(word_audio_path),
                            "filename": w_audio_name,
                            "fields": ["Back"]
                        },
                        {
                            "data": to_base64(sentence_audio_path),
                            "filename": s_audio_name,
                            "fields": ["Back"]
                        }
                    ]
                }
            }
        }
    else:
        payload = {
            "action": "addNote",
            "version": 6,
            "params": {
                "note": {
                    "deckName": deck_name,
                    "modelName": "Basic",
                    "fields": {
                        "Word": front_html,
                        "Meaning": back_html
                    }
                }
            }
        }

    return requests.post(ANKI_URL, json=payload).json()

def connect_all(text: str, deck_names: list[str], deck_front_side_names: list[str], n: int, generate_image: bool, generate_voice: bool, deck_name: str, offload_cpu: bool):
    """
    Executes the full pipeline to process text and generate Anki cards.

    This function orchestrates the entire workflow: extracts unique words, selects
    the top-N most valuable terms, generates AI prompts, fetches word data,
    creates images and audio, and finally uploads the resulting cards to Anki
    via AnkiConnect.
    """
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    unique_words = valid_words(text, deck_names, deck_front_side_names)
    prompts = get_prompts(give_n_best(unique_words, n))

    torch.cuda.empty_cache()
    gc.collect()
    sleep(1)

    answers = []
    image_tasks = []

    k = 0
    for prompt in prompts:
        answer = get_answer(prompt)
        answers.append(answer)
        match = re.search(r"Word:\s*(.+)", prompt)
        answers[k]['word'] = match.group(1).strip()
        image_tasks.append({
                  'image_prompt': answer['image_prompt'],
                  'filename': match.group(1).strip() + '.jpeg'
              })
        k += 1

    ollama.chat(model='my-model', keep_alive=0)
    torch.cuda.empty_cache()
    sleep(1)

    if generate_image and device == 'cuda':
        for task in image_tasks:
            image_generating(task['image_prompt'], task['filename'], offload_cpu)

        unload_sdxl()
        sleep(1)
    if generate_voice:
        sentences = [x['example_jp'] for x in answers]
        words = [x['word'] for x in answers]
        create_audio(sentences, words)

    k = 0
    for i in answers:
        add_anki_card(i, f'images/{image_tasks[k]['filename']}', f'audio/{i['word']}_word.mp3', f'audio/{i['word']}_sent.mp3', deck_name, generate_image, generate_voice)
        k += 1

    folders_to_clean = ['audio', 'images']

    for folder in folders_to_clean:
        if os.path.exists(folder):
            shutil.rmtree(folder)
        os.makedirs(folder)