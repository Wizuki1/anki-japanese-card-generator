from huggingface_hub import hf_hub_download
import os
import json
from valid_words import get_deck_name_and_front_card_name

def setup_project():

    with open('app_config.json', 'r', encoding='utf-8') as f:
        data = json.load(f)

    ness_models = True

    if not data['necessary_models']:

        print('Please choose models which you want to download:')
        print('1 - Only necessary models (Without image generating and TTS)')
        print('2 - Necessary models + image generating models')
        print('3 - Necessary models + image generating models + TTS')
        print('4 - Necessary models + TTS')
        variant = int(input('>>> '))
        while variant not in [1, 2, 3, 4]:
            print('Please choose valid option:')
            variant = int(input('>>> '))

        os.makedirs('model', exist_ok=True)

        print("Downloading Qwen...")

        qwen_model = hf_hub_download(
            repo_id="unsloth/Qwen3-4B-Instruct-2507-GGUF",
            filename="Qwen3-4B-Instruct-2507-Q5_K_M.gguf",
            local_dir="model")

        print("Qwen downloaded.")

    else:
        print('Please choose models which you want to download:')
        print('1 - image generating models')
        print('2 - image generating models + TTS')
        print('3 - TTS')
        print('4 - skip')
        variant = int(input('>>> '))
        while variant not in [1, 2, 3, 4]:
            print('Please choose valid option:')
            variant = int(input('>>> '))
        if variant == 4:
            return
        variant += 1

    image_models = variant in [2, 3]
    TTS_models = variant in [3, 4]



    if variant in [2, 3]:
        print('Downloading NoobAI...')

        noob_ai = hf_hub_download(
            repo_id="Laxhar/noobai-XL-1.1",
            filename="NoobAI-XL-v1.1.safetensors",
            local_dir="model")


    if variant >= 3:

        print('Downloading TTS model...')

        model_file = "jvnv-F1-jp/jvnv-F1-jp_e160_s14000.safetensors"
        config_file = "jvnv-F1-jp/config.json"
        style_file = "jvnv-F1-jp/style_vectors.npy"

        for file in [model_file, config_file, style_file]:
            print(f'downloading {file}...')
            hf_hub_download("litagin/style_bert_vits2_jvnv", file, local_dir="./")

            downloaded_path = os.path.join('TTS_model', file)
            final_path = os.path.join('TTS_model', file)

            if os.path.exists(downloaded_path) and downloaded_path != final_path:
                os.replace(downloaded_path, final_path)

        extra_dir = os.path.join('TTS_model', "jvnv-F1-jp")
        if os.path.exists(extra_dir) and not os.listdir(extra_dir):
            os.rmdir(extra_dir)

    with open('app_config.json', 'r', encoding='utf-8') as f:
        data = json.load(f)


    data['necessary_models'] = ness_models
    data['image_models'] = image_models
    data['TTS_models'] = TTS_models

    with open('app_config.json', 'w', encoding='utf-8') as f:
        json.dump(data, f, ensure_ascii=False, indent=4)

   
    print('All completed successfully. Please download LoRA by yourself from https://civitai.com/models/1234435/noobai-touhou-memories-of-phantasm-style')
    print('After downloading LoRA please put it in the "model" directory.')

def anki_sync():
    with open('app_config.json', 'r', encoding='utf-8') as f:
        data = json.load(f)

    anki_decks_data = get_deck_name_and_front_card_name()

    data['deck_name'] = []
    data['front_card_name'] = []

    for deck_name, front_card_name in anki_decks_data:
        data['deck_name'].append(deck_name)
        data['front_card_name'].append(front_card_name)

    with open('app_config.json', 'w', encoding='utf-8') as f:
        json.dump(data, f, ensure_ascii=False, indent=4)


if __name__ == "__main__":
    setup_project()
    print('Do you want to sync data from anki? (Y/N)')
    inp = input('>>> ').strip().lower()
    if inp == 'y':
        print('Press Enter if anki app is opened and have anki connect addon.')
        anki_sync()