from style_bert_vits2.nlp import bert_models
from style_bert_vits2.constants import Languages
from transformers import logging as transformers_logging
from pathlib import Path
from style_bert_vits2.tts_model import TTSModel
import torch
import soundfile as sf
import gc

transformers_logging.set_verbosity_error()

def create_audio(sentences: list[str], words: list[str]):
    """Generates TTS audio files for the provided sentences and individual words."""
    bert_models.load_model(Languages.JP, "ku-nlp/deberta-v2-large-japanese-char-wwm")
    bert_models.load_tokenizer(Languages.JP, "ku-nlp/deberta-v2-large-japanese-char-wwm")

    device = 'cuda' if torch.cuda.is_available() else 'cpu'

    model_file = "jvnv-F1-jp_e160_s14000.safetensors"
    config_file = "config.json"
    style_file = "style_vectors.npy"

    assets_root = Path("jvnv-F1-jp")

    model = TTSModel(
        model_path=assets_root / model_file,
        config_path=assets_root / config_file,
        style_vec_path=assets_root / style_file,
        device=device,

    )

    with torch.amp.autocast(device_type=device, dtype=torch.float32):
        for i in words:
            sr, audio = model.infer(text=i,
                                    length=1.1,
                                    noise=0.6,
                                    noise_w=0.8,
                                    split_interval=0.5,
                                    style='Happy',
                                    style_weight=1.5
                                    )
            sf.write(f"audio/{i}_word.mp3", audio, sr)

        k = 0
        for i in sentences:
            sr, audio = model.infer(text=i,
                                    length=1.03,
                                    noise=0.6,
                                    noise_w=0.8,
                                    split_interval=0.5,
                                    style='Happy',
                                    style_weight=1.5
                                    )
            sf.write(f"audio/{words[k]}_sent.mp3", audio, sr)
            k += 1

    del model
    gc.collect()
    torch.cuda.empty_cache()