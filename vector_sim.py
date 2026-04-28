from transformers import logging as transformers_logging
from sentence_transformers import SentenceTransformer, util
import torch
transformers_logging.set_verbosity_error()
import gc

_MODEL_INSTANCE = None

def get_labse_model():
    global _MODEL_INSTANCE
    if _MODEL_INSTANCE is None:
        device = 'cuda' if torch.cuda.is_available() else 'cpu'
        _MODEL_INSTANCE = SentenceTransformer('sentence-transformers/LaBSE').to(device)
    return _MODEL_INSTANCE

def get_best_meaning(target_word, context_sentence, jsons, weight_word=0.6, weight_context=0.4):
    """
    Находит лучший перевод, учитывая вес самого слова и вес предложения.
    """
    model = get_labse_model()
    translations = get_meanings(jsons)
    ids = range(len(translations))

    word_emb = model.encode(target_word, convert_to_tensor=True)
    context_emb = model.encode(context_sentence, convert_to_tensor=True)
    trans_embs = model.encode(translations, convert_to_tensor=True)

    score_word = util.cos_sim(word_emb, trans_embs)[0]
    score_context = util.cos_sim(context_emb, trans_embs)[0]

    final_scores = (score_word * weight_word) + (score_context * weight_context)

    # 5. Сортировка и вывод
    results = zip(translations, final_scores.tolist(), ids)
    sorted_results = sorted(results, key=lambda x: x[1], reverse=True)

    unload_labse_model()

    return jsons[sorted_results[0][-1]]


def get_meanings(jsons):
    meanings = []
    for json in jsons:
        meaning = json['meanings']
        meanings.append(meaning.split('.')[0] if meaning[0] != '1' else meaning[2:].split('.')[0])
    return meanings


def unload_labse_model():
    global _MODEL_INSTANCE
    if _MODEL_INSTANCE is not None:
        _MODEL_INSTANCE.to('cpu')
        del _MODEL_INSTANCE
        _MODEL_INSTANCE = None

    gc.collect()
    
    if torch.cuda.is_available():
        torch.cuda.empty_cache()