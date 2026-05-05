import torch
from vector_sim import get_labse_model, unload_labse_model
import numpy as np
import re
import gc
import xgboost

def give_n_best(valid_words: dict[str, list[str]], n):
    """
    This function ranks candidate words based on a trained XGBoost familiarity model
    and returns the top n selections along with their originating sentences
    """

    model = xgboost.XGBRegressor()
    model.load_model('model/xgb_model.json')
    max_familiarity = 2.356011

    vec_valid_words = []
    re_kana = re.compile(r'[\u3040-\u309F]')
    re_kanji = re.compile(r'[\u4E00-\u9FFF]')
    labse_model = get_labse_model()
    for k, v in valid_words.items():
        vecs = labse_model.encode(v, convert_to_numpy=True)

        word_infos = []
        
        for word in v:

            word_infos.append([
                len(word),
                len(re_kana.findall(word)),
                len(re_kanji.findall(word))
            ])

        input_data = np.column_stack([vecs, np.array(word_infos)])

        predict = model.predict(input_data)
        und_predict = max_familiarity + 1 - np.exp(predict)

        for i, w_score in enumerate(und_predict):
            vec_valid_words.append((w_score, v[i], k))


    vec_valid_words.sort(reverse=True)
    result = [v[1:] for v in vec_valid_words[:n]]

    unload_labse_model()

    del model
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        torch.cuda.ipc_collect()

    return result
