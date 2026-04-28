import torch
from vector_sim import get_labse_model, unload_labse_model
import numpy as np
import re
import gc
import xgboost

def give_n_best(valid_words: dict[str, list[str]], n):
    model = xgboost.XGBRegressor()
    model.load_model('model/xgb_model.json')
    max_familiarity = 2.356011

    vec_valid_words = []
    for k, v in valid_words.items():
        for word in v:
            labse_model = get_labse_model()

            vec = labse_model.encode(word, convert_to_numpy=True)
            word_info = np.array([len(word),
                         sum([len(sym) for sym in re.findall(r'[\u3040-\u309F]+', word)]),
                         sum([len(sym) for sym in re.findall(r'[\u4E00-\u9FFF]+', word)])])
            input_data = np.hstack([vec, word_info]).reshape(1, -1)

            predict = model.predict(input_data)
            und_predict = max_familiarity + 1 - np.exp(predict)

            vec_valid_words.append((und_predict, word, k))


    vec_valid_words.sort(reverse=True)
    result = [v[1:] for v in vec_valid_words[:n]]

    unload_labse_model()

    del model
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        torch.cuda.ipc_collect()

    return result