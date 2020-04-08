#!/usr/bin/env python
# coding: utf-8
import collections
import gc
import json
import os
from glob import glob
from time import time
os.environ["CUDA_VISIBLE_DEVICES"] = "0"
from random import choice, seed, randint, random
import pandas as pd
import numpy as np
import tensorflow as tf
import keras.backend as K
import keras
from keras.models import Sequential, Model
from keras.layers import Input, CuDNNGRU as GRU, CuDNNLSTM as LSTM, Dropout, BatchNormalization
from keras.layers import Dense, Concatenate, Activation, Embedding, SpatialDropout1D, Bidirectional, Lambda, Conv1D
from keras.layers import Add, Average
from keras.optimizers import Nadam, Adam, Adamax
from keras.activations import absolute_import
from keras.legacy import interfaces
from keras.preprocessing.sequence import pad_sequences
from keras.callbacks import Callback
from keras.utils import to_categorical
from keras_bert.loader import load_trained_model_from_checkpoint
from keras_bert import AdamWarmup, calc_train_steps
from keras.engine import Layer
from keras.engine import InputSpec
from keras.objectives import categorical_crossentropy
from keras.objectives import sparse_categorical_crossentropy
from keras import activations, initializers, regularizers, constraints
from keras.models import load_model
from keras_bert import get_custom_objects
from tqdm import tqdm
from special_tokens import CHINESE_MAP
from metric_utils import compute_f1, compute_exact
from collections import OrderedDict, Counter
from bert4keras.layers import ZeroMasking


DEBUG = False
# BERT_PRETRAINED_DIR = "../../../chinese_bert/chinese_roberta_wwm_large_ext_L-24_H-1024_A-16/"
# TRN_FILENAME = "../data/train_20200228.csv"
TEST_FILENAME = "../data/test.csv"
save_filename = "../prediction_result/result_{}.csv"
# DEV_FILENAME = "../data/dev_20200228.csv"


MODEL_DIR = "../user_data/"
PREFIX = "1.25"
MAX_EPOCH = 15
MAX_LEN = 60
MAX_DOC_LEN = MAX_LEN // 2
THRE = 0.5
B_SIZE = 32
ACCUM_STEP = int(32 // B_SIZE)
FOLD_ID = [-1]
FOLD_NUM = 20
SEED = 2020
PREFIX += "_seed" + str(SEED)
SHUFFLE = True
DOC_STRIDE = 128
cfg = {}
cfg["verbose"] = PREFIX
cfg["span_mode"] = True
cfg["lr"] = 5e-6
cfg['min_lr'] = 6e-8 
cfg["ch_type"] = "tx_ft"
cfg["trainable"] = True
cfg["bert_trainable"] = True
cfg["accum_step"] = ACCUM_STEP
cfg["cls_num"] = 4
cfg["unit1"] = 128
cfg["unit2"] = 128
cfg["unit3"] = 512
cfg["conv_num"] = 128
cfg['maxlen'] = MAX_LEN
cfg["adv_training"] = False

# train_data = pd.read_csv(TRN_FILENAME)
# train_data.fillna("", inplace=True)
# dev_data = pd.read_csv(DEV_FILENAME)
# dev_data.fillna("", inplace=True)
# all_data = pd.concat([train_data, dev_data], axis=0, ignore_index=True)

def get_data(df_data):

    df_gb = df_data.groupby('query1')
    res = {}
    for index, data in df_gb:
        query2s = data["query2"]
        lables = data["label"]
        ele = {}
        pos_qs = []
        neg_qs = []
        for q, lable in zip(query2s, lables):
            if lable == 1:
                pos_qs.append(q)
            elif lable == 0:
                neg_qs.append(q)
            else:
                print("wrong data", index, q, lable)
        ele["pos"] = pos_qs
        ele["neg"] = neg_qs
        res[index] = ele
    return res


# In[3]:


def get_vocab(base_dir, albert=False, clue=False):
    if albert or "albert"in cfg["verbose"].lower():
        dict_path = os.path.join(base_dir, 'vocab_chinese.txt')
    elif clue:
        dict_path = os.path.join(base_dir, 'vocab_clue.txt')
    else:
        dict_path = os.path.join(base_dir, 'vocab.txt')
    print(dict_path)
    with open(dict_path, mode="r", encoding="utf8") as f:
        lines = f.readlines()
        lines = [l.strip() for l in lines]

    word_index = {v: k  for k, v in enumerate(lines)}
    for k, v in CHINESE_MAP.items():
        assert v in word_index
        if k in word_index:
            print("[!] CHINESE_MAP k = {} is in word_index, DON'T using `{}` to replace".format(k, v))
            continue
        del word_index[v]
    return word_index


def token2id_X(x, x_dict, x2=None, maxlen=None, maxlen1=None):
    if x2:
        x1 = x
        del x
        maxlen -= 3
        maxlen1 -= 2
        assert maxlen > maxlen1
        maxlen2 = maxlen - maxlen1 - 1
        x1 = ["[CLS]"] + list(x1)[: maxlen1] + ["[SEP]"] 
        x1 = [x_dict[e] if e in x_dict else x_dict["[UNK]"] for e in x1]
        seg1= [0 for _ in x1]
        
        x2 = list(x2)[: maxlen2] + ["[SEP]"] 
        x2= [x_dict[e] if e in x_dict else x_dict["[UNK]"] for e in x2]
        seg2 = [1 for _ in x2]
        x = x1 + x2
        seg = seg1 + seg2
        
    else:
        maxlen -= 2
        x = ["[CLS]"] + list(x)[: maxlen] + ["[SEP]"] 
        x = [x_dict[e] if e in x_dict else x_dict["[UNK]"] for e in x]
        seg = [0 for _ in x]        
    return x, seg


def seq_padding(X, maxlen=None, padding_value=None, debug=False):
    L = [len(x) for x in X]
    if maxlen is None:
        maxlen = max(L)

    pad_X = np.array([
        np.concatenate([x, [padding_value] * (maxlen - len(x))]) if len(x) < maxlen else x for x in X
    ])
    if debug:
        print("[!] before pading {}\n".format(X))
        print("[!] after pading {}\n".format(pad_X))
    return pad_X


def get_model(model_):
    model_inp_ind = [0, 1]
    inputs = [model_.inputs[e] for e in model_inp_ind]
    sub_model = Model(inputs=inputs, outputs=[model_.get_layer("po1").output])
    return sub_model


# In[ ]:


def test(sub_model, data, bs=32, x_dict=None):
    idxs = list(range(len(data)))
    T1, T2, O1, O2 = [], [], [], []
    preds = []
    for i in idxs:
        d = data.iloc[i]
        text = d["query1"]
        label_text = d["query2"]

        t1, t2 = token2id_X(text, x2=label_text, x_dict=word_index, maxlen=MAX_LEN, maxlen1=MAX_DOC_LEN)
        assert len(t1) == len(t2)

        T1.append(t1)
        T2.append(t2)

        if len(T1) == bs or i == idxs[-1]:
            T1 = seq_padding(T1, padding_value=cfg["x_pad"])
            T2 = seq_padding(T2, padding_value=0)
            assert T1.shape == T2.shape
            pred = sub_model.predict([T1, T2])
            preds.append(pred)
            T1, T2 = [], []
    
    preds = np.concatenate(preds, axis=0).reshape(-1)
    return preds


def ensemble_predictions(predictions, weights=None, type_="linear"):
    if not weights:
        # print("[!] AVE_WGT")
        weights = [1./ len(predictions) for _ in range(len(predictions))]
    assert len(predictions) == len(weights)
    if np.sum(weights) != 1.0:
        weights = [w / np.sum(weights) for w in weights]
    # print("[!] weights = {}".format(weights))
    assert np.isclose(np.sum(weights), 1.0)
    if type_ == "linear":
        res = np.average(predictions, weights=weights, axis=0)
    elif type_ == "harmonic":
        res = np.average([1 / p for p in predictions], weights=weights, axis=0)
        return 1 / res
    elif type_ == "geometric":
        numerator = np.average(
            [np.log(p) for p in predictions], weights=weights, axis=0
        )
        res = np.exp(numerator / sum(weights))
        return res
    elif type_ == "rank":
        from scipy.stats import rankdata
        res = np.average([rankdata(p) for p in predictions], weights=weights, axis=0)
        return res / (len(res) + 1)
    return res


test_data = pd.read_csv(TEST_FILENAME)

model_files_v1 = sorted(glob(os.path.join(MODEL_DIR, "*v12*.h5")))
len_1 = len(model_files_v1)
model_files_v3 = sorted(glob(os.path.join(MODEL_DIR, "*v13*.h5")))
len_3 = len(model_files_v3)


model_files = model_files_v1 + model_files_v3

print(len_1, len_3)

if DEBUG:
    from random import shuffle, seed
    seed(124)
    shuffle(model_files)
    model_files = model_files[: 2]

print(PREFIX, TEST_FILENAME, save_filename, model_files)
assert len(model_files) == len(set(model_files)) 
assert all([os.path.exists(f) for f in model_files]) 
preds = []
t0 = time()
for f in model_files:
    print("-" * 80)
    _t0 = time()
    print(f)
    if "albert" in f:
        word_index = get_vocab(base_dir="./", albert=True)
    elif "pair" in f or "clue" in f:
        word_index = get_vocab(base_dir="./", clue=True)
    else:
        word_index = get_vocab(base_dir="./")
    cfg["x_pad"] = word_index["[PAD]"]
    K.clear_session()
    print("[!] x_pad = {}".format(cfg["x_pad"]))
    if "albert" in f.lower() or "nezha" in f.lower():
        model = load_model(f)
    else:
        model = load_model(f, custom_objects=get_custom_objects())
    sub_model = get_model(model)
    pred = test(sub_model, test_data, x_dict=word_index)
#     auc = roc_auc_score(O1, pred)
#     acc = accuracy_score(O1, np.array(pred > 0.5, "int32"))    
#     print("[{}]".format(time() - t0), auc, acc)
    print("[{}] f = `{}`, finish".format(time() - _t0, f))
    print(pred.shape)
    preds.append(pred)
    del model, word_index, pred
    gc.collect()

print("[{}]".format(time() - t0))
print(len_1, len_3)
pred1 = ensemble_predictions(preds[0: len_1])
pred3 = ensemble_predictions(preds[len_1: len_1 + len_3])

pred = ensemble_predictions([pred1, pred3])
print(pred1[: 3], pred3[: 3], pred[: 3])

for q in [0.382, 0.387, 0.392, 0.397, 0.402, 0.407, 0.412]:
    thre = np.quantile(pred, q=1-q)
    print("[!]", q, thre, (pred > thre).astype("int32").sum())

POS_QUAN = 0.3945
thre = np.quantile(pred, q=1 - POS_QUAN)
test_data["prob"] = pred
test_data["label"] = (pred > thre).astype("int32") 
print(test_data.describe())
print("-" * 81)
print(pred.shape, thre, POS_QUAN, PREFIX, TEST_FILENAME, save_filename, model_files)
print(test_data["label"].value_counts())

import datetime
save_filename = save_filename.format(datetime.datetime.now().strftime('%Y%m%d_%H%M%S'))
test_data[["id", "label"]].to_csv(save_filename, index=False)
print("verbose = {} FINISH".format(PREFIX))

