#!/usr/bin/env python
# coding: utf-8

# In[1]:


import collections
import gc
import json
import os
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
# from keras.preprocessing.sequence import pad_sequaencesget_
from keras.callbacks import Callback
from keras.utils import to_categorical
from sklearn.model_selection import KFold as KF
from sklearn.model_selection import StratifiedKFold as SKF
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
from sklearn.metrics import roc_auc_score, accuracy_score
from special_tokens import CHINESE_MAP
from metric_utils import compute_f1, compute_exact
from collections import OrderedDict, Counter
from sklearn.metrics import classification_report
from time import time


# In[2]:


BERT_PRETRAINED_DIR = "../data/External/UER-large/"
TRN_FILENAME = "../data/train_20200228.csv"
DEV_FILENAME = "../data/dev_20200228.csv"
SAVE_DIR = "../user_data/"
PREFIX = "USE_v12_augm"
if "large-clue" in BERT_PRETRAINED_DIR or "large-pair" in BERT_PRETRAINED_DIR:
    W2V_FILE = "./word_embedding_matrix_v2"
else:
    W2V_FILE = "./word_embedding_matrix"
MAX_EPOCH = 15
RUN_EPOCH = 10
MAX_LEN = 60
MAX_DOC_LEN = MAX_LEN // 2
THRE = 0.5
B_SIZE = 32
ACCUM_STEP = int(32 // B_SIZE)
FOLD_ID = list(range(10, 15))
FOLD_NUM = 25
SEED = 2020

SHUFFLE = True
DOC_STRIDE = 128
cfg = {}

cfg["base_dir"] = BERT_PRETRAINED_DIR
cfg["span_mode"] = True
cfg["lr"] = 9e-6
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
cfg["adv_training"] = True
cfg["W2V_FILE"] = W2V_FILE
cfg["use_embed"] = True
cfg["use_embed_v2"] = True
PREFIX += "_seed" + str(SEED)
cfg["verbose"] = PREFIX
PREFIX = PREFIX + "_embed_v2" if cfg["use_embed_v2"] else PREFIX

train_data = pd.read_csv(TRN_FILENAME)
train_data.fillna("", inplace=True)
dev_data = pd.read_csv(DEV_FILENAME)
dev_data.fillna("", inplace=True)
all_data = pd.concat([train_data, dev_data], axis=0, ignore_index=True)

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

# train_data_dict = get_data(train_data)


# In[3]:


def get_vocab(base_dir=BERT_PRETRAINED_DIR, albert=False):
    if albert or "albert"in cfg["verbose"].lower():
        dict_path = os.path.join(base_dir, 'vocab_chinese.txt')
    else:
        dict_path = os.path.join(base_dir, 'vocab.txt')
    with open(dict_path, mode="r", encoding="utf8") as f:
        lines = f.readlines()
        lines = [l.strip() for l in lines]

    word_index = {v: k  for k, v in enumerate(lines)}
    for k, v in CHINESE_MAP.items():
        assert v in word_index
        if k in word_index:
            print("[!] CHINESE_MAP k = {} is in word_index, DON'T using `{}` to replace".format(k, v))
            continue
        # word_index[k] = word_index[v]
        del word_index[v]
    return word_index


def get_label():
    labels = ["0", "1"]
    label2id = {k: v for v, k in enumerate(labels)}
    id2label = {v: k for k, v in label2id.items()}
    return label2id, id2label, labels
    
    
def get_coefs(word, *arr):
    return word, np.asarray(arr, dtype=np.float16)


def load_embed(path, dim=300, word_index=None):
    embedding_index = {}
    with open(path, mode="r", encoding="utf8") as f:
        lines = f.readlines()
        for l in lines:
            l = l.strip().split()
            word, arr = l[0], l[1:]
            if len(arr) != dim:
                print("[!] l = {}".format(l))
                continue
            if word_index and word not in word_index:
                continue
            word, arr = get_coefs(word, arr)
            embedding_index[word] = arr
    return embedding_index


def build_matrix(path, word_index=None, max_features=None, dim=300):
    embedding_index = load_embed(path, dim=dim, word_index=word_index)
    max_features = len(word_index) + 1 if max_features is None else max_features 
    embedding_matrix = np.zeros((max_features + 1, dim))
    unknown_words = []
    
    for word, i in word_index.items():
        if i <= max_features:
            try:
                embedding_matrix[i] = embedding_index[word]
            except KeyError:
                unknown_words.append(word)
    return embedding_matrix, unknown_words


def load_word_embed(word_embed_f1="../../../chinese_embedding/Tencent_AILab_ChineseEmbedding.txt", 
               word_embed_f2="../../../chinese_embedding/cc.zh.300.vec", 
               save_filename=W2V_FILE,
               word_index=None):
    if os.path.exists(save_filename + ".npy"):
        word_embedding_matrix = np.load(save_filename + ".npy").astype("float32")
    else:
        if "tx" in cfg["ch_type"]:
            tx_embed, tx_unk = build_matrix(word_embed_f1, word_index=word_index, dim=200)
        else:
            tx_embed = np.zeros(shape=(len(word_index) + 2, 0))
            tx_unk = []
        if "ft" in cfg["ch_type"]:
            ft_embed, ft_unk = build_matrix(word_embed_f2, word_index=word_index, dim=300)
        else:
            ft_embed = np.zeros(shape=(len(word_index) + 2, 0))
            ft_unk = []    

        word_embedding_matrix = np.concatenate([tx_embed, ft_embed], axis=-1).astype("float32")
        print(word_embedding_matrix.shape, len(tx_unk), len(ft_unk))
        np.save(save_filename, word_embedding_matrix )
    return word_embedding_matrix
    
    
word_index = get_vocab()
label2id, id2label, labels = get_label()
word_embedding_matrix = load_word_embed(word_index=word_index)

NUM_CLASS = len(label2id)
cfg["x_pad"] = word_index["[PAD]"]
cfg["num_class"] = NUM_CLASS
cfg["filename"] = "{}_{}_{}_{}".format(PREFIX, cfg["ch_type"], FOLD_NUM, cfg["lr"])
cfg["filename"] = cfg["filename"] + "_adv_training" if cfg["adv_training"] else cfg["filename"]
cfg["filename"] = cfg["filename"] + "_embed" if cfg["use_embed"] else cfg["filename"]
cfg["filename"] = cfg["filename"] + "_v2" if cfg["use_embed_v2"]and cfg["use_embed"] else cfg["filename"]
print(label2id, id2label, labels, len(word_index), cfg["filename"])


# In[4]:


def build_model(cfg, summary=False, word_embedding_matrix=None):
    def _get_model(base_dir, cfg_=None):
        if "albert"in cfg["verbose"].lower():
            from bert4keras.bert import build_bert_model
            config_file = os.path.join(base_dir, 'albert_config.json')
            checkpoint_file = os.path.join(base_dir, 'model.ckpt-best')
            model = build_bert_model(
                    config_path=config_file,
                    checkpoint_path=checkpoint_file,
                    model='albert',
                    return_keras_model=True
            )
            if cfg_["cls_num"] > 1:
                output = Concatenate(axis=-1)([model.get_layer("Encoder-1-FeedForward-Norm").get_output_at(-i) for i in range(1, cfg["cls_num"] + 1)])
                model = Model(model.inputs[: 2], outputs=output)
            model.trainable = cfg_["bert_trainable"]
        else:
            config_file = os.path.join(base_dir, 'bert_config.json')
            checkpoint_file = os.path.join(base_dir, 'bert_model.ckpt')
            if not os.path.exists(config_file):
                config_file = os.path.join(base_dir, 'bert_config_large.json')
                checkpoint_file = os.path.join(base_dir, 'roberta_l24_large_model')            
            model = load_trained_model_from_checkpoint(config_file, 
                                                       checkpoint_file, 
                                                       training=False, 
                                                       trainable=cfg_["bert_trainable"], 
                                                       output_layer_num=cfg_["cls_num"],
                                                       seq_len=cfg_['maxlen'])
            
            # model = Model(inputs=model.inputs[: 2], outputs=model.layers[-7].output)

        return model
    
    def _get_opt(num_example, warmup_proportion=0.1, lr=2e-5, min_lr=None):
        total_steps, warmup_steps = calc_train_steps(
            num_example=num_example,
            batch_size=B_SIZE,
            epochs=MAX_EPOCH,
            warmup_proportion=warmup_proportion,
        )
        opt = AdamWarmup(total_steps, warmup_steps, lr=lr, min_lr=min_lr)
        if cfg.get("accum_step", None) and cfg["accum_step"] > 1:
            print("[!] using accum_step = {}".format(cfg["accum_step"]))
            from accum_optimizer import AccumOptimizer
            opt = AccumOptimizer(opt, steps_per_update=cfg["accum_step"])
        
        return opt

    bert_model = _get_model(cfg["base_dir"], cfg)

    if word_embedding_matrix is not None:
        embed = Embedding(input_dim=word_embedding_matrix.shape[0], 
                          output_dim=word_embedding_matrix.shape[1],
                          weights=[word_embedding_matrix],
                          trainable=cfg["trainable"],
                          name="char_embed"
                         )
    
    t1_in = Input(shape=(None, ))
    t2_in = Input(shape=(None, ))
    o1_in = Input(shape=(1, ))
    o2_in = Input(shape=(1, ))

    t1, t2, o1, o2 = t1_in, t2_in, o1_in, o2_in
    
    t = bert_model([t1, t2])
    mask = Lambda(lambda x: K.cast(K.not_equal(x, cfg["x_pad"]), 'float32'))(t1)
    ## Char information
    if word_embedding_matrix is not None:
        word_embed = embed(t1)
        if cfg.get("use_embed_v2", False):
            _t2 = Lambda(lambda x: K.expand_dims(x, axis=-1))(t2)
            word_embed = Concatenate(axis=-1)([word_embed, _t2])
        word_embed = Lambda(lambda x: x[0] * K.expand_dims(x[1], axis=-1))([word_embed, mask])
        word_embed = Bidirectional(LSTM(cfg["unit1"], return_sequences=True), merge_mode="sum")(word_embed)
        word_embed = Lambda(lambda x: x[0] * K.expand_dims(x[1], axis=-1))([word_embed, mask])
        t = Concatenate(axis=-1)([t, word_embed])
    
    t = Lambda(lambda x: x[0] * K.expand_dims(x[1], axis=-1))([t, mask])     
    t = Bidirectional(LSTM(cfg["unit3"], return_sequences=True), merge_mode="concat")(t)
    # t = Lambda(lambda x: x[0] * K.expand_dims(x[1], axis=-1))([t, mask]) 
    # t = Conv1D(cfg["conv_num"], kernel_size=3, padding="same")(t) 
    t = Lambda(lambda x: x[:, 0, :], name="extract_layer")(t)
    if cfg.get("num_class", 1) == 2:
        po1_logit = Dense(1, name="po1_logit")(t)
        po1 = Activation('sigmoid', name="po1")(po1_logit)
        train_model = Model(inputs=[t1_in, t2_in, o1_in],
                            outputs=[po1])        
        o1_loss = K.binary_crossentropy(o1, po1)
        loss = K.mean(o1_loss)
    else:
        po1_logit = Dense(cfg["num_class"], name="po1_logit")(t)
        po1 = Activation('softmax', name="po1")(po1_logit)
        train_model = Model(inputs=[t1_in, t2_in, o1_in],
                            outputs=[po1])
        loss = K.categorical_crossentropy(o1, po1, axis=-1)
        loss = K.mean(loss)

    train_model.add_loss(loss)
    opt = _get_opt(num_example=cfg["num_example"], lr=cfg["lr"], min_lr=cfg['min_lr'])
    train_model.compile(optimizer=opt)
    if summary:
        train_model.summary()
    return train_model


# print("----------------build model ---------------")
# model = build_model(cfg, summary=True, word_embedding_matrix=word_embedding_matrix if cfg["use_embed"] else None)
# del model


# In[5]:


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


class data_generator:
    
    def __init__(self, data, batch_size=B_SIZE, shuffle=SHUFFLE, augm_frac=0.75):
        self.data = data
        self.batch_size = batch_size
        self.steps = cfg["num_example"] // self.batch_size
        self.shuffle = shuffle
        self.data_dict = get_data(data)
        self.augm_frac = augm_frac
        if cfg["num_example"] % self.batch_size != 0:
            self.steps += 1

    def __len__(self):
        return self.steps
    
    def __iter__(self):
        
        while True:
            idxs = list(range(len(self.data)))
            if self.shuffle:
                np.random.shuffle(idxs)
            T1, T2, O1, O2 = [], [], [], []
            for i in idxs:
                d = self.data.iloc[i]
                text = d["query1"]
                label_text = d["query2"]
                o1 = d["label"]
                
                if random() > self.augm_frac:
                    data_d = self.data_dict[text]
                    pos_data = data_d["pos"]
                    neg_data = data_d["neg"]
                    if pos_data and neg_data:
                        if random() > 0.5:
                            o1 = 1
                            label_text = choice(pos_data)
                            if len(pos_data) >= 2:
                                _pos_data = [e for e in pos_data if e != label_text]
                                text = choice(_pos_data)
                        else:
                            o1 = 0
                            text = choice(pos_data)
                            label_text = choice(neg_data)   
                
                if random() > 0.5:
                    text, label_text = label_text, text
                
                if o1 == "":
                    continue
                o1 = float(o1)
                assert 0 <= o1 <= 1
                
                O1.append(o1)                
                t1, t2 = token2id_X(text, x2=label_text, x_dict=word_index, maxlen=MAX_LEN, maxlen1=MAX_DOC_LEN)
                assert len(t1) == len(t2)
                
                T1.append(t1)
                T2.append(t2)

                if len(T1) == self.batch_size or i == idxs[-1]:
                    O1 = np.array(O1).reshape(-1, 1)
                    T1 = seq_padding(T1, padding_value=cfg["x_pad"])
                    T2 = seq_padding(T2, padding_value=0)
                    assert T1.shape == T2.shape and T1.shape[0] == O1.shape[0]

                    yield [T1, T2, O1], None
                    T1, T2, O1, = [], [], []
                    


# In[6]:


def get_model(model_):
    model_inp_ind = [0, 1]
    inputs = [model_.inputs[e] for e in model_inp_ind]
    sub_model = Model(inputs=inputs, outputs=[model_.get_layer("po1").output])
    return sub_model


def find_best_acc_score(y_pred, y_true, use_plt=True, bins=1000):
    thres = [i / bins for i in range(1, bins)]
    scores = [accuracy_score(y_true, np.array(y_pred > thre, "int32")) for thre in thres]
#     if use_plt:
#         import matplotlib
#         import matplotlib.pyplot as plt
#         %matplotlib inline
#         plt.plot(scores)
#         plt.show()
    ind = np.argmax(scores)
    max_score = np.max(scores)
    assert abs(scores[ind] - max_score) < 1e-15
    return max_score, thres[ind]
            

def evaluate(sub_model, data, bs=32):
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

        o1 = float(d["label"])
        O1.append(o1)
        if len(T1) == bs or i == idxs[-1]:
            T1 = seq_padding(T1, padding_value=cfg["x_pad"])
            T2 = seq_padding(T2, padding_value=0)
            assert T1.shape == T2.shape
            pred = sub_model.predict([T1, T2])
            preds.append(pred)
            T1, T2 = [], []
    
    preds = np.concatenate(preds, axis=0).reshape(-1)
    O1 = np.array(O1).reshape(-1)
    O1 = O1.astype("int32")
    auc = roc_auc_score(O1, preds)
    best_res = find_best_acc_score(preds, O1)
    print("[!] best accurary&threshold = {}".format(best_res))
    print("[!] best threshold classification_report")
    print(classification_report(O1,  np.array(preds > best_res[1], "int32"), digits=6))    
    print("-" * 80)
    print("[!] np.mean(preds) = {}".format(np.mean(preds)))
    print("[!] classification_report")
    print(classification_report(O1,  np.array(preds > 0.5, "int32"), digits=6))
    acc = accuracy_score(O1, np.array(preds > 0.5, "int32"))
    return auc, acc
    

class Evaluate(Callback):
    def __init__(self, data, filename=None):
        self.F1 = []
        self.best = 0.
        self.filename = filename
        self.data = data
    
    def on_epoch_begin(self, epoch, logs=None):
        if epoch ==  0:
            print("[!] test load&save model")
            f = self.filename + ".h5"
            f = os.path.join(SAVE_DIR, f)
            self.model.save(f, include_optimizer=False, overwrite=False)
            if "albert" in cfg["verbose"]:
                model_ = load_model(f) 
            else:
                model_ = load_model(f, custom_objects=get_custom_objects()) 

    def on_epoch_end(self, epoch, logs=None):
        if epoch + 1 < 1:
            return
#         if epoch + 1 in [3, 6, 9, 10, 12, 15, 18, 20]:
#             f = self.filename + "_{}.h5".format(epoch + 1)
#             f = os.path.join(SAVE_DIR, f)
#             self.model.save(f, include_optimizer=False)
            
        sub_model = get_model(self.model)
        f1, class_f1 = evaluate(sub_model, data=self.data)
        self.F1.append(f1)
        if f1 > self.best:
            f = self.filename + ".h5"
            f = os.path.join(SAVE_DIR, f)
            self.model.save(f, include_optimizer=False)
            
        if f1 > self.best:
            self.best = f1
            print("[!] epoch = {}, new best_auc = {}".format(epoch + 1,  f1))
        print('[!] epoch = {}, auc = {}, best auc {}'.format(epoch + 1, f1, self.best))
        print('[!] epoch = {}, acc = {}\n'.format(epoch + 1, class_f1))


# In[7]:


def search_layer(inputs, name, exclude_from=None):
    """根据inputs和name来搜索层
    说明：inputs为某个层或某个层的输出；name为目标层的名字。
    实现：根据inputs一直往上递归搜索，直到发现名字为name的层为止；
         如果找不到，那就返回None。
    """
    if exclude_from is None:
        exclude_from = set()

    if isinstance(inputs, keras.layers.Layer):
        layer = inputs
    else:
        layer = inputs._keras_history[0]

    if layer.name == name:
        return layer
    elif layer in exclude_from:
        return None
    else:
        exclude_from.add(layer)
        if isinstance(layer, keras.models.Model):
            model = layer
            for layer in model.layers:
                if layer.name == name:
                    return layer
        inbound_layers = layer._inbound_nodes[0].inbound_layers
        if not isinstance(inbound_layers, list):
            inbound_layers = [inbound_layers]
        if len(inbound_layers) > 0:
            for layer in inbound_layers:
                layer = search_layer(layer, name, exclude_from)
                if layer is not None:
                    return layer
                
def adversarial_training(model, embedding_names, epsilon=1):
    """给模型添加对抗训练
    其中model是需要添加对抗训练的keras模型，embedding_names
    则是model里边Embedding层的名字。要在模型compile之后使用。
    """
    if model.train_function is None:  # 如果还没有训练函数
        model._make_train_function()  # 手动make
    old_train_function = model.train_function  # 备份旧的训练函数

    # 查找Embedding层
    embedding_layers = []
    for embedding_name in embedding_names:
        for output in model.outputs:
            embedding_layer = search_layer(output, embedding_name)
            if embedding_layer is not None:
                embedding_layers.append(embedding_layer)
                break
    for embedding_layer in embedding_layers:
        if embedding_layer is None:
            raise Exception('Embedding layer not found')

    # 求Embedding梯度
    embeddings = [embedding_layer.embeddings for embedding_layer in embedding_layers] # Embedding矩阵
    gradients = K.gradients(model.total_loss, embeddings)  # Embedding梯度
    # gradients = K.zeros_like(embeddings) + gradients[0]  # 转为dense tensor
    gradients = [K.zeros_like(embedding) + gradient for embedding, gradient in zip(embeddings, gradients)]

    # 封装为函数
    inputs = (model._feed_inputs +
              model._feed_targets +
              model._feed_sample_weights)  # 所有输入层
    embedding_gradients = K.function(
        inputs=inputs,
        outputs=gradients,
        name='embedding_gradients',
    )  # 封装为函数

    def train_function(inputs):  # 重新定义训练函数
#         grads = embedding_gradients(inputs)[0]  # Embedding梯度
#         delta = epsilon * grads / (np.sqrt((grads**2).sum()) + 1e-8)  # 计算扰动
        grads = embedding_gradients(inputs)  # Embedding梯度
        deltas = [epsilon * grad / (np.sqrt((grad**2).sum()) + 1e-8) for grad in grads]  # 计算扰动
        # 注入扰动
        # K.set_value(embeddings, K.eval(embeddings) + delta)  
        for embedding, delta in zip(embeddings, deltas):
            K.set_value(embedding, K.eval(embedding) + delta)
            
        outputs = old_train_function(inputs)  # 梯度下降
        # 删除扰动
        # K.set_value(embeddings, K.eval(embeddings) - delta)  # 删除扰动
        for embedding, delta in zip(embeddings, deltas):
            K.set_value(embedding, K.eval(embedding) - delta)       
        return outputs

    model.train_function = train_function  # 覆盖原训练函数


adv_layer_names = ['Embedding-Token', 'char_embed']

if -1 in FOLD_ID:
    fold_id = -1
    cfg["num_example"] = len(train_data)
    print("-" * 81)
    print("[!] start fold_id =", fold_id, train_data.shape, dev_data.shape)
    print(cfg)
    K.clear_session()
    gc.collect()
    train_D = data_generator(train_data)
    seed(SEED + fold_id)
    np.random.seed(SEED + fold_id)
    tf.random.set_random_seed(SEED + fold_id)
    model = build_model(cfg, summary=True, word_embedding_matrix=word_embedding_matrix if cfg["use_embed"] else None)
    if cfg["adv_training"]:
        print("[!] using adv_training")
        adversarial_training(model, adv_layer_names, 0.5)
    evaluator = Evaluate(filename=cfg["filename"] + "_fold{}".format(fold_id), data=dev_data)
    model.fit_generator(train_D.__iter__(),
                              steps_per_epoch=len(train_D),
                              epochs=RUN_EPOCH,
                              callbacks=[evaluator],
                              shuffle=True
                              )
    del model, train_data, dev_data
    gc.collect()
    print("[!] finish fold_id =", fold_id)
    print("-" * 81)
    

skf = SKF(FOLD_NUM, shuffle=False, random_state=SEED)

print(all_data.shape)
_t0 = time()
for fold_id, (trn_ind, val_ind) in enumerate(skf.split(range(len(all_data)), all_data["label"])):
    if fold_id not in FOLD_ID:
        continue
    t0 = time()
    dev_data = all_data.iloc[val_ind].reset_index(drop=True)
    train_data = all_data.iloc[trn_ind].reset_index(drop=True)
    cfg["num_example"] = len(train_data)
    print("-" * 81)
    print("[!] start fold_id =", fold_id, train_data.shape, dev_data.shape)
    print(cfg)
    K.clear_session()
    gc.collect()
    train_D = data_generator(train_data)
    seed(SEED + fold_id)
    np.random.seed(SEED + fold_id)
    tf.random.set_random_seed(SEED + fold_id)
    model = build_model(cfg, summary=True, word_embedding_matrix=word_embedding_matrix if cfg["use_embed"] else None)
    if cfg["adv_training"]:
        print("[!] using adv_training")
        adversarial_training(model, adv_layer_names, 0.5)
    evaluator = Evaluate(filename=cfg["filename"] + "_fold{}".format(fold_id), data=dev_data)
    model.fit_generator(train_D.__iter__(),
                              steps_per_epoch=len(train_D),
                              epochs=RUN_EPOCH,
                              callbacks=[evaluator],
                              shuffle=True
                              )
    print(evaluator.F1, max(evaluator.F1))    
    print("[{}] finish fold_id =".format(time() - t0), fold_id)
    print("-" * 81)
    del model, train_data, dev_data, evaluator
    gc.collect()    
print("[{}] finish =".format(time() - _t0))


# In[9]:


sub_model = get_model(model)
evaluate(sub_model=sub_model, data=dev_data)




