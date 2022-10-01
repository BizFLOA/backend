import os
import sys
# sys.path.append("C:\\Users\\take\\PycharmProjects\\ICT_competition_mysql\\data_prep\\Classifier_지원분야")
sys.path.append("./Data_Engineering/utilities")
sys.path.append("./Data_Engineering/Classifier_지원분야/sources")
import pickle

from tqdm import tqdm
import random as rnd

import pandas as pd
from pandas import DataFrame as dataframe, Series as series
import numpy as np
from numpy import random as np_rnd
import re

from sklearn.preprocessing import LabelEncoder, OneHotEncoder
from sklearn.model_selection import StratifiedKFold
from sklearn import metrics
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.decomposition import TruncatedSVD

# import torch
# from torch.utils.data import Dataset, DataLoader
# from transformers import DistilBertModel
# from tokenization_kobert import KoBertTokenizer

import tensorflow as tf
from tensorflow import random as tf_rnd
from tensorflow import keras
from tensorflow.keras import layers
from tensorflow.keras import Model, Sequential
from tensorflow.keras import metrics as tf_metrics
from tensorflow.keras import callbacks as tf_callbacks
from tqdm.keras import TqdmCallback

import tensorflow_addons as tfa
import tensorflow_recommenders as tfrs

from pykospacing import Spacing
from ckonlpy.tag import Twitter as Okt

from text_prep_funcs import *
# from main_predict_class_v6_tfidfV2_dense import prep3_correct_spacing, prep4_remove_special_char, prep5_remove_content_rightside, \
#     prep6_get_text_KorEng, prep7_spacing_between_KorEng, remove_stopwords, extract_nouns


#     df = prep3_correct_spacing(df)
#     df = prep4_remove_special_char(df)
#     df = prep5_remove_content_rightside(df)
#     df = prep6_get_text_KorEng(df)
#     df = prep7_spacing_between_KorEng(df)
#     df = remove_stopwords(df)
#     df = extract_nouns(df)

# pd.set_option('display.max_columns', 100)
# pd.set_option('display.max_rows', 100)
# pd.set_option('display.width', 1000)


def seed_everything(seed=42):
    # python random module
    rnd.seed(seed)
    # numpy random
    np_rnd.seed(seed)
    # tf random
    try:
        tf_rnd.set_seed(seed)
    except:
        pass
    # RAPIDS random
    try:
        cp.random.seed(seed)
    except:
        pass
    # pytorch random
    try:
        torch.manual_seed(seed)
    except:
        pass

# def prep3_correct_spacing(df):
#     spacing = Spacing(rules=['R&D'])
#     df["title"] = df["title"].apply(lambda x: spacing(x))
#     return df
# def prep4_remove_special_char(df):
#     prep_str = []
#     for i in df["title"]:
#         tmp = i
#         while True:
#             index = re.search('[A-Za-z]&[A-Za-z]', tmp)
#             if index is None:
#                 break
#             else:
#                 target_pos = index.span()[0]
#                 tmp = "".join([tmp[:(target_pos + 1)], tmp[(target_pos + 2):]])
#         prep_str.append(tmp)
#     df["title"] = prep_str
#     return df
# def prep5_remove_content_rightside(df):
#     prep_str = []
#     for i in df["title"]:
#         tmp = i
#         # tmp = df_full_x["title"].iloc[18755]
#         if tmp.rstrip()[-1] == ")":
#             tmp = tmp.split("(")
#             prep_str.append(tmp[0]) if len(tmp) == 1 else prep_str.append(" ".join(tmp[:-1]))
#         else:
#             prep_str.append(tmp)
#     df["title"] = prep_str
#     return df
# def prep6_get_text_KorEng(df):
#     prep_str = []
#     for i in df["title"]:
#         if i.lstrip()[0] == "[":
#             try:
#                 prep_str.append(" ".join(re.sub('[^ A-Za-z가-힣]', ' ', " ".join(i.split("]")[1:])).split()).lower())
#             except:
#                 prep_str.append(" ".join(re.sub('[^ A-Za-z가-힣]', ' ', i).split()).lower())
#         else:
#             prep_str.append(" ".join(re.sub('[^ A-Za-z가-힣]', ' ', i).split()).lower())
#     df["title"] = prep_str
#     return df
# def prep7_spacing_between_KorEng(df):
#     prep_str = []
#     for i in df["title"]:
#         tmp = i
#         while True:
#             index = re.search('[a-zA-Z][가-힣]', tmp)
#             if index is None:
#                 break
#             else:
#                 target_pos = index.span()[0]
#                 tmp = " ".join([tmp[:(target_pos + 1)], tmp[(target_pos + 1):]])
#         while True:
#             index = re.search('[가-힣][a-zA-Z]', tmp)
#             if index is None:
#                 break
#             else:
#                 target_pos = index.span()[0]
#                 tmp = " ".join([tmp[:(target_pos + 1)], tmp[(target_pos + 1):]])
#         prep_str.append(tmp)
#     df["title"] = prep_str
#     return df
# def remove_stopwords(df):
#     # remove stop words & only 1 word
#     with open("C:\\Users\\take\\PycharmProjects\\ICT_competition_mysql\\Data_Engineering\\utilities\\korean_stopwords.txt", "r", encoding="utf8") as f:
#         stop_words_ranknl = f.read().split("\n")[:-1]
#     stop_words = list(set(stop_words_ranknl + ["년", "년도", "차", "및", "모집", "추가모집", "추가", "연장", "공고", "재공고", "안내", "서식", "표", "작성가이드" "page"]))
#     df["title"] = df["title"].apply(lambda x: " ".join([i for i in x.split() if (i not in stop_words) & (len(i) > 1)]))
#     # df_test_x["title"] = df_test_x["title"].apply(lambda x: " ".join([i for i in x.split() if (i not in stop_words) & (len(i) > 1)]))
#     return df
# def extract_nouns(df):
#     okt = Okt()
#     okt.add_dictionary(words=["rd", "ict", "cloud", "bigdata", "digital", "ai", "4차산업", "메타버스"
#                               "metaverse", "인공지능", "ar", "vr", "arvr", "블록체인", "빅데이터"], tag='Noun')
#     tmp_words = [okt.nouns(i) for i in df["title"]]
#     new_tmp_words = []
#     for i in tmp_words:
#         tmp = []
#         for j in i:
#             if len(j) > 1:
#                 tmp.append(j)
#         new_tmp_words.append(" ".join(tmp))
#     df["title"] = new_tmp_words
#     return df


def preprocessing_make_pipeline(df):
    df = prep1_lowerEng_removeNewline(df)
    df = prep2_remove_prefix(df)
    df = prep3_correct_spacing(df)
    df = prep4_remove_special_char(df)
    df = prep5_remove_content_rightside(df)
    df = prep5_remove_bracket_attmText(df)
    df = prep6_get_text_KorEng(df)
    df = prep7_spacing_between_KorEng(df)
    df = remove_stopwords(df)
    df = get_nouns(df)
    return df

def get_category(df_test_x, architecture_name="tfidf_dense_v8"):
    folder_path = "./Data_Engineering/Classifier_지원분야/"
    architecture_root_path = folder_path + "architectures/" + architecture_name + "/"
    seed_everything()

    with open(folder_path + "dataset/target_label_encoder.pkl", "rb") as f:
        target_label_encoder = pickle.load(f)
    with open(folder_path + "dataset/target_oh_encoder.pkl", "rb") as f:
        target_oh_encoder = pickle.load(f)

    df_test_x = preprocessing_make_pipeline(df_test_x)
    n_folds = 5

    test_pred = np.zeros((df_test_x.shape[0], len(target_oh_encoder.categories_[0])))
    for fold in range(n_folds):
        with open(architecture_root_path + "feature_tf/feature_tf_fold" + str(fold), "rb") as f:
            feature_tf = pickle.load(f)
        model = tf.keras.models.load_model(architecture_root_path + "models/model_fold" + str(fold))

        text_feature = df_test_x["content"].copy().to_numpy()
        for i in feature_tf:
            text_feature = i.transform(text_feature)

        test_features = (
            df_test_x[["inst"]].to_numpy(),
            text_feature
        )

        test_pred += model.predict(test_features, batch_size=128) / n_folds
    return list(target_label_encoder.inverse_transform(test_pred.argmax(axis=1)))



