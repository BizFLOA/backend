import sys

import torch

sys.path.append("C:\\Users\\take\\PycharmProjects\\ICT_competition_mysql\\Data_Engineering\\utilities\\")
import os
import copy
import pickle

from tqdm import tqdm
import random as rnd
import pymysql

import pandas as pd
from pandas import DataFrame as dataframe, Series as series
import numpy as np
from numpy import array
from numpy import random as np_rnd
import re

from sklearn.preprocessing import LabelEncoder, OneHotEncoder
from sklearn.model_selection import StratifiedKFold
from sklearn import metrics
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.decomposition import TruncatedSVD
from scipy import sparse

# import torch
# from torch.utils.data import Dataset, DataLoader
# from tokenization_kobert import KoBertTokenizer
# from transformers import DistilBertModel

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

from ckonlpy.tag import Twitter as Okt

from pykospacing import Spacing
# from func_transformer import TransformerBlock_Dense, TransformerBlock_LSTM, TransformerBlock_BIDLSTM, TokenAndPositionEmbedding, ExtractAttentionMask

pd.set_option('display.max_columns', 100)
pd.set_option('display.max_rows', 100)
pd.set_option('display.width', 1000)

def seed_everything(seed=42):
    # python seed
    os.environ["PYTHONHASHSEED"] = str(seed)
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
def createFolder(directory):
    try:
        if not os.path.exists(directory):
            os.makedirs(directory)
    except OSError:
        print('Error: Creating directory. ' + directory)

# === version 6 ===
# kobert가 아닌 TF-IDF embedding 활용
# train 및 test에 대한 preprocessing을 같은 방식 적용

folder_path = "C:\\Users\\take\\PycharmProjects\\ICT_competition_mysql\\Data_Engineering\\Classifier_지원분야\\"

mysql_id = "ict1"
mysql_pwd = "ict1"
db_name = "ict1"
table_name = "prepV2_bizinfo_20220929"

def get_df():
    df = None
    # load raw dataset
    try:
        conn = pymysql.connect(
            host='localhost',  # 호출변수 정의
            user=mysql_id,
            password=mysql_pwd,
            db=db_name,
            charset='utf8mb4'
        )
    except:
        print("ERROR : DB Connection")

    try:
        with conn.cursor() as cursor:
            query = "SELECT * FROM " + table_name
            cursor.execute(query)
            df = pd.DataFrame(cursor.fetchall(), columns=[i[0] for i in cursor.description])
    except:
        print("SQL ERROR")
    conn.close()
    return df

df_full = get_df()

# # for test
# df_full_x = df_full_x.sample(1000, random_state=42).reset_index(drop=True)

# # for memory issue
# df_full_x["attm_text"] = df_full_x["attm_text"].apply(lambda x: x[:5000])

# # 1. data loading
# df_full_x = pd.read_excel(folder_path + "rawdata/test_202208030/20220830지원사업.xlsx")
# df_test_x1 = pd.read_csv(folder_path + "rawdata/test_202208030/예산할당_2022년_중소기업지원사업_현황대상(중앙부처).csv", encoding="cp949")
# df_test_x2 = pd.read_csv(folder_path + "rawdata/test_202208030/예산할당_2022년_중소기업지원사업_현황대상(지자체).csv", encoding="cp949")
# df_test_x2.columns = df_test_x1.columns
# df_test_x = pd.concat([df_test_x1, df_test_x2], axis=0, ignore_index=True)


# df_test_x.head()

feature_version = 1
df_full = df_full[["수행기관", "공고명", "attm_text", "지원분야"]]
df_full.columns = ["inst", "title", "attm_text"]

print(df_full.head())
# 2. preprocessing

# 2-1-1. 소관명에 대해서 지자체는 모두 같은 category로 변환
# div_category = [
#     '충청북도', '강원도', '울산광역시', '경상남도', '부산광역시', '중소벤처기업부', '서울특별시',
#    '과학기술정보통신부', '국토교통부', '경상북도', '광주광역시', '제주특별자치도', '대전광역시', '농촌진흥청',
#    '인천광역시', '전라북도', '충청남도', '전라남도', '산업통상자원부', '해양수산부', '세종특별자치시',
#    '대구광역시', '환경부', '문화체육관광부', '경기도', '특허청', '농림축산식품부', '고용노동부',
#    '금융위원회', '보건복지부', '방위사업청', '조달청', '산림청', '여성가족부', '교육부', '행정안전부',
#    '식품의약품안전처', '관세청', '방송통신위원회', '문화재청', '통일부', '기상청', '국방부',
#    '개인정보보호위원회', '소방청', '국세청', '기획재정부', '외교부', '공정거래위원회', '법무부'
# ]
# admin_div = [
#     '충청북도', '강원도', '울산광역시', '경상남도', '부산광역시', '서울특별시',
#     '경상북도', '광주광역시', '제주특별자치도', '대전광역시',
#     '인천광역시', '전라북도', '충청남도', '전라남도', '세종특별자치시',
#     '대구광역시', '경기도'
# ]
# len(admin_div)
#
# df_full_x["inst"] = df_full_x["inst"].apply(lambda x: "none" if x in admin_div else x)
# df_test_x["inst"] = df_test_x["inst"].apply(lambda x: "none" if x in admin_div else x)

# df_full_x.head()
# df_test_x.head()


print(df_full.head())


# def prep1_lowerEng_removeNewline(df):
#     print("prep1_lowerEng_removeNewline start")
#     # cols : title, attm_text
#     # 영어를 모두 소문자로 변환 및 개행문자를 공백으로 변환
#     if "title" in df.columns:
#         df["title"] = df["title"].apply(lambda x: x.lower())
#         df["title"] = df["title"].apply(lambda x: " ".join(x.replace("\n", " ").replace("\t", " ").replace("_x000d_", " ").split()))
#
#     df["attm_text"] = df["attm_text"].apply(lambda x: x.lower())
#     df["attm_text"] = df["attm_text"].apply(lambda x: " ".join(x.replace("\n", " ").replace("\t", " ").replace("_x000d_", " ").split()))
#     print("prep1_lowerEng_removeNewline end")
#     return df
# def prep2_remove_prefix(df):
#     print("prep2_remove_prefix start")
#     # cols : title, attm_text
#     # K-Drama, K-Bio 처럼 'K-' 접두사 제거 및 [충북] 과 같은 단어 제거
#     if "title" in df.columns:
#         prep_str = []
#         for i in df["title"]:
#             tmp = copy.deepcopy(i)
#             # index = None
#             while True:
#                 index = re.search(r'k-[a-z가-힣]', tmp)
#                 if index is None:
#                     break
#                 else:
#                     target_pos = index.span()
#                     tmp = "".join([tmp[:(target_pos[0])], tmp[(target_pos[1] - 1):]])
#             prep_str.append(tmp)
#         df["title"] = prep_str
#
#         prep_str = []
#         for i in df["title"]:
#             if i.lstrip()[0] == "[":
#                 tmp = i.split("]")
#                 prep_str.append(tmp[0]) if len(tmp) == 1 else prep_str.append(" ".join(tmp[1:]))
#             else:
#                 prep_str.append(i)
#         df["title"] = prep_str
#
#     prep_str = []
#     for i in df["attm_text"]:
#         tmp = copy.deepcopy(i)
#         while True:
#             index = re.search(r'k-[a-z가-힣]', tmp)
#             if index is None:
#                 break
#             else:
#                 target_pos = index.span()
#                 tmp = "".join([tmp[:(target_pos[0])], tmp[(target_pos[1] - 1):]])
#         prep_str.append(tmp)
#
#     df["attm_text"] = prep_str
#     print("prep2_remove_prefix end")
#     return df
# def prep3_correct_spacing(df):
#     print("prep3_correct_spacing start")
#     # cols : title
#     # 훈련 및 테스트 데이터 내 공고명 띄어쓰기 교정
#     if "title" in df.columns:
#         spacing = Spacing(rules=['r&d', "s&m", "g&a", "rd", "ict", "cloud", "bigdata", "digital", "digital transformation", "digitaltransformation", "딥러닝", "deeplearning", "deep learning",
#                "머신러닝", "machine learning", "machinelearning", "디지털", "디지털트랜스포메이션", "디지털 트랜스포메이션", "ai", "4차산업", "메타버스", "sw",
#                "metaverse", "meta verse", "인공지능", "인공 지능", "ar", "vr", "xr", "arvr", "블록체인", "blockchain", "block chain", "빅데이터"])
#         df["title"] = df["title"].apply(lambda x: spacing(x))
#     print("prep3_correct_spacing end")
#     return df
# def prep4_remove_special_char(df):
#     print("prep4_remove_special_char start")
#     # cols : title, attm_text
#     # 영어 내 R&D, S&M 과 같은 약어에서 특수문자에서 &를 제거
#     if "title" in df.columns:
#         prep_str = []
#         for i in df["title"]:
#             tmp = copy.deepcopy(i)
#             while True:
#                 index = re.search(r'[a-z]&[a-z]', tmp)
#                 if index is None:
#                     break
#                 else:
#                     target_pos = index.span()
#                     tmp = "".join([tmp[:(target_pos[0] + 1)], tmp[(target_pos[0] + 2):]])
#             prep_str.append(tmp)
#         df["title"] = prep_str
#
#     prep_str = []
#     for i in df["attm_text"]:
#         tmp = copy.deepcopy(i)
#         while True:
#             index = re.search(r'[a-z]&[a-z]', tmp)
#             if index is None:
#                 break
#             else:
#                 target_pos = index.span()
#                 tmp = "".join([tmp[:(target_pos[0] + 1)], tmp[(target_pos[0] + 2):]])
#         prep_str.append(tmp)
#     df["attm_text"] = prep_str
#     print("prep4_remove_special_char end")
#     return df
# def prep5_remove_content_rightside(df):
#     print("prep5_remove_content_rightside start")
#     # cols : title
#     # 훈련 및 테스트 데이터 내 맨 뒤 괄호 안 내용 제거 - 상위 지원사업 공고명이 여러 세부 공고명 뒤에 일괄적으로 붙는 경우
#     # ex. 기술 - 반도체특허(4차산업), 인력 - SW인력지원(4차산업)
#     if "title" in df.columns:
#         prep_str = []
#         for i in df["title"]:
#             if i.rstrip()[-1] == ")":
#                 tmp = i.split("(")
#                 # 마지막 괄호는 있으나 시작하는 괄호는 없을 경우 그냥 원래 문자열 삽입
#                 prep_str.append(tmp[0]) if len(tmp) == 1 else prep_str.append(" ".join(tmp[:-1]))
#             else:
#                 prep_str.append(i)
#         df["title"] = prep_str
#     print("prep5_remove_content_rightside end")
#     return df
# def prep5_remove_bracket_attmText(df):
#     print("prep5_remove_bracket_attmText start")
#     # cols : attm_text
#     # 첨부 파일의 경우 <> 괄호 안 내용은 필요없는 부분이 많으므로 제거
#     # ex. <표>, <서식>, <그림1>, <'URL인코딩문자열'>
#     re_obj = re.compile(r'<[^>]*>')
#     df["attm_text"] = df["attm_text"].apply(lambda x: re.sub(re_obj, "", x))
#     print("prep5_remove_bracket_attmText end")
#     return df
# def prep6_get_text_KorEng(df):
#     print("prep6_get_text_KorEng start")
#     # cols : title, attm_text
#     # 한글 및 영어만 추출 & 공백 한 칸으로 제한
#     if "title" in df.columns:
#         prep_str = []
#         for i in df["title"]:
#             prep_str.append(" ".join(re.sub('[^ a-z가-힣]', ' ', i).split()))
#         df["title"] = prep_str
#
#     prep_str = []
#     for i in df["attm_text"]:
#         prep_str.append(" ".join(re.sub('[^ a-z가-힣]', ' ', i).split()))
#     df["attm_text"] = prep_str
#     print("prep6_get_text_KorEng end")
#     return df
# def prep7_spacing_between_KorEng(df):
#     print("prep7_spacing_between_KorEng start")
#     # cols : title, attm_text
#     # 한글과 영어 사이 공백으로 분리
#     if "title" in df.columns:
#         prep_str = []
#         for i in df["title"]:
#             tmp = copy.deepcopy(i)
#             while True:
#                 index = re.search('[a-z][가-힣]|[가-힣][a-z]', tmp)
#                 if index is None:
#                     break
#                 else:
#                     target_pos = index.span()
#                     tmp = " ".join([tmp[:(target_pos[0] + 1)], tmp[(target_pos[0] + 1):]])
#             prep_str.append(tmp)
#         df["title"] = prep_str
#
#     prep_str = []
#     for i in df["attm_text"]:
#         tmp = copy.deepcopy(i)
#         while True:
#             index = re.search('[a-z][가-힣]|[가-힣][a-z]', tmp)
#             if index is None:
#                 break
#             else:
#                 target_pos = index.span()
#                 tmp = " ".join([tmp[:(target_pos[0] + 1)], tmp[(target_pos[0] + 1):]])
#         prep_str.append(tmp)
#     df["attm_text"] = prep_str
#     print("prep7_spacing_between_KorEng end")
#     return df
# def remove_stopwords(df):
#     print("remove_stopwords start")
#     # remove stop words & only 1 word
#     with open("./Data_Engineering/utilities/korean_stopwords.txt", "r",
#               encoding="utf8") as f:
#         stop_words_ranknl = f.read().split("\n")[:-1]
#     stop_words = list(set(stop_words_ranknl + ["년", "년도", "차", "및", "모집", "추가모집", "추가", "연장", "공고", "재공고", "안내", "서식", "표", "작성가이드",
#                                                "붙임", "첨부", "페이지", "page"]))
#     if "title" in df.columns:
#         df["title"] = df["title"].apply(lambda x: " ".join([i for i in x.split() if (i not in stop_words) & (len(i) > 1)]))
#     df["attm_text"] = df["attm_text"].apply(lambda x: " ".join([i for i in x.split() if (i not in stop_words) & (len(i) > 1)]))
#     print("remove_stopwords end")
#     return df
# def get_nouns(df):
#     print("get_nouns start")
#     # extract nouns
#     okt = Okt()
#     # 'sw' 단어 추가 필요
#     okt.add_dictionary(
#         words=["rd", "ict", "cloud", "bigdata", "digital", "digital transformation", "digitaltransformation", "딥러닝", "deeplearning", "deep learning",
#                "머신러닝", "machine learning", "machinelearning", "디지털", "디지털트랜스포메이션", "디지털 트랜스포메이션", "ai", "4차산업", "메타버스", "sw",
#                "metaverse", "meta verse", "인공지능", "인공 지능", "ar", "vr", "xr", "arvr", "블록체인", "blockchain", "block chain", "빅데이터"], tag='Noun'
#     )
#     # concat title & attachment text
#     if ("title" in df.columns) & ("attm_text" in df.columns):
#         df["content"] = df["title"] + " " + df["attm_text"]
#         df["content"] = df["content"].apply(lambda x: x.rstrip())
#         # df["title"] = df["title"].apply(lambda x: " ".join([i for i in okt.nouns(x) if len(i) > 1]))
#         # df["attm_text"] = df["attm_text"].apply(lambda x: " ".join([i for i in okt.nouns(x) if len(i) > 1]))
#         df["content"] = df["content"].apply(lambda x: " ".join([i for i in okt.nouns(x) if len(i) > 1]))
#     elif "title" in df.columns:
#         df["title"] = df["title"].apply(lambda x: " ".join([i for i in okt.nouns(x) if len(i) > 1]))
#     elif "attm_text" in df.columns:
#         df["attm_text"] = df["attm_text"].apply(lambda x: " ".join([i for i in okt.nouns(x) if len(i) > 1]))
#     print("get_nouns end")
#     return df
# def get_feature_vector(df):
#     print("get_feature_vector start")
#
#     distilbert_tokenizer = KoBertTokenizer.from_pretrained('monologg/kobert')
#     distilbert_model = DistilBertModel.from_pretrained('monologg/distilkobert')
#
#     df["content"] = df["title"] + " " + df["attm_text"]
#     df["content"] = df["content"].apply(lambda x: x.rstrip())
#     fv_list = []
#     for i in DataLoader(df["content"], batch_size=32):
#         token_obj = distilbert_tokenizer.batch_encode_plus(i, max_length=512, padding=True, truncation=True)
#         output_obj = distilbert_model(input_ids=torch.tensor(token_obj["input_ids"]), attention_mask=torch.tensor(token_obj["attention_mask"]))
#         fv_list.append(output_obj.last_hidden_state.mean(axis=1).detach().numpy())
#
#     print("get_feature_vector end")
#     return np.concatenate(fv_list, axis=0)
# def get_sentencepiece_tokens(df):
#     print("get_feature_vector start")
#
#     distilbert_tokenizer = KoBertTokenizer.from_pretrained('monologg/kobert')
#     # distilbert_model = DistilBertModel.from_pretrained('monologg/distilkobert')
#
#     df["content"] = df["title"] + " " + df["attm_text"]
#     df["content"] = df["content"].apply(lambda x: x.rstrip())
#     input_ids_list = []
#     attn_mask_list = []
#     for i in DataLoader(df["content"], batch_size=32):
#         token_obj = distilbert_tokenizer.batch_encode_plus(i, max_length=1024, padding='max_length', truncation=True)
#         # output_obj = distilbert_model(input_ids=torch.tensor(token_obj["input_ids"]), attention_mask=torch.tensor(token_obj["attention_mask"]))
#         input_ids_list.append(array(token_obj["input_ids"]))
#         attn_mask_list.append(array(token_obj["attention_mask"]))
#
#     print("get_feature_vector end")
#     return (np.concatenate(input_ids_list, axis=0), np.concatenate(attn_mask_list, axis=0))

# === preprocessing ===

from text_prep_funcs import *

df_full = prep1_lowerEng_removeNewline(df_full)
df_full = prep2_remove_prefix(df_full)
df_full = prep3_correct_spacing(df_full)
df_full = prep4_remove_special_char(df_full)
df_full = prep5_remove_content_rightside(df_full)
df_full = prep5_remove_bracket_attmText(df_full)
df_full = prep6_get_text_KorEng(df_full)
df_full = prep7_spacing_between_KorEng(df_full)
df_full = get_nouns(df_full)
df_full = remove_stopwords(df_full)

# df_full["inst", "title", "attm_text"] = prep1_lowerEng_removeNewline(df_full["inst", "title", "attm_text"])
# df_full["inst", "title", "attm_text"] = prep2_remove_prefix(df_full["inst", "title", "attm_text"])
# df_full["inst", "title", "attm_text"] = prep3_correct_spacing(df_full["inst", "title", "attm_text"])
# df_full["inst", "title", "attm_text"] = prep4_remove_special_char(df_full["inst", "title", "attm_text"])
# df_full["inst", "title", "attm_text"] = prep5_remove_content_rightside(df_full["inst", "title", "attm_text"])
# df_full["inst", "title", "attm_text"] = prep5_remove_bracket_attmText(df_full["inst", "title", "attm_text"])
# df_full["inst", "title", "attm_text"] = prep6_get_text_KorEng(df_full["inst", "title", "attm_text"])
# df_full["inst", "title", "attm_text"] = prep7_spacing_between_KorEng(df_full["inst", "title", "attm_text"])
# df_full["inst", "title", "attm_text"] = remove_stopwords(df_full["inst", "title", "attm_text"])
# df_full["inst", "title", "attm_text"] = get_nouns(df_full["inst", "title", "attm_text"])

# df = df_full_x
# df["content"] = df["title"] + " " + df["attm_text"]
# df["content"] = df["content"].apply(lambda x: x.rstrip())
# input_ids_list = []
# attn_mask_list = []
# for i in DataLoader(df["content"], batch_size=32):
#     token_obj = distilbert_tokenizer.batch_encode_plus(i, max_length=1024, padding=True, truncation=True)
#     # output_obj = distilbert_model(input_ids=torch.tensor(token_obj["input_ids"]), attention_mask=torch.tensor(token_obj["attention_mask"]))
#     input_ids_list.append(array(token_obj["input_ids"]))
#     attn_mask_list.append(array(token_obj["attention_mask"]))
#     break
#
# input_ids_list[0].shape
df_full = df_full.drop(["title", "attm_text"], axis=1)

# with open(folder_path + "dataset\\feV" + str(feature_version) + "_df_full", "wb") as f:
#     pickle.dump(df_full, f)

with open(folder_path + "dataset\\feV" + str(feature_version) + "_df_full", "rb") as f:
    df_full = pickle.load(f)

# target_label_encoder = LabelEncoder()
# df_full_y = target_label_encoder.fit_transform(df_full_y).reshape(-1, 1)
# with open(folder_path + "dataset\\target_label_encoder.pkl", "wb") as f:
#     pickle.dump(target_label_encoder, f)
#
# target_oh_encoder = OneHotEncoder(sparse=False, handle_unknown="ignore", dtype="float32")
# target_oh_encoder.fit(df_full_y)
# with open(folder_path + "dataset\\target_oh_encoder.pkl", "wb") as f:
#     pickle.dump(target_oh_encoder, f)

# df_full_y = dataframe(df_full_y)


# # 단어 수 분포 파악
# df_word = series([len(i.split()) for i in df_full_x[:,1]])
# df_word.describe()
# df_word.hist()


# === training ===
architecture_name = "tfidf_dense_v8"
# dataset_name = "tfidf_dense_v8"
architecture_root_path = folder_path + "architectures\\" + architecture_name + "\\"
createFolder(architecture_root_path)
createFolder(architecture_root_path + "models")
createFolder(architecture_root_path + "metrics")
# # MAX_LEN = 512
n_svd = 1024
# tfidf_size = -1


# with open(folder_path + "dataset\\" + dataset_name + "+df_full_x", "wb") as f:
#     pickle.dump(df_full_x, f)
# # with open(folder_path + "dataset\\" + dataset_name + "+df_full_x_input_ids", "wb") as f:
# #     pickle.dump(df_full_x_input_ids, f)
# # with open(folder_path + "dataset\\" + dataset_name + "+df_full_x_attn_mask", "wb") as f:
# #     pickle.dump(df_full_x_attn_mask, f)
# with open(folder_path + "dataset\\" + dataset_name + "+df_full_y", "wb") as f:
#     pickle.dump(df_full_y, f)




# with open(folder_path + "dataset\\" + dataset_name + "+df_full_x", "rb") as f:
#     df_full_x = pickle.load(f)
# # with open(folder_path + "dataset\\" + dataset_name + "+df_full_x_input_ids", "rb") as f:
# #     df_full_x_input_ids = pickle.load(f)
# # with open(folder_path + "dataset\\" + dataset_name + "+df_full_x_attn_mask", "rb") as f:
# #     df_full_x_attn_mask = pickle.load(f)
# with open(folder_path + "dataset\\" + dataset_name + "+df_full_y", "rb") as f:
#     df_full_y = pickle.load(f)


# with open(folder_path + "dataset\\target_label_encoder.pkl", "rb") as f:
#     target_label_encoder = pickle.load(f)
# with open(folder_path + "dataset\\target_oh_encoder.pkl", "rb") as f:
#     target_oh_encoder = pickle.load(f)




# # shuffling
# np_rnd.seed(42)
# shuffled_idx = np_rnd.permutation(df_full_x.shape[0])
# df_full_x = df_full_x.iloc[shuffled_idx].reset_index(drop=True)
# df_full_y = df_full_y.iloc[shuffled_idx].reset_index(drop=True)
#
# print(df_full_x.iloc[:3])
# print(df_full_y.iloc[:3])

# shuffling
df_full = df_full.sample(frac=1, random_state=42)
# np_rnd.seed(42)
# shuffled_idx = np_rnd.permutation(df_full_x.shape[0])
# df_full_x = df_full_x.iloc[shuffled_idx].reset_index(drop=True)
# df_full_y = df_full_y.iloc[shuffled_idx].reset_index(drop=True)

target_label_encoder = LabelEncoder()
df_full[["지원분야"]] = target_label_encoder.fit_transform(df_full[["지원분야"]]).reshape(-1, 1)
target_oh_encoder = OneHotEncoder(sparse=False, handle_unknown="ignore", dtype="float32")
target_oh_encoder.fit(df_full[["지원분야"]])

# print(df_full_x.iloc[:3])
# print(df_full_y.iloc[:3])

def convert_sparse_matrix_to_sparse_tensor(x):
    coo = x.tocoo()
    indices = np.mat([coo.row, coo.col]).transpose()
    return tf.sparse.reorder(tf.sparse.SparseTensor(indices, coo.data, coo.shape))

# def create_model(vocab_inst=["1", "2", "3"], vocab_content=["1", "2", "3"]):
#     dropoutRate = 0.5
#
#     input1 = layers.Input(shape=(1,), dtype=tf.string)
#     x = layers.StringLookup(vocabulary=vocab_inst, output_mode='int')(input1)
#     x = layers.Embedding(input_dim=len(vocab_inst)+1, output_dim=max(4, len(vocab_inst) // 4),
#                          embeddings_initializer=tf.keras.initializers.TruncatedNormal(mean=0.25, stddev=0.1))(x)
#     concat1 = layers.Flatten()(x)
#
#     input2 = layers.Input(shape=(len(vocab_content),), dtype=tf.float32)
#     concat2 = input2
#
#     x = layers.Concatenate()([concat1, concat2])
#     x = tfa.layers.NoisyDense(2048, activation="relu")(x)
#
#     last_layer = tfa.layers.NoisyDense(512, activation="relu")(x)
#
#     classifier = layers.Dense(len(target_oh_encoder.categories_[0]), activation="softmax")(last_layer)
#     return Model([input1, input2], classifier)

# def create_model(vocab_inst=["1", "2", "3"], vocab_content=["1", "2", "3"]):
#     dropoutRate = 0.5
#
#     input1 = layers.Input(shape=(1,), dtype=tf.string)
#     x = layers.StringLookup(vocabulary=vocab_inst, output_mode='int')(input1)
#     x = layers.Embedding(input_dim=len(vocab_inst)+1, output_dim=max(4, len(vocab_inst) // 4),
#                          embeddings_initializer=tf.keras.initializers.TruncatedNormal(mean=0.25, stddev=0.1))(x)
#     concat1 = layers.Flatten()(x)
#
#     input2 = layers.Input(shape=(768,), dtype=tf.float32)
#     concat2 = input2
#
#     x = layers.Concatenate()([concat1, concat2])
#     x = tfa.layers.NoisyDense(256, activation="relu")(x)
#
#     x = layers.Dropout(dropoutRate)(x)
#     last_layer = layers.Dense(64, activation="relu")(x)
#
#     classifier = layers.Dense(len(target_oh_encoder.categories_[0]), activation="softmax")(last_layer)
#     return Model([input1, input2], classifier)

dropoutRate = 0.5
def create_model(vocab_inst=["1", "2", "3"], vocab_content=["1", "2", "3"]):
    input1 = layers.Input(shape=(1,), dtype=tf.string)
    x1 = layers.StringLookup(vocabulary=vocab_inst, output_mode='int')(input1)
    x1 = layers.Embedding(input_dim=len(vocab_inst)+1, output_dim=4)(x1)
    concat1 = layers.Flatten()(x1)

    input2 = layers.Input(shape=(n_svd,), dtype=tf.float32)
    concat2 = input2

    x = layers.Concatenate()([concat1, concat2])
    hidden1 = layers.Dense(256, activation="relu", activity_regularizer="l2")(x)

    hidden2 = layers.Dropout(dropoutRate)(hidden1)
    hidden2 = tfa.layers.WeightNormalization(
        layers.Dense(units=128, activation='relu', kernel_initializer="lecun_normal")
    )(hidden2)

    hidden3 = layers.Dropout(dropoutRate)(layers.Concatenate()([hidden1, hidden2]))
    hidden3 = tfa.layers.WeightNormalization(
        layers.Dense(units=128, activation='relu', kernel_initializer="lecun_normal")
    )(hidden3)

    output = layers.Dropout(dropoutRate)(layers.Concatenate()([hidden1, hidden2, hidden3]))
    last_layer = tfa.layers.WeightNormalization(
        layers.Dense(units=128, activation='relu', kernel_initializer="lecun_normal")
    )(output)

    classifier = layers.Dense(len(target_oh_encoder.categories_[0]), activation="softmax")(last_layer)
    return Model([input1, input2], classifier, name="model_" + str(fold))

print(create_model().summary())




def create_dataset(x, y, batch_size, shuffle=True, sample_weight=None):
    if sample_weight is not None:
        dataset = tf.data.Dataset.from_tensor_slices((x, y, sample_weight))
    else:
        dataset = tf.data.Dataset.from_tensor_slices((x, y))
    dataset = dataset.shuffle(int(batch_size * 8), reshuffle_each_iteration=True) if shuffle else dataset
    dataset = dataset.batch(batch_size)
    dataset = dataset.prefetch(tf.data.AUTOTUNE)
    return dataset



def do_fold_training(train_idx, valid_idx):
    print(f"=== fold {fold} ===")
    # # tfidf = tf.keras.layers.TextVectorization(
    # #     max_tokens=tfidf_size,
    # #     output_mode='tf_idf',
    # #     pad_to_max_tokens=True
    # # )
    # tfidf = tf.keras.layers.TextVectorization(
    #     max_tokens=8192,
    #     output_mode='int',
    #     output_sequence_length=64,
    # )
    # tfidf.adapt(df_full_x["title"].iloc[train_idx].to_list())
    # vocab_size = tfidf.vocabulary_size()
    # tfidf_vectorizer = Sequential([
    #     layers.Input(shape=(), dtype=tf.string),
    #     tfidf
    # ])

    tfidf = TfidfVectorizer(min_df=3, norm="l2", dtype=np.float32)
    svd = TruncatedSVD(n_components=n_svd, random_state=42)

    train_features = (
        df_full[["inst"]].iloc[train_idx].to_numpy(),
        svd.fit_transform(tfidf.fit_transform(df_full["content"].iloc[train_idx]))
    )

    valid_features = (
        df_full[["inst"]].iloc[valid_idx].to_numpy(),
        svd.transform(tfidf.transform(df_full["content"].iloc[valid_idx]))
    )

    train_ds = create_dataset(train_features, target_oh_encoder.transform(df_full[["지원분야"]].iloc[train_idx]), batch_size=batch_size, shuffle=True)
    valid_ds = create_dataset(valid_features, target_oh_encoder.transform(df_full[["지원분야"]].iloc[valid_idx]), batch_size=batch_size, shuffle=False)

    vocab_inst = df_full["inst"].unique()
    # vocab_content = []
    # for i in df_full_x[["content"]].iloc[train_idx, 0]:
    #     vocab_content.extend(i.split())
    # vocab_content = array(list(set(vocab_content)))

    # model = create_model(vocab_inst=vocab_inst, vocab_content=vocab_content)
    model = create_model(vocab_inst=vocab_inst, vocab_content=tfidf.get_feature_names_out())

    total_iter = int(np.ceil(len(train_idx) / batch_size) * epochs)
    SCHEDULE_BOUNDARIES = [int(total_iter * 0.2), int(total_iter * 0.4), int(total_iter * 0.6), int(total_iter * 0.8)]
    lr_schedule = tf.keras.optimizers.schedules.PiecewiseConstantDecay(
        boundaries=SCHEDULE_BOUNDARIES,
        values=[5e-4, 1e-4, 5e-5, 1e-5, 5e-6],
    )

    cb_earlyStopping = tf_callbacks.EarlyStopping(
        patience=patient_epochs, monitor='val_f1', mode='max'
    )
    cb_modelsave = tf_callbacks.ModelCheckpoint(
        filepath=checkpoint_filepath, monitor='val_f1', mode='max', save_weights_only=True, save_best_only=True
    )

    model.compile(
        loss=tf.keras.losses.CategoricalCrossentropy(name="loss"),
        optimizer=tfa.optimizers.AdamW(learning_rate=lr_schedule, weight_decay=1e-4),
        metrics=[
            tf_metrics.CategoricalAccuracy(name="accuracy"),
            tfa.metrics.F1Score(num_classes=len(target_oh_encoder.categories_[0]), average="macro", name="f1")
        ]
    )

    model.fit(train_ds, validation_data=valid_ds, epochs=epochs, verbose=0,
              callbacks=[cb_earlyStopping, cb_modelsave, TqdmCallback(verbose=0)])
    model.load_weights(checkpoint_filepath)
    model.trainable = False
    valid_pred[valid_idx] += model.predict(valid_features)

    fold_metrics.append(
        {
            "train_logloss": metrics.log_loss(df_full[["지원분야"]].iloc[train_idx, 0], model.predict(train_features)),
            "train_accuracy": metrics.accuracy_score(df_full[["지원분야"]].iloc[train_idx, 0], model.predict(train_features).argmax(axis=1)),
            "train_f1": metrics.f1_score(df_full[["지원분야"]].iloc[train_idx, 0], model.predict(train_features).argmax(axis=1), average="macro"),
            "valid_logloss": metrics.log_loss(df_full[["지원분야"]].iloc[valid_idx, 0], valid_pred[valid_idx]),
            "valid_accuracy": metrics.accuracy_score(df_full[["지원분야"]].iloc[valid_idx, 0], valid_pred[valid_idx].argmax(axis=1)),
            "valid_f1": metrics.f1_score(df_full[["지원분야"]].iloc[valid_idx, 0], valid_pred[valid_idx].argmax(axis=1), average="macro"),
        }
    )
    print(fold_metrics[-1])
    fold_feature_tf.append((tfidf, svd))
    fold_models.append(model)







# setting learning parameter
epochs = 30
patient_epochs = 10
batch_size = 32
checkpoint_filepath = folder_path + "tmp/"

n_folds = 5
kfolds_spliter = StratifiedKFold(n_folds, shuffle=True, random_state=42)

valid_pred = np.zeros((df_full.shape[0], len(target_oh_encoder.categories_[0])))
fold_models = []
fold_feature_tf = []
fold_metrics = []


for fold, (train_idx, valid_idx) in enumerate(kfolds_spliter.split(range(len(df_full)), df_full["지원분야"])):
    seed_everything(42)
    tf.keras.backend.clear_session()
    do_fold_training(train_idx, valid_idx)

# # model & fold data save
# for fold, (model_obj, tfidf_obj, metric_obj) in enumerate(zip(fold_models, fold_tfidf, fold_metrics)):
#     model_obj.save(architecture_root_path + "models\\model_fold" + str(fold))
#     with open(architecture_root_path + "metrics\\tfidf_fold" + str(fold), "wb") as f:
#         pickle.dump(tfidf_obj, f)
#     with open(architecture_root_path + "metrics\\metric_fold" + str(fold), "wb") as f:
#         pickle.dump(metric_obj, f)


# def create_final_model(base_models):
#     common_input = create_model().input
#
#     x = layers.Concatenate(axis=1)([layers.Reshape((1, -1))(i(common_input)) for i in base_models])
#     output = layers.GlobalAveragePooling1D()(x)
#
#     return Model(common_input, output, name="final_model")
#
# final_model = create_final_model(fold_models)
# final_model.summary()

# model & fold data save
for fold, (model_obj, feature_tf_obj, metric_obj) in enumerate(zip(fold_models, fold_feature_tf, fold_metrics)):
    model_obj.save(architecture_root_path + "models\\model_fold" + str(fold))
    with open(architecture_root_path + "metrics\\feature_tf_fold" + str(fold), "wb") as f:
        pickle.dump(feature_tf_obj, f)
    with open(architecture_root_path + "feature_tf\\metrics_fold" + str(fold), "wb") as f:
        pickle.dump(metric_obj, f)

# # model & fold data save
# for fold, (model_obj, metric_obj) in enumerate(zip(fold_models, fold_metrics)):
#     model_obj.save(architecture_root_path + "models\\model_fold" + str(fold))
#     # with open(architecture_root_path + "metrics\\tfidf_fold" + str(fold), "wb") as f:
#     #     pickle.dump(tfidf_obj, f)
#     with open(architecture_root_path + "metrics\\metric_fold" + str(fold), "wb") as f:
#         pickle.dump(metric_obj, f)

df_score = dataframe(fold_metrics)
df_score.loc["average"] = df_score.mean()
df_score.reset_index().to_csv(architecture_root_path + "metrics\\score_table.csv", index=True)
print(df_score)



