import os
import sys
# sys.path.append("/data_prep/Classifier_지원분야")
# sys.path.append("/data_prep/DB_MGR")
sys.path.append("./Data_Engineering/DB_MGR")
sys.path.append("./Data_Engineering/Classifier_지원분야/sources")
sys.path.append("./Data_Engineering/utilities")
from datetime import datetime
import re
import copy
from pykospacing import Spacing

import numpy as np
import pandas as pd
import pickle
from pandas import DataFrame as dataframe, Series as series
import pymysql
from tqdm import tqdm
from scipy import sparse
# from main_category_inference import get_category
# from bizinfo_getAttachment_fromURL_FUNC_v2 import bizinfo_get_attm_string_from_file
# from main_get_grant_from_string_v2 import get_grant_from_string
# from main_get_date_from_string_v2 import get_date_from_string

pd.set_option('display.max_columns', 100)
pd.set_option('display.max_rows', 100)
pd.set_option('display.width', 1000)
pd.set_option('max_colwidth', 200)

folder_path = "./Data_Engineering/Recommendation_system/categorization_tfidf_kmeans/"

mysql_id = "ict1"
mysql_pwd = "ict1"
db_name = "ict1"
table_name = "mlv2_20220929"

# create data pipeline
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

df = df.fillna(np.nan)
df["등록일자"] = pd.to_datetime(df["등록일자"])
df["신청시작일자"] = pd.to_datetime(df["신청시작일자"])
df["신청종료일자"] = pd.to_datetime(df["신청종료일자"])
df = df.sort_values("등록일자", ascending=False)

# conn.commit()
conn.close()

df_full_x = df[["지원분야", "공고명", "attm_text"]]
df_full_x.columns = ["지원분야", "title", "attm_text"]
df_full_x = pd.concat([df_full_x.drop("지원분야", axis=1), pd.get_dummies(df_full_x[["지원분야"]])], axis=1)
# df_full_x = df_full_x[["title", "attm_text"]]

from text_prep_funcs import *

# === preprocessing ===
df_full_x = prep1_lowerEng_removeNewline(df_full_x)
df_full_x = prep2_remove_prefix(df_full_x)
df_full_x = prep3_correct_spacing(df_full_x)
df_full_x = prep4_remove_special_char(df_full_x)
df_full_x = prep5_remove_content_rightside(df_full_x)
df_full_x = prep5_remove_bracket_attmText(df_full_x)
df_full_x = prep6_get_text_KorEng(df_full_x)
df_full_x = prep7_spacing_between_KorEng(df_full_x)
df_full_x = remove_stopwords(df_full_x)
df_full_x = get_nouns(df_full_x)

# tmp_suffix = "20220930"
# with open(folder_path + "prep_dataset_" + tmp_suffix + ".pkl", "wb") as f:
#     pickle.dump(df_full_x, f)

tmp_suffix = "20220930"
with open(folder_path + "prep_dataset_" + tmp_suffix + ".pkl", "rb") as f:
    df_full_x = pickle.load(f)


# === create data pipeline ===
architencture_name = "공고명+첨부파일내용_tfidf_svd_v1"
architencture_root_path = folder_path + architencture_name + "/"
if not os.path.exists(architencture_root_path):
    os.makedirs(architencture_root_path)

from sklearn.feature_extraction.text import TfidfVectorizer
tfidf = TfidfVectorizer(min_df=3)

from sklearn.decomposition import TruncatedSVD
svd = TruncatedSVD(n_components=1024)

from scipy import sparse
train_features = svd.fit_transform(tfidf.fit_transform(df_full_x["content"]))

from numpy import random as np_rnd
np_rnd.seed(42)
shuffled_idx = np_rnd.permutation(train_features.shape[0])
train_features = train_features[shuffled_idx]


# === clustering ===
# hyper-parameter tuning (number of cluster)
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score

silhouette_score_list = []
inertia_score_list = []
for i in range(3, 21, 1):
    model_kmeans = KMeans(n_clusters=i, random_state=42)
    model_kmeans.fit(train_features)
    silhouette_score_list.append(silhouette_score(train_features, model_kmeans.labels_))
    inertia_score_list.append(model_kmeans.inertia_)
    print(i, "--->", silhouette_score_list[-1])
    print(i, "--->", inertia_score_list[-1])


silhouette_score_list = series(silhouette_score_list, index=list(range(3, 21, 1)))
print(silhouette_score_list.sort_values(ascending=False))
inertia_score_list = series(inertia_score_list, index=list(range(3, 21, 1)))
print(inertia_score_list.sort_values(ascending=False))
silhouette_score_list.to_csv(architencture_root_path + "silhouette_score.csv", index=True)
inertia_score_list.to_csv(architencture_root_path + "inertia_score.csv", index=True)







# === Training with best cluster ===
best_k = 10
model_kmeans = KMeans(n_clusters=best_k, random_state=42)
model_kmeans.fit(train_features)
del train_features

with open(architencture_root_path + "ft_tfidf.pkl", "wb") as f:
    pickle.dump(tfidf, f)

with open(architencture_root_path + "ft_svd_1024.pkl", "wb") as f:
    pickle.dump(svd, f)

with open(architencture_root_path + "model_kmeans_.pkl", "wb") as f:
    pickle.dump(model_kmeans, f)




# === Inference ===

# for custom test
architencture_name = "공고명+첨부파일내용_tfidf_svd_v1"
architencture_root_path = folder_path + architencture_name + "/"

inference_name = "지원분야_첨부파일내용_tfidf_svd_v1_inf_20220930"
inference_root_path = folder_path + inference_name + "/"
if not os.path.exists(inference_root_path):
    os.makedirs(inference_root_path)

# with open(architencture_root_path + "tfidf_categorization_tfidf_kmeans.pkl", "rb") as f:
#     tfidf = pickle.load(f)
#
# with open(architencture_root_path + "model_categorization_tfidf_kmeans.pkl", "rb") as f:
#     model_kmeans = pickle.load(f)

with open(architencture_root_path + "ft_tfidf.pkl", "rb") as f:
    tfidf = pickle.load(f)

with open(architencture_root_path + "ft_svd_1024.pkl", "rb") as f:
    svd = pickle.load(f)

with open(architencture_root_path + "model_kmeans_.pkl", "rb") as f:
    model_kmeans = pickle.load(f)


# test_features = sparse.hstack([
#     tfidf.transform(df_full_x["content"])
# ], format="csr")
test_features = svd.transform(tfidf.transform(df_full_x["content"]))


df_full_x["kmeans_cluster"] = model_kmeans.predict(test_features)
df["kmeans_cluster"] = df_full_x["kmeans_cluster"].values
print(df_full_x["kmeans_cluster"].value_counts(True))
# del test_features


# === export samples (export raw string) & define cluster property ===
cluster_sample_path = inference_root_path + "cluster_sample/"
if not os.path.exists(cluster_sample_path):
    os.makedirs(cluster_sample_path)

for cluster in range(df_full_x["kmeans_cluster"].nunique()):
    cluster_path = cluster_sample_path + "cluster_" + str(cluster) + "/"
    if not os.path.exists(cluster_path):
        os.makedirs(cluster_path)
    tmp_title = df["공고명"].loc[df_full_x["kmeans_cluster"] == cluster].reset_index(drop=True)
    tmp_content = df["attm_text"].loc[df_full_x["kmeans_cluster"] == cluster].reset_index(drop=True)
    tmp_df = pd.concat([tmp_title, tmp_content], axis=1)
    for j in tmp_df.sample(min(len(tmp_df), 100), random_state=42).iterrows():
        with open(cluster_path + "idx_" + str(j[0]) + "&title_" + j[1]["공고명"].replace("/", "_") + ".txt", "w", encoding="utf8") as f:
            f.write(re.sub(r'\n+', '\n', j[1]["attm_text"]).strip())
    df.loc[df_full_x["kmeans_cluster"] == cluster].reset_index(drop=True).to_csv(cluster_path + "output.csv", index=False, encoding="utf8")
    print(df.loc[df_full_x["kmeans_cluster"] == cluster, "지원분야"].value_counts())



