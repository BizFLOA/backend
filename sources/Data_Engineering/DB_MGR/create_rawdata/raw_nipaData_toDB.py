import os
import sys
from glob import glob
sys.path.append("/Data_Engineering/DB_MGR")
sys.path.append("/Data_Engineering/Classifier_지원분야")
import re

import numpy as np
import pandas as pd
import pickle
from pandas import DataFrame as dataframe, Series as series
import pymysql
from tqdm import tqdm

pd.set_option('display.max_columns', 100)
pd.set_option('display.max_rows', 100)
pd.set_option('display.width', 1000)
pd.set_option('max_colwidth', 200)

folder_path = "C:\\Users\\take\\PycharmProjects\\ICT_competition_mysql\\Data_Engineering\\DB_MGR\\"
rawdata = pd.read_excel(folder_path + "rawdata\\20220929\\nipa_20220929_1329.xlsx", sheet_name="정보통신산업진흥원(NIPA)")

# rawdata = rawdata.iloc[5:]
rawdata.columns = ["키워드", "공고명", "접수기간", "주관부처", "등록일자", "공고상세URL", "attm_text"]
rawdata.reset_index(drop=True, inplace=True)
print(rawdata.head())


# if keywords exists
tmp_df = dataframe()
keywords = []
valid_keywords = np.where(~pd.isna(rawdata["키워드"]))[0]
for i in range(1, len(valid_keywords), 1):
    tmp = rawdata.iloc[valid_keywords[i-1]:valid_keywords[i]]
    keywords.append(tmp.iloc[0, 0])
    if len(tmp) != 1:
        tmp["키워드"] = keywords[-1]
        tmp_df = pd.concat([tmp_df, tmp.iloc[1:]], ignore_index=True)

    # if last keyword
    if i == len(valid_keywords) - 1:
        tmp = rawdata.iloc[valid_keywords[i]:]
        keywords.append(tmp.iloc[0, 0])
        if len(tmp) != 1:
            tmp["키워드"] = keywords[-1]
            tmp_df = pd.concat([tmp_df, tmp.iloc[1:]], ignore_index=True)
rawdata = tmp_df

# if keywords doesn't exist
rawdata = rawdata.drop("키워드", axis=1)

# rawdata.to_csv(folder_path + "/prepdata/prep_정보통신산업진흥원.csv", index=False, encoding="utf8")
print(rawdata.head())
print(rawdata.shape)
rawdata = rawdata.iloc[2:].reset_index(drop=True)

attm_list = []
for i in glob("C:\\정부사업공고\\Download_Files\\*.txt"):
    with open(i, "r", encoding="utf8") as f:
        attm_list.append(f.read())

attm_df = dataframe()
attm_df["path_name"] = glob("C:\\정부사업공고\\Download_Files\\*.txt")
attm_df["attm_text"] = attm_list
attm_df["gb_idx"] = attm_df["path_name"].apply(lambda x: re.search(r'GBIDX_[0-9]+',x ).group(0))

for i in range(len(attm_df)):
    int(attm_df["gb_idx"].iloc[i].split("_")[1])
    rawdata["attm_text"].iloc[int(attm_df["gb_idx"].iloc[i].split("_")[1])] = attm_df["attm_text"].iloc[i]
del attm_df


rawdata["attm_text"] = rawdata["attm_text"].fillna("")
rawdata["접수기간"] = rawdata["접수기간"].fillna("")
rawdata.sort_values(["등록일자"], ascending=False)
rawdata = rawdata.drop_duplicates(["공고명"], keep="first")
rawdata["번호"] = range(len(rawdata))
rawdata = rawdata[["번호"] + list(rawdata.columns)[:-1]]


rawdata.head()
rawdata.info()

mysql_id = "ict1"
mysql_pwd = "ict1"
db_name = "ict1"
table_name = "raw_inpa_20220929"

# idx_counter = 1
# rawdata["번호"] = list(range(idx_counter, len(rawdata)+1, 1))
# rawdata = rawdata[["번호"] + list(rawdata.columns[:-1])]

# with open(folder_path + "prepdata/inpa_idx_counter.pkl", "wb") as f:
#     pickle.dump(rawdata["번호"].max() + 1, f)

# insert initial dataset
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

# rawdata.info()

# with conn.cursor() as cursor:
#     cursor.execute("drop table raw_inpa_20220830")
# # with conn.cursor() as cursor:
# #     cursor.execute("select * from raw_inpa_20220830")
# conn.commit()

try:
    with conn.cursor() as cursor:
        sql = f"create table {table_name} ( " \
              "번호 INT not null primary key," \
              "공고명 VARCHAR(128)," \
              "신청기간정보 VARCHAR(512)," \
              "주관부처 VARCHAR(32)," \
              "등록일자 DATE," \
              "공고상세URL VARCHAR(1024)," \
              "attm_text LONGTEXT" \
              ");"
        cursor.execute(sql)
except:
    print("already exists - recreate")
    with conn.cursor() as cursor:
        cursor.execute(f"drop table {table_name}")
        sql = f"create table {table_name} ( " \
              "번호 INT not null primary key," \
              "공고명 VARCHAR(128)," \
              "신청기간정보 VARCHAR(512)," \
              "주관부처 VARCHAR(32)," \
              "등록일자 DATE," \
              "공고상세URL VARCHAR(1024)," \
              "attm_text LONGTEXT" \
              ");"
        cursor.execute(sql)



# with conn.cursor() as cursor:
#     cursor.execute("truncate raw_inpa_20220830")
# with conn.cursor() as cursor:
#     cursor.execute("select * from raw_inpa_20220830")
# conn.commit()

params = ",".join(["%s"] * len(rawdata.columns))
with conn.cursor() as cursor:
    for idx, value in tqdm(rawdata.iterrows()):
        try:
            sql = f"INSERT INTO {table_name} VALUES " \
                  f"({params})"
            cursor.execute(sql, tuple([str(i) for i in value.to_list()]))
        except:
            print("insert error", idx, value)

conn.commit()
conn.close()
print("done")
