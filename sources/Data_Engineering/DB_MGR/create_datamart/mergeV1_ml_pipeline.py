import os
import sys
# sys.path.append("/data_prep/DB_MGR")
# sys.path.append("/data_prep/Classifier_지원분야")
sys.path.append("C:\\Users\\take\\PycharmProjects\\ICT_competition_mysql\\Data_Engineering\\DB_MGR")
sys.path.append("C:\\Users\\take\\PycharmProjects\\ICT_competition_mysql\\Data_Engineering\\Classifier_지원분야\\sources")

import numpy as np
import pandas as pd
import pickle
from pandas import DataFrame as dataframe, Series as series
import pymysql
from tqdm import tqdm
# from main_category_inference import get_category
# from bizinfo_getAttachment_fromURL_FUNC_v2 import bizinfo_get_attm_string_from_file
# from main_get_grant_from_string_v2 import get_grant_from_string
# from main_get_date_from_string_v2 import get_date_from_string

pd.set_option('display.max_columns', 100)
pd.set_option('display.max_rows', 100)
pd.set_option('display.width', 1000)
pd.set_option('max_colwidth', 200)

folder_path = "/Data_Engineering/"

mysql_id = "ict1"
mysql_pwd = "ict1"
db_name = "ict1"
table_name_list = [
    "prepV2_bizinfo_20220929",
    "prepV2_inpa_20220929"
]
new_table_name = "mlv2_20220929"

# load raw dataset
try:
    conn = pymysql.connect(
        host='localhost',  # 호출변수 정의
        user=mysql_id,
        password=mysql_pwd,
        db="ict1",
        charset='utf8mb4'
    )
except:
    print("ERROR : DB Connection")

df_list = []
df_cols = []
try:
    with conn.cursor() as cursor:
        for table_name in table_name_list:
            query = "SELECT * FROM " + table_name
            cursor.execute(query)
            df_list.append(pd.DataFrame(cursor.fetchall(), columns=[i[0] for i in cursor.description]))
            df_cols.append(list(df_list[-1].columns))
except:
    print("SQL ERROR")

df = pd.concat(df_list, axis=0, ignore_index=True)[list(set.intersection(*map(set, df_cols)))]
df["신청시작일자"].head()

df = df[["번호", "지원분야", "수행기관", "공고명", "공고명_명사추출", "grant_rawstring", "grant_prepstring", "등록일자", "신청시작일자", "신청종료일자", "공고상세URL", 'attm_text']]
df = df.fillna(np.nan)
df["번호"] = list(range(len(df)))
df["등록일자"] = pd.to_datetime(df["등록일자"])
df["신청시작일자"] = pd.to_datetime(df["신청시작일자"])
df["신청종료일자"] = pd.to_datetime(df["신청종료일자"])

df = df.sort_values("등록일자", ascending=False)
df.head()
df.info()

# (df.groupby(["공고명"], as_index=False).size()["size"] > 1).sum()

try:
    with conn.cursor() as cursor:
        sql = f"create table {new_table_name} (" \
              "번호 int not null primary key," \
              "지원분야 VARCHAR(32)," \
              "관리기관 VARCHAR(32)," \
              "공고명 VARCHAR(128)," \
              "공고명_명사추출 LONGTEXT," \
              "grant_rawstring VARCHAR(64)," \
              "grant_prepstring VARCHAR(64)," \
              "등록일자 DATE," \
              "신청시작일자 DATE," \
              "신청종료일자 DATE," \
              "공고상세URL VARCHAR(1024)," \
              'attm_text LONGTEXT' \
              ");"
        cursor.execute(sql)
except:
    print("already exists")
    with conn.cursor() as cursor:
        cursor.execute(f"drop table {new_table_name}")
        sql = f"create table {new_table_name} (" \
              "번호 int not null primary key," \
              "지원분야 VARCHAR(32)," \
              "관리기관 VARCHAR(32)," \
              "공고명 VARCHAR(128)," \
              "공고명_명사추출 LONGTEXT," \
              "grant_rawstring VARCHAR(64)," \
              "grant_prepstring VARCHAR(64)," \
              "등록일자 DATE," \
              "신청시작일자 DATE," \
              "신청종료일자 DATE," \
              "공고상세URL VARCHAR(1024)," \
              'attm_text LONGTEXT' \
              ");"
        cursor.execute(sql)

# with conn.cursor() as cursor:
#     cursor.execute("truncate raw_bizinfo_20220830")
# with conn.cursor() as cursor:
#     cursor.execute("select * from raw_bizinfo_20220830")
# conn.commit()

params = ",".join(["%s"] * len(df.columns))
try:
    with conn.cursor() as cursor:
        for idx, value in tqdm(df.iterrows()):
            sql = f"INSERT INTO {new_table_name} VALUES " \
                  f"({params})"
            cursor.execute(sql, tuple([None if pd.isna(i) else str(i) for i in value.to_list()]))
except:
    pass

conn.commit()
conn.close()
print("done")


