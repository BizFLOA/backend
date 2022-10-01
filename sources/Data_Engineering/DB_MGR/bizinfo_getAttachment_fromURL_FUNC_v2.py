import os
import shutil
import sys
sys.path.append("/data_prep/DB_MGR")
sys.path.append("/data_prep/Classifier_지원분야")
import glob
from time import sleep

import numpy as np
import pandas as pd
import pickle
from pandas import DataFrame as dataframe, Series as series
import pymysql
from tqdm import tqdm
# from main_category_inference import get_category

from bs4 import BeautifulSoup
from urllib import request
import requests
from PyPDF2 import PdfReader
from tika import parser as tika_pdf

# pd.set_option('display.max_columns', 100)
# pd.set_option('display.max_rows', 100)
# pd.set_option('display.width', 1000)
# pd.set_option('max_colwidth', 200)

# folder_path = "C:/Users/take/PycharmProjects/ICT_competition_mysql/data_prep/"

# mysql_id = "ict1"
# mysql_pwd = "ict1"
# db_name = "ict1"
# table_name = "prepV1_bizinfo_20220830"
#
# # load raw dataset
# try:
#     conn = pymysql.connect(
#         host='localhost',  # 호출변수 정의
#         user=mysql_id,
#         password=mysql_pwd,
#         db="ict1"
#     )
# except:
#     print("ERROR : DB Connection")
#
# try:
#     with conn.cursor() as cursor:
#         query = "SELECT * FROM " + table_name
#         cursor.execute(query)
#         df = pd.DataFrame(cursor.fetchall(), columns=[i[0] for i in cursor.description])
# except:
#     print("SQL ERROR")
#
# conn.close()
#
# df.head()

#container > div.sub_cont > div.sub_cont.support_project > div.support_project_detail > div.attached_file_list > ul > li:nth-child(2) > div.file_name
#container > div.sub_cont > div.sub_cont.support_project > div.support_project_detail > div.attached_file_list > ul > li:nth-child(4) > div.file_name

#container > div.sub_cont > div.sub_cont.support_project > div.support_project_detail > div.attached_file_list > ul > li:nth-child(2) > div.right_btn > a.basic-btn01.btn-gray-bd.icon_download
#container > div.sub_cont > div.sub_cont.support_project > div.support_project_detail > div.attached_file_list > ul > li:nth-child(4) > div.right_btn > a.basic-btn01.btn-gray-bd.icon_download


# 1. crawling data & get string
def bizinfo_get_attm_string_from_file(df):

    # GBINDEX = 0
    download_folder_path = "C:\\기업마당\\Download_Files\\"
    os.makedirs(os.path.dirname(download_folder_path), exist_ok=True)

    attm_list = []
    for idx, value in enumerate(tqdm(df["공고상세URL"])):
        target_url = value
        # target_url = "https://www.bizinfo.go.kr/web/lay1/bbs/S1T122C128/AS/74/view.do?pblancId=PBLN_000000000032515"
        response = requests.get(target_url)
        # break
        if response.status_code == 200:
            html = response.text
            soup = BeautifulSoup(html, 'html.parser')
            # print(soup)
        else :
            print(response.status_code)
            attm_list.append("")
            continue

        download_button = soup.select("#container > div.sub_cont > div.sub_cont.support_project > div.support_project_detail > div.attached_file_list > ul > li > div.right_btn > a.basic-btn01.btn-gray-bd.icon_download")
        if len(download_button) != 0:
            file_rawString = soup.select("#container > div.sub_cont > div.sub_cont.support_project > div.support_project_detail > div.attached_file_list > ul > li > div.file_name")[-1]
            # file_name = " ".join(file_rawString.string.split(".")[:-1])
            file_extension = file_rawString.string.split(".")[-1]
            file_path = download_folder_path + "tmp." + file_extension

            # file_name = "2022년 비대면 IR 하반기 통합공고문"
            # file_extension = "pdf"

            try:
                request.urlretrieve("https://www.bizinfo.go.kr" + download_button[-1]["href"], file_path)
            except:
                attm_list.append("")
                continue
            # convert_file_path = download_folder_path + "GBINDEX_" + str(GBINDEX) + "&번호_" + str(11) + "&" + file_name.replace("&","_") + ".txt"
            try:
                if file_extension == "hwp":
                    os.system(f'hwp5txt --output={download_folder_path + "tmp.txt"} {file_path}')
                    # os.rename(download_folder_path + "tmp.txt", convert_file_path)
                    with open(download_folder_path + "tmp.txt", "r", encoding="utf-8") as f:
                        attm_list.append(f.read())
                    # os.remove(download_folder_path + "tmp.txt")
                elif file_extension == "pdf":
                    # # way1 - pypdf2
                    # attm_list.append("\n".join([page.extract_text() for page in PdfReader(file_path).pages]))
                    # way2 - tika
                    attm_list.append(tika_pdf.from_file(file_path)["content"])
                    # os.rename(download_folder_path + "tmp.txt", convert_file_path)
                    # os.remove(download_folder_path + "tmp.txt")
                else:
                    attm_list.append("")
            except:
                attm_list.append("")
        else:
            attm_list.append("")
        # GBINDEX += 1

        # if GBINDEX % 1000 == 0:
        #     sleep(5)

    # shutil.rmtree(download_folder_path)
    return attm_list



