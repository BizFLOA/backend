import numpy as np
import pandas as pd
import pickle
from pandas import DataFrame as dataframe, Series as series
import pymysql
from tqdm import tqdm

# from matplotlib import pyplot as plt
# plt.rc('font', family='NanumSquareB')
from matplotlib import rc
import matplotlib
rc('font', family='NanumSquareB')

# import matplotlib
# import matplotlib.font_manager as fm
# fm.get_fontconfig_fonts()
# # font_location = '/usr/share/fonts/truetype/nanum/NanumGothicOTF.ttf'
# font_location = 'C:/Windows/Fonts/나눔스퀘어R.ttf' # For Windows
# font_name = fm.FontProperties(fname=font_location).get_name()
# matplotlib.rc('font', family=font_name)




pd.set_option('display.max_columns', 100)
pd.set_option('display.max_rows', 100)
pd.set_option('display.width', 1000)
pd.set_option('max_colwidth', 200)

# folder_path = "/Data_Engineering/DB_MGR/"

mysql_id = "ict1"
mysql_pwd = "ict1"
db_name = "ict1"
table_name = "searching_20220929"

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
df["grant_rawstring"] = df["grant_rawstring"].fillna("")
df["grant_prepstring"] = df["grant_prepstring"].fillna("")
df.head()
df.shape


font_path = ".\\Data_Engineering\\utilities\\nanum-square\\NanumSquareB.ttf"
# 'Accent', 'Accent_r', 'Blues', 'Blues_r', 'BrBG', 'BrBG_r', 'BuGn', 'BuGn_r', 'BuPu', 'BuPu_r', 'CMRmap', 'CMRmap_r',\
# 'Dark2', 'Dark2_r', 'GnBu', 'GnBu_r', 'Greens', 'Greens_r', 'Greys', 'Greys_r',\
# 'OrRd', 'OrRd_r', 'Oranges', 'Oranges_r', 'PRGn', 'PRGn_r', 'Paired', 'Paired_r', 'Pastel1', 'Pastel1_r', 'Pastel2', 'Pastel2_r'
import matplotlib.pyplot as plt
from collections import Counter
from wordcloud import WordCloud
for i in ["연구개발", "사회적기업", "자금지원", "박람회", "해외수출", "창업", "일자리", "소상공인"]:
    tmp = df.loc[df["cluster_kmeans"] == i]
    word_list = ""
    for j in tmp["공고명_명사추출"]:
        word_list += " " + " " .join(list(set(j.split())))
    word_list = word_list.strip()
    word_cnt = 0
    word_cloud_text = ""
    counter_obj = Counter(word_list.split())
    # break
    # for i in counter_obj.most_common():
    #     if i[1] >= 3:
    #         word_cloud_text += " " + ((i[0] + " ") * i[1]).rstrip()
    #         word_cnt += 1
    # word_cloud_text = word_cloud_text.lstrip()
    # [for i in counter_obj]
    #
    # dict(counter_obj)

    print("total word :", word_cnt)
    if i in ["사회적기업", "박람회", "창업", "소상공인"]:
        wordcloud = WordCloud(width=800, height=400, scale=1.5, colormap=matplotlib.cm.inferno,
                              background_color="white", max_words=300, relative_scaling=0.5, min_font_size=10, max_font_size=100,
                              font_path=font_path).generate_from_frequencies(dict([i for i in counter_obj.most_common() if i[1] >= 3]))
    else:
        wordcloud = WordCloud(width=800, height=400, scale=1.5,
                              background_color="white", max_words=300, relative_scaling=0.5, min_font_size=10, max_font_size=100,
                              font_path=font_path).generate_from_frequencies(dict([i for i in counter_obj.most_common() if i[1] >= 3]))
    # plt.figure(figsize=(8, 8))
    # plt.figure()
    plt.imshow(wordcloud, interpolation="bilinear")
    plt.axis("off")
    plt.savefig("C:\\Users\\take\\PycharmProjects\\ICT_competition_mysql\\Data_Engineering\\Recommendation_system\\categorization_tfidf_kmeans\\viz\\word_cloud_" + i + ".png")
    plt.close()
    # break
# word_list
