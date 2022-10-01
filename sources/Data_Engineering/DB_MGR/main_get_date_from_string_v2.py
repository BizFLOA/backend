import dateutil.parser as dparser
import datefinder
import re
from glob import glob
import pandas as pd

# convert_dic = {
#     "신청기간": [
#         "신청기간", "공고기간", "접수기간", "지원기간",
#         "신청 기간", "공고 기간", "접수 기간", "지원 기간",
#         "본 접수기간", "신청서 제출기간", "온라인 신청 제출기간", "온라인신청 제출기간",
#         "본 접수 기간", "신청서 제출 기간", "온라인 신청 제출 기간", "온라인신청 제출 기간",
#     ]
# }

convert_dic = {
    "신청기간": [
        "신청기간", "공고기간", "접수기간", "점수마감", "신청서 접수마감", "신청서 접수기한",
        "신청 기간", "공고 기간", "접수 기간", "접수 마감", "신청서 접수 마감", "신청서 접수 기한",
        "본 접수기간", "신청서 제출기간", "온라인 신청 제출기간", "온라인신청 제출기간",
        "본 접수 기간", "신청서 제출 기간", "온라인 신청 제출 기간", "온라인신청 제출 기간",
    ]
}

# INFO : 폴더 경로는 직접설정필요
# test_cases = [
#     "신청기간 : 2022.3.4.(금) ~ 2022.4.4(월) 10:00까지",
#     "신청서 제출기간 : 2022년 5월 11일(수) ~ 31일(화)",
#     "조사기간 : 2022. 1. 20.(목) ~ 2022. 02. 08.(화) 16:00 시까지",
#     "신청서 제출기간 : 2022년 8월 12일(금)~ 8월 24일(수)(15시까지)",
#     "신청서 제출기간 : 2022년 5월 31일(화) 이후 수시 신청",
#     "신청기간 : 2022년 6월 17일(금) ~ 7월 29일(금)",
#     "(공고기간) 2022. 2. 28(월) ~ 2022. 4. 5(화) (접수마감) 2022. 4. 5(화) 16:00까지",
#     "(공고기간) 2022. 6. 7(화) ~ 2022. 7. 6(수) (접수마감) 2022. 7. 6(수) 16:00까지",
#     "(접수기간) ’22.2.22(화)~’22.3.31(목) 16:00까지",
#     "신청기간 :‘22. 1. 26.(수) ~ 2. 25(금) 총 30일간"
# ]


def get_date_from_string(file_str_list, max_len=40, findInt_maxLen=10):
    # # # # for test
    # # # folder_path = "C:\\정부사업공고\\Download_Files\\"
    # # # folder_path = "C:\\Users\\take\\Desktop\\임시파일\\"
    # # folder_path = "C:\\Users\\take\\Desktop\\에러1\\"
    # max_len = 40
    # findInt_maxLen = 10
    # file_str_list = []
    # # for i in sorted(glob(folder_path + "*.txt"))[:1]:
    # #     with open(i, encoding="utf8") as f:
    # #         file_str_list.append(f.read())
    # # file_str_list = df_test_x["attm_text"].copy().iloc[3:4]
    # #
    # with open("C:\\Users\\take\\Desktop\\에러11.txt", encoding="utf8") as f:
    #     file_str_list.append(f.read())

    result = []
    # 년원일 추출 정규표현식 v1
    # regex_ymd = [
    #     re.compile(r'\d{4}[.가-힣 ]{0,2}\d{1,2}[.가-힣 ]{0,2}\d{1,2}[.가-힣]{0,1}'),
    #     re.compile(r'\d{4}[.가-힣 ]{0,2}\d{1,2}[.가-힣]{0,1}'),
    #     re.compile(r'\d{1,2}[.가-힣 ]{0,2}\d{1,2}[.가-힣]{0,1}')
    # ]
    # 년원일 추출 정규표현식 v2
    regex_ymd = [
        re.compile(r'\d{4}[.년 ]{1,2}\d{1,2}[.월 ]{1,2}\d{1,2}[.일]{0,1}'),
        re.compile(r'\d{4}[.년 ]{1,2}\d{1,2}[.월]{0,1}'),
        re.compile(r'\d{1,2}[.월 ]{1,2}\d{1,2}[.일]{0,1}'),
    ]
    regex_hm = [
        re.compile(r'\d{1,2}[:시 ]{1,2}\d{1,2}[:분]{0,1}'),
    ]

    # for file_path in file_pathes:
    #     with open(file_path, encoding="utf-8") as f:
    #         df = f.read().replace("\n", " ").replace("’", "`").replace("‘", "`").replace("∼", "~").replace(":", ":").replace("`", "20")
    for file_str in file_str_list:
        df = " ".join(file_str.lower().replace("\n", " ").replace("_x000d_", " ").replace("’", "`").replace("‘", "`").replace("∼", "~").replace(":", ":").replace("`", "20").split())

        if (df.strip() == '\ufeffSystem.String[]') or (df.strip() == ""):
            result.append(tuple([None, None]))
            continue

        for target_word in convert_dic.keys():
            for replaced_word in convert_dic[target_word]:
                df = df.replace(replaced_word, target_word)

        # 신청기간으로 split 한 뒤 10문자 내 숫자가 리스트 중인 부분 선택
        tmp = df.split("신청기간")

        # '신청기간'으로 split했는데 길이가 1인 경우, 즉 신청기간 키워드가 없는 경우 pass
        if len(tmp) == 1:
            result.append(tuple([None, None]))
            continue
        else:
            for i in tmp[1:]:
                # break
                findInt_reObj = re.search(r'[0-9]', "".join(i.split())[:findInt_maxLen])
                if findInt_reObj is not None:
                    df = i
                    break

        # 불필요한 문자열을 자르기 위해 max_len 만큼만 문자열 추출
        df = df[:max_len]
        # 물결 표시로 split후 앞의 두 원소 선택
        df_list = df.split('~')[:2]

        tmp_result = []
        for idx, value in enumerate(df_list):
            # # # for test
            # if idx == 1:
            #     break
            # break

            # 물결표시로 기간이 표시되어 있지 않아 df_list의 길이가 2보다 작으면 None을 먼저 원소로 삽입
            if len(df_list) < 2:
                tmp_result.append(None)

            # # for test
            # if idx == 1:
            #     break

            # 괄호 안 문자 제거 (요일 관련 정보를 제거하기 위함)
            tmp = re.sub(pattern=r'\([^)]*\)', repl='', string=value)
            # 한글, 숫자, ~, :, . 빼고 모두 제거 후 공백 한칸으로 제한
            tmp = " ".join(re.sub(pattern=r'[^0-9가-힣~.: ]', repl='', string=tmp).split())

            ymd = ""
            hm = ""
            findPoint_ymd = 0
            for i in regex_ymd:
                ymd = re.search(i, tmp.replace("년도", "년"))
                if ymd is not None:
                    findPoint_ymd = ymd.span()[1]
                    ymd = ymd.group(0)
                    ymd = ".".join(re.sub(r'[가-힣.]', " ", ymd).split())
                    break
                else:
                    ymd = ""

            # ymd 가 찾아지지 않았으면 기간이 나와있지 않은 것이라고 간주하고 2번째 리스트 검색
            if ymd == "":
                tmp_result.append(None)
                continue

            hm = re.search(regex_hm[-1], tmp[findPoint_ymd:])
            if hm is not None:
                hm = hm.group(0)
                hm = " " + ":".join(re.sub(r'[가-힣:]', " ", hm).split())
            else:
                hm = ""
            # 시분이 24시로 되어 있을 경우 pd.to_datetime 함수에서 에러 발생
            # 강제로 23:59 로 설정
            hm = " 23:59" if hm == ' 24:00' else hm

            # 월.일 형식일 경우 (앞에 년도가 없는 경우)
            if (len(ymd.split(".")) < 3) & (len(ymd.split(".")[0]) != 4):
                # 앞 리스트의 년도인데도 몇 년인지 정보가 없을 경우 기본값으로 2022 설정
                if idx == 0:
                    year = "2022"
                # 입력한 year와 기존 월 및 일 정보를 합하여 다시 저장
                ymd = year + "." + ymd
            # 년.월.일 형식의 경우
            else:
                # 물결 표시로 split한 리스트 중 첫번째 일 경우
                if idx == 0:
                    # 뒤의 리스트에 년도가 없을 경우를 대비하여 year라는 변수에 년도를 추출하여 할당
                    year = ".".join(ymd.split()).split(".")[0]

            tmp = ymd + hm
            try:
                tmp_result.append(pd.to_datetime(tmp, yearfirst=True))
            except:
                tmp_result.append(None)
        result.append(tuple(tmp_result))
    return result



# output = get_date_from_string(sorted(glob(folder_path + "*.txt")))
#
# for i in output:
#     print(i, "\n")
#
# pd.DataFrame(output)

# tmp_result[0] == pd.to_timedelta(0)

