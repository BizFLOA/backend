import copy
import dateutil.parser as dparser
import datefinder
import re
from glob import glob
import pandas as pd
from tqdm import tqdm

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
        "신청기간", "공고기간", "접수기간",
        "신청 기간", "공고 기간", "접수 기간",
        "본 접수기간", "신청서 제출기간", "온라인 신청 제출기간", "온라인신청 제출기간",
        "본 접수 기간", "신청서 제출 기간", "온라인 신청 제출 기간", "온라인신청 제출 기간",
    ],
    "지원금": ["지원금", "지원예산 및 규모", "금    액", "지원금액", "지원범위", "지원규모", "사업규모", "예산규모", "사업비규모", "지원예산", "수행규모",
            "지원 예산 및 규모", "금액", "지원 금액", "지원 범위", "지원 규모", "사업 규모", "예산 규모", "사업비 규모", "지원 예산", "수행 규모"]
}

# # INFO : 폴더 경로는 직접설정필요
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

folder_path = "C:\\정부사업공고\\Download_Files\\"
# folder_path = "C:\\Users\\take\\Desktop\\임시파일\\지원금 관련 test\\"
# folder_path = "C:\\Users\\take\\Desktop\\에러1\\"

# C:\Users\take\Desktop\임시파일\지원금 관련 test

def get_grant_from_string(file_str_list, max_len=40):

    # # for test
    # max_len = 40
    # file_pathes = sorted(glob(folder_path + "*.txt"))[:10]
    # file_pathes = sorted(glob(folder_path + "*.txt"))[4:5]
    # file_pathes = ["C:\\정부사업공고\\Download_Files\\GBIDX_6&SCHIDX_6&SCHTYPE_제목&ATMIDX_0&ATMTYPE_HWP&붙임1. 2022년 인공지능반도체 혁신기업 집중육성 사업 추가 공고문.txt"]

    result = []
    regex_grant = [
        re.compile(r'[0-9]+[.]?[0-9]+ ?억원?|[0-9]+[.]?[0-9]+ ?천만원?|[0-9]+[.]?[0-9]+ ?백만원?')
    ]

    # for file_path in file_pathes:
    #     print(file_path)
    #     with open(file_path, encoding="utf-8") as f:
    #         df = f.read().replace("\n", " ").replace("’", "`").replace("‘", "`").replace("∼", "~").replace(":", ":").replace("`", "20")
    for file_str in tqdm(file_str_list):
        df = " " .join(file_str.replace("\n", " ").replace("’", "`").replace("‘", "`").replace("∼", "~").replace(":", ":").replace("`", "20").split())

        if (df.strip() == '\ufeffSystem.String[]')or (df.strip() == ""):
            result.append(tuple([None] * 3))
            continue

        # 지원금 관련 키워드를 모두 '지원금' 단어로 통일
        for replaced_word in convert_dic["지원금"]:
            df = df.replace(replaced_word, "지원금")

        # 지원금 단어로 split
        split_str = df.split("지원금")
        tmp = None
        # '지원금'으로 split했는데 길이가 1인 경우, 즉 지원금 키워드가 없는 경우 pass
        if len(split_str) == 1:
            result.append(tuple([None] * 3))
            continue
        # 지원금으로 split한 부분 중 금액관련 단어가 있는 원소 선택
        else:
            for i in split_str[1:]:
                # break
                prep_str = i.replace("억 원", "억원").replace("천만 원", "천만원").replace("백만 원", "백만원")
                prep_str = " ".join(prep_str.split())[:max_len]
                findInt_reObj = re.search(regex_grant[-1], prep_str)
                if findInt_reObj is not None:
                    tmp = prep_str[:findInt_reObj.span()[1]]
                    break

        # 금액 및 금액 단위가 찾아진게 없으면 continue
        if tmp is None:
            result.append(tuple([None] * 3))
            continue
        else:
            df = tmp

        # 한글, 숫자, 영어, . 빼고 모두 제거 후 공백 한 칸으로 제한
        tmp = " ".join(re.sub(pattern=r'[^가-힣a-z0-9. ]', repl='', string=df.lower()).split())

        # 변수이름 변경 (별도 사유 없음)
        grant_rawString = copy.deepcopy(tmp)

        # prefix & suffix 추출
        prefix_list = []
        suffix_list = []
        # suffix
        # 분석용 re 클래스 변수 생성
        re_obj = re.search(regex_grant[-1], string=grant_rawString)
        # 1. '이내' 혹은 '최대' 라는 단어 찾기 (추후 suffix로 삽입할 예정)
        if "이내" in grant_rawString[re_obj.span()[1]:(re_obj.span()[1]+5)]:
            grant_rawString = grant_rawString.replace("이내", "")
            suffix_list.append("이내")
        elif "최대" in tmp[:re_obj.span()[1]]:
            grant_rawString = grant_rawString.replace("최대", "")
            suffix_list.append("이내")
        # 2. VAT 별도 또는 VAT 포함 단어 찾기
        tmp = re.search(r'vat ?[가-힣]+', grant_rawString)
        if tmp is not None:
            suffix_list.append(tmp.group(0).replace(" ", "").upper())

        # prefix
        # 분석용 re 클래스 변수 생성
        re_obj = re.search(regex_grant[-1], grant_rawString)

        # 금액 및 금액단위 앞에 있는 string 추출 (prefix 추출)
        rawPrefix = grant_rawString[:re_obj.span()[0]].strip()

        prefix_dic = {
            "총": ["총"],
            "연간": ["연간", "연 간"],
            "과제당": ["과제당", "과제 당"],
            "기업별": ["기업별"],
        }
        for i in prefix_dic.keys():
            for j in prefix_dic[i]:
                rawPrefix = rawPrefix.replace(j, i)
            if re.search(rf'{i}', rawPrefix) is not None:
                prefix_list.append(i)

        prefix = " ".join(prefix_list)
        suffix = " ".join(suffix_list)

        # # 괄호 안 문자 제거 (요일 관련 정보를 제거하기 위함)
        # tmp = re.sub(pattern=r'\([^)]*\)', repl='', string=df)
        grant_raw_amount_unit = "".join(re_obj.group(0).split())
        re_obj = re.search(r'[가-힣]', grant_raw_amount_unit)
        grant_rawAmount = grant_raw_amount_unit[:re_obj.span()[0]]
        grant_rawUnit = grant_raw_amount_unit[re_obj.span()[0]:]

        # 백만(million) 단위로 모두 통일
        money_unit_dic = {
            "억": ["억원", "억"],
            "천만": ["천만", "천만원"],
            "백만": ["백만", "백만원"],
        }

        if grant_rawUnit in money_unit_dic["억"]:
            grant_amount = str(round(float(grant_rawAmount) * 100, 2))

        elif grant_rawUnit in money_unit_dic["천만"]:
            grant_amount = str(round(float(grant_rawAmount) * 10, 2))
        else:
            grant_amount = grant_rawAmount
            if grant_rawUnit not in money_unit_dic["백만"]:
                print("INFO : 단위가 측정되지 않았습니다.")

        # 소수점 뒤가 모두 0이면 소수점 이하 모두 제거하여 정수로만 표시
        flag = True
        for i in grant_amount.split(".")[-1]:
            if i != "0":
                flag = False
                break
        if flag:
            grant_amount = grant_amount.split(".")[0]
        grant_unit = "백만원"

        # print(grant_rawString, "\n")
        #
        # print(rawPrefix)
        # print(grant_rawAmount)
        # print(grant_rawUnit, "\n")
        #
        # print(prefix)
        # print(grant_amount)
        # print(grant_unit)

        tmp_result = [prefix, grant_amount, grant_unit, suffix]

        # return 객체 내용 설명
        # 인덱스0번 : 금액 및 prefix, suffix 추출 전 문자열
        # 인덱스1번 : 추출한 정보 (prefix, 금액, 단위, suffix)
        # 인덱스2번 : 추출한 정보를 띄어쓰기로 join한 문자열
        result.append((grant_rawString, tuple(tmp_result), (" ".join(tmp_result)).strip()))
    return result


#
# output = get_grant_from_string(sorted(glob(folder_path + "*.txt")))
#
# for i in output:
#     print(i, "\n")
#
# for i in sorted(glob(folder_path + "*.txt"))[:10]:
#     print(i, "\n")






