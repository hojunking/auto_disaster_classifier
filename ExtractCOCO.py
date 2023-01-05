import json
import os

fn1 = "captions_train2017.json"
fn2 = "captions_val2017.json"
fn3 = "captions_train2014.json"
fn4 = "captions_val2014.json"
fn5 = "keywords.json"

#경로설정
dir = "/Users/hojun/Dataset/disaster_research/coco"

file_name = fn1
path = dir + "/" + file_name #annotation 파일 경로
with open(path, 'r') as f1:
    json_data = json.load(f1)

path = dir + "/" + fn5
with open(path, 'r') as f2:
    keywords = json.load(f2)

print("dataset: ", file_name)

cap_list = list(m['caption'] for m in json_data['annotations']) #딕셔너리로 이루어진 리스트에서 특정 키의 value만 추출해서 새로운 리스트로 만들기
#print(cap_list)
#file_name = list(m['file_name'])

image_list = list(m['image_id'] for m in json_data['annotations'])
#print(image_list)
image_cnt = list(dict.fromkeys(image_list))
print("num of data: ", len(image_cnt))

# #키워드 목록
# search = ['wild fire', 'fireman', 'firefighter', 'forest fire', 'prairie fire', 'on fire', 'set fire to']

# #리스트 내 특정 문자열(여러개 중에 하나라도)을 포함하는 이미지ID 출력 + 중복제거!
# resultlist = []
# for i in range(len(cap_list)):
#     if any(keyword in cap_list[i].lower() for keyword in search):
#         resultlist.append(image_list[i])
# resultlist = list(dict.fromkeys(resultlist))
# print("재난 키워드: ", len(resultlist))

sum = 0
exclude_words = ['floodlight','flood light','fireplace', 'hydrant','sign','after', 'like a', 'thailand', 'surfboard','bathroom', 'storm']
for i in range(len(keywords)):
    #키워드를 label code로 변환
    search = keywords[i]['keywords']
    resultlist = []

    #위 키워드가 얼마나 포함되었는지
    for k in range(len(cap_list)):
        if any(keyword in cap_list[k].lower() for keyword in search)and not any(keyword in cap_list[k].lower() for keyword in exclude_words):
            resultlist.append(image_list[k])
    
    resultlist = list(dict.fromkeys(resultlist)) #중복제거
    print(keywords[i]['type'],":", len(resultlist))
    sum += len(resultlist)
    if resultlist:
        print(resultlist)

print("total: ", sum)

f1.close()
f2.close()