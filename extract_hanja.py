# -*- coding: utf-8 -*-
"""
Created on Tue Jan 29 13:50:29 2019

@author: Instorial_Design
"""

import pandas as pd
import regex as re

#master = pd.read_csv("master.csv")
#master = pd.read_csv("master2.csv")
master = pd.read_csv("master_1910_V3.csv")
#master=pd.read_csv("test.csv",header=None)
hanja = re.compile('[^\p{Han}]+')
hangul = re.compile('[^\p{Hangul}]{2,}')

#test = master['본문'][0]
##test = master[5][0]
##hanja.sub(' ',test)
#hanja.sub(',',test)
#hangul.sub(' ',test)


master = master.dropna(subset=['본문'])  ## 본문 없는 기사 삭제
##기사형태 및 잡지종류 제외
# 제외 기사형태 : '기행문','문예기타','소설','시'
master['기사형태'].unique()
master = master[~master['기사형태'].isin(['기행문','문예기타','소설','시'])]

#제외 잡지종류 : 대조선독립협회회보
master = master[~master['잡지종류'].isin(['대조선독립협회회보'])]

master.reset_index(drop=True, inplace=True)

#YYYYMM 추가(zfill 활용)
str(1).zfill(2)
master['yyyymm'] = master.apply(lambda x:int(str(x['year'])+str(x['month']).zfill(2)),axis=1)

##본문에 제목과 필자 부분 띄어쓰기 처리 -> master2로 해결
#re.sub(master['기사제목'][1],'\g<0> ',master['본문'][1],1)
#re.sub(master['필자'][2],'\g<0> ',master['본문'][2],1)
#def make_space(title_author,contents):
#    result = re.sub(title_author,'\g<0> ',contents,1)
#    return result
#master['본문'] = master.apply(lambda x:make_space(x['기사제목'],x['본문']),axis=1)
#master.apply(lambda x:re.sub(x['기사제목'],'\g<0> ',x['본문'],1),axis=1)

        
###본문 전처리 작업(분석대상 선별)###
#1.본문 한자 모두 포함
#2. 2음절 이상 한글 포함? -> 기각
#3. 한글 색인어 포함?
#* 색인어는 결과 분석 참고 자료로 활용. 색인어 자체를 형태소 분석 대상으로 삼지는 않음.

##한자 extract
#master['본문_한자'] = master['본문'].map(lambda x : hanja.sub(' ',x))
master['본문_한자'] = master['본문'].map(lambda x : hanja.sub(',',x))
##맨끝 컴마 제거
re.sub('\,$','',master['본문_한자'][0])
master['본문_한자'] = master['본문_한자'].map(lambda x:re.sub('\,$','',x))
master['본문_한자'] = master['본문_한자'].map(lambda x:re.sub('\s','',x))



#for i in range(len(master))
#    master["본문_한자"][i] = hanja.sub(' ',master["본문"][i]) #한자만 남기기
#    if(i%10000==0):
#        print(i)

##unique 색인어 처리
master['색인어_전부'].unique() #전체 고유 색인어

###str -> list, unique 색인어 list###
#test2 = '한틩돈,영국 미국,英國'
#test2.split(',')
master['색인어_unique_list'] = master['색인어_전부'].dropna().map(lambda x:pd.unique(x.split(',')))

master['색인어_list'] = master['색인어_전부'].dropna().map(lambda x:x.split(','))

##한글포함 색인어 extract
master['색인어_list'][2]
hangul2 = re.compile(r'\p{Hangul}+')
#hangul_filter = filter(hangul2.search, master['색인어_list'][2])

master['색인어_list_한글'] = master['색인어_list'].dropna().map(lambda x:list(filter(hangul2.search,x))).dropna()



###Export ###
#master.to_csv('master_1910_v5_t.csv',index=None,sep="\t")

master.to_excel('master_1910_v6.xlsx',index=None)
master[['잡지종류','잡지명','발행일','기사제목','필자','기사형태','본문','본문_한자','url']].to_excel('master_1910_v6_대조작업용.xlsx')
