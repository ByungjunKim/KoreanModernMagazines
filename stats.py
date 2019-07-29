# -*- coding: utf-8 -*-
"""
Created on Fri Feb 15 20:43:54 2019

@author: Instorial_Design
"""
###기술통계량###
##잡지 종류별 기사수
master.groupby(master['잡지종류']).size()

##잡지 종류별 unique 단어수

##잡지 종류별 본문 단어수(len)
len(master['본문'][0])
length_text = pd.concat([master['잡지종류'],master['본문'].map(lambda x:len(x))],axis=1)
length_text.groupby(length_text['잡지종류']).mean()

##연도별 기사수
master.groupby(master['year']).size()

##기사형태 종류
master.groupby(master['기사형태']).size()


#필자별 기사량
master.groupby(master['필자']).size().sort_values(ascending = False)
master.groupby(master['필자']).size().sort_values(ascending = False).mean() #평균 2.1건
master.groupby(master['필자']).size().sort_values(ascending = False).median()

master.groupby(master['필자']).size().sort_values(ascending = False).to_excel('필자별 기사량.xlsx')
