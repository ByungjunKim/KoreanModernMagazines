#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Apr 16 01:43:39 2019

@author: byungjunkim
"""

from sklearn.metrics.pairwise import cosine_similarity
import seaborn
import pandas as pd

import matplotlib as mpl
import matplotlib.pyplot as plt
import matplotlib.font_manager as fm


###1. 서북학회와 태극학보의 연관성은 어디서 기인하는가? 필진이 겹치는가?
master[master['잡지종류']=='서북학회월보']['필자'].unique()
master[master['잡지종류']=='태극학보']['필자'].unique()

#필진 export
with pd.ExcelWriter('authors_seobuk_taegeuk.xlsx',engine='openpyxl',mode='a') as writer:
    pd.DataFrame(master[master['잡지종류']=='서북학회월보']['필자'].unique()).to_excel(writer,index=None,sheet_name='Sheet1')
with pd.ExcelWriter('authors_seobuk_taegeuk.xlsx',engine='openpyxl',mode='a') as writer:
    pd.DataFrame(master[master['잡지종류']=='태극학보']['필자'].unique()).to_excel(writer,index=None,sheet_name='Sheet2') 
    
master[master['잡지종류']=='서북학회월보']['필자'].unique().to_excel()

###2.*내용의 유사도 역시 이 시기(1909.10.)부터 서북학회월보와 태극학보가 겹치는 것은 아닌가?
seobuk = master[master['잡지종류']=='서북학회월보']
seobuk.reset_index(drop=True,inplace=True)
seobuk['yyyymm'].unique() #19개월

taegeuk = master[master['잡지종류']=='태극학보']

tfidf_vector_taegeuk = vectorizer_top_tfidf.fit_transform(taegeuk['token_hanja_stopwords_str'])
dist_taegeuk = np.mean(tfidf_vector_taegeuk,axis=0)
tfidf_vector_seobuk_190910 = vectorizer_top_tfidf.fit_transform(seobuk.token_hanja_stopwords_str[seobuk['yyyymm'].isin([190910,190911,190912,191001])])
dist_seobuk_190910 = np.mean(tfidf_vector_seobuk_190910,axis=0)
cosine_similarity(dist_taegeuk,dist_seobuk_190910)

tfidf_vector_seobuk_190910_before = vectorizer_top_tfidf.fit_transform(seobuk.token_hanja_stopwords_str[~seobuk['yyyymm'].isin([190910,190911,190912,191001])])
dist_seobuk_190910_before = np.mean(tfidf_vector_seobuk_190910_before,axis=0)
cosine_similarity(dist_taegeuk,dist_seobuk_190910_before)




###3. ↳*그렇다면 서북학회월보 내에서 언어가 변하는 시기가 언제부터인가도 살펴볼 필요가 있습니다.
#tfidf_vector_seobuk = pipeline.fit_transform(seobuk['token_hanja_stopwords_str'])
tfidf_vector_seobuk = vectorizer_top_tfidf.fit_transform(seobuk['token_hanja_stopwords_str'])

seobuk_month = seobuk['yyyymm'].unique()

for month in seobuk_month:
    dist_seobuk_res =np.mean(tfidf_vector_seobuk[seobuk.index[seobuk['yyyymm']==month].tolist()],axis=0)
    if(month==seobuk_month[0]):
        dist_seobuk = dist_seobuk_res
    else:
        dist_seobuk = np.vstack([dist_seobuk,dist_seobuk_res])
        
cos_seobuk = pd.DataFrame(cosine_similarity(dist_seobuk))
cos_seobuk.columns = seobuk_month
cos_seobuk.index=seobuk_month

seaborn.heatmap(cos_seobuk)
