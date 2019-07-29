#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Mar 24 17:45:09 2019

@author: byungjunkim
"""

from sklearn.feature_extraction.text import TfidfTransformer
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.pipeline import Pipeline
from nltk.corpus import words
from nltk import FreqDist
import numpy as np
import pandas as pd
from sklearn.metrics.pairwise import cosine_similarity
import seaborn
import hanja

import matplotlib as mpl
import matplotlib.pyplot as plt
import matplotlib.font_manager as fm

plt.rc('font', family='Malgun Gothic')

#[(f.name, f.fname) for f in fm.fontManager.ttflist if 'Nanum' in f.name]
#plt.rcParams["font.family"] = 'NanumGothic.ttf'
#plt.rc('font', family='Nanum Gothic')




#tfidfv = TfidfVectorizer().fit(master['token_stopwords_str'])
#print(tfidfv.transform(master['token_stopwords_str']).toarray())
#print(tfidfv.vocabulary_)



vectorizer = CountVectorizer(analyzer='word',
                             lowercase=False,
                             tokenizer=None,
                             preprocessor=None,
                             min_df=2,
                             ngram_range=(1,1), #max_feature는?
                             )
vectorizer
pipeline = Pipeline([
    ('vect', vectorizer),
    ('tfidf', TfidfTransformer(smooth_idf = True)),
])  
pipeline

tfidf_vector = pipeline.fit_transform(master['token_hanja_stopwords_str'])

##TF 기준 단어 추출
tf_vector = vectorizer.fit_transform(master['token_hanja_stopwords_str'])
tf_scores = tf_vector.toarray().sum(axis=0)
tf_idx = np.argsort(-tf_scores)
tf_scores = tf_scores[tf_idx]
tf_vocab = np.array(vectorizer.get_feature_names())[tf_idx]

plt.bar(range(len(tf_scores)), tf_scores)
plt.show()
print(list(zip(vocab, tf_scores)))

##TF-IDF 기준 단어 추출
tfidf_scores = tfidf_vector.toarray().sum(axis=0)
idx = np.argsort(-tfidf_scores)
tfidf_scores = tfidf_scores[idx]
vocab = np.array(vectorizer.get_feature_names())[idx]
#dist=np.sum(tfidf_vector,axis=0)
#for tag, count in zip(vocab, dist):
#    print(count, tag)
#
#test= pd.DataFrame(dist, columns=vocab)
plt.bar(range(len(tfidf_scores)), tfidf_scores)
plt.show()
print(list(zip(vocab, tfidf_scores))[:100])

##TF, TF-IDF 단어 테이블 정리
list(zip(tf_vocab, tf_scores,vocab,tfidf_scores))[:100] #상위 100개
pd.DataFrame(list(zip(tf_vocab, tf_scores)),column)

tf_tfidf_vocab = pd.DataFrame(list(zip(tf_vocab, tf_scores,vocab,tfidf_scores)))
#terms_freq_magazines[magazine]['hangul'] = terms_freq_magazines[magazine]['terms'].map(lambda x:hanja.translate(x,'substitution'))

tf_tfidf_vocab.to_excel("tf_tfidf_vocab_min2.xlsx",index=None)

#상위 500개 -> 2000개
list(zip(vocab, tfidf_scores))[:500]
vocab[:500]
vectorizer_top = TfidfVectorizer(analyzer='word',
                             lowercase=False,
                             tokenizer=None,
                             preprocessor=None,
                             min_df=2,
                             ngram_range=(1,1), 
                             max_features=2000, #tf 기준 상위 N개
                             smooth_idf=True
                             )

vectorizer_top_tfidf = TfidfVectorizer(analyzer='word',
                             lowercase=False,
                             tokenizer=None,
                             preprocessor=None,
                             min_df=2,
                             ngram_range=(1,1), 
                             vocabulary=vocab[:2000], #tf-idf 기준 상위 N개
                             smooth_idf=True
                             )

vectorizer_2000 = CountVectorizer(analyzer='word',
                             lowercase=False,
                             tokenizer=None,
                             preprocessor=None,
                             min_df=2,
                             ngram_range=(1,1),
                             vocabulary=vocab[:2000]#max_feature는?
                             )

tfidf_top_vector = vectorizer_top_tfidf.fit_transform(master['token_hanja_stopwords_str'])


##TF, TF-IDF 단어 테이블 최종정리
tfidf_scores_2000 = tfidf_top_vector.toarray().sum(axis=0)
idx_2000 = np.argsort(-tfidf_scores_2000)
tfidf_scores_2000 = tfidf_scores_2000[idx_2000]
vocab_2000 = np.array(vectorizer_top_tfidf.get_feature_names())[idx_2000]

##TF 기준 단어 추출
tf_vector_2000 = vectorizer_2000.fit_transform(master['token_hanja_stopwords_str'])
tf_scores_2000 = tf_vector_2000.toarray().sum(axis=0)
tf_idx_2000 = np.argsort(-tf_scores_2000)
tf_scores_2000 = tf_scores_2000[tf_idx_2000]
tf_vocab_2000 = np.array(vectorizer_2000.get_feature_names())[tf_idx_2000]
##취합
tf_tfidf_vocab_2000 = pd.DataFrame(list(zip(tf_vocab_2000, tf_scores_2000,vocab_2000,tfidf_scores_2000)))

##서북, 태극 TFIDF, TF 수치 정리 및 Term-Term matrix
#서북
tfidf_vector_seobuk
tfidf_scores_seobuk = tfidf_vector_seobuk.toarray().sum(axis=0)
seobuk_idx = np.argsort(-tfidf_scores_seobuk)
tfidf_scores_seobuk = tfidf_scores_seobuk[seobuk_idx]
seobuk_vocab = np.array(vectorizer_top_tfidf.get_feature_names())[seobuk_idx]

tf_vector_seobuk = vectorizer_2000.fit_transform(seobuk['token_hanja_stopwords_str'])
tf_scores_seobuk = tf_vector_seobuk.toarray().sum(axis=0)
tf_idx_seobuk = np.argsort(-tf_scores_seobuk)
tf_scores_seobuk = tf_scores_seobuk[tf_idx_seobuk]
tf_vocab_seobuk = np.array(vectorizer_2000.get_feature_names())[tf_idx_seobuk]

tf_tfidf_vocab_seobuk = pd.DataFrame(list(zip(tf_vocab_seobuk, tf_scores_seobuk,seobuk_vocab,tfidf_scores_seobuk)))
tf_tfidf_vocab_seobuk.to_excel('.\\태극_서북 심층분석\\태극_서북_TF-IDF.xlsx',index=None)


#태극
tfidf_vector_taegeuk
tfidf_scores_taegeuk = tfidf_vector_taegeuk.toarray().sum(axis=0)
taegeuk_idx = np.argsort(-tfidf_scores_taegeuk)
tfidf_scores_taegeuk = tfidf_scores_taegeuk[taegeuk_idx]
taegeuk_vocab = np.array(vectorizer_top_tfidf.get_feature_names())[taegeuk_idx]

tf_vector_taegeuk = vectorizer_2000.fit_transform(taegeuk['token_hanja_stopwords_str'])
tf_scores_taegeuk = tf_vector_taegeuk.toarray().sum(axis=0)
tf_idx_taegeuk = np.argsort(-tf_scores_taegeuk)
tf_scores_taegeuk = tf_scores_taegeuk[tf_idx_taegeuk]
tf_vocab_taegeuk = np.array(vectorizer_2000.get_feature_names())[tf_idx_taegeuk]

tf_tfidf_vocab_taegeuk = pd.DataFrame(list(zip(tf_vocab_taegeuk, tf_scores_taegeuk,taegeuk_vocab,tfidf_scores_taegeuk)))
tf_tfidf_vocab_taegeuk.to_excel('.\\태극_서북 심층분석\\태극_서북_TF-IDF_.xlsx',index=None)




##TF-IDF Term-Term Matrix
tfidf_term_term_mat = cosine_similarity(tfidf_top_vector.T)
tfidf_term_term_mat = pd.DataFrame(tfidf_term_term_mat,index=vectorizer_top_tfidf.vocabulary_,
                                   columns=vectorizer_top_tfidf.vocabulary_)
tfidf_term_term_mat.to_csv('.\\matrix\\tfidf_term_term_mat.csv',encoding='UTF-8')

tfidf_term_term_mat_100 = tfidf_term_term_mat.iloc[:100,:100]
#hanja.translate(u'大韓民國은 民主共和國이다.', 'substitution')
tfidf_term_term_mat_100.to_csv('.\\matrix\\tfidf_term_term_mat_100.csv',encoding='UTF-8')

seobuk_term_term_mat = cosine_similarity(tfidf_vector_seobuk.T)
seobuk_term_term_mat = pd.DataFrame(seobuk_term_term_mat,index=vectorizer_top_tfidf.vocabulary_,
                                   columns=vectorizer_top_tfidf.vocabulary_)
taegeuk_term_term_mat = cosine_similarity(tfidf_vector_taegeuk.T)
taegeuk_term_term_mat = pd.DataFrame(taegeuk_term_term_mat,index=vectorizer_top_tfidf.vocabulary_,
                                   columns=vectorizer_top_tfidf.vocabulary_)

seobuk_term_term_mat.to_csv('.\\matrix\\seobuk_term_term_mat.csv',encoding='UTF-8')
seobuk_term_term_mat.iloc[:100,:100].to_csv('.\\matrix\\seobuk_term_term_mat_100.csv',encoding='UTF-8')
taegeuk_term_term_mat.to_csv('.\\matrix\\taegeuk_term_term_mat.csv',encoding='UTF-8')
taegeuk_term_term_mat.iloc[:100,:100].to_csv('.\\matrix\\taegeuk_term_term_mat_100.csv',encoding='UTF-8')

###잡지/연도별 문서 row 위치 확인###
magazines
years
master.index[master['잡지종류']==magazine].tolist()
tfidf_vector[master.index[master['잡지종류']==magazine].tolist()]
###각 잡지 문서별 feature 값(tfidf) 더한 후 문서수로 나눔(평균)
#dist_magazine_2 =np.mean(tfidf_vector[master.index[master['잡지종류']=='대한자강회월보'].tolist()],axis=0)

for magazine in magazines:
    dist_res =np.mean(tfidf_vector[master.index[master['잡지종류']==magazine].tolist()],axis=0)
    if(magazine==magazines[0]):
        dist_magazines = dist_res
    else:
        dist_magazines = np.vstack([dist_magazines,dist_res])
        
#TF-IDF 상위 N개로만 구성        
for magazine in magazines:
    dist_res =np.mean(tfidf_top_vector[master.index[master['잡지종류']==magazine].tolist()],axis=0)
    if(magazine==magazines[0]):
        dist_magazines_top = dist_res
    else:
        dist_magazines_top = np.vstack([dist_magazines_top,dist_res])

#for magazine in magazines:
#    dist_sum_res =np.sum(tfidf_vector[master.index[master['잡지종류']==magazine].tolist()],axis=0)
#    if(magazine==magazines[0]):
#        dist_sum_magazines = dist_sum_res
#    else:
#        dist_sum_magazines = np.vstack([dist_sum_magazines,dist_sum_res])

for year in years:
    dist_res =np.mean(tfidf_vector[master.index[master['year']==year].tolist()],axis=0)
    if(year==years[0]):
        dist_years = dist_res
    else:
        dist_years = np.vstack([dist_years,dist_res])
    #dist_years = np.concatenate((dist_years,dist_res),axis=0)


###코사인 유사도 계산###
cos_magazines = pd.DataFrame(cosine_similarity(dist_magazines))
cos_magazines.columns = magazines
cos_magazines.index=magazines

cos_magazines_top = pd.DataFrame(cosine_similarity(dist_magazines_top))
cos_magazines_top.columns = magazines
cos_magazines_top.index=magazines

cos_sum_magazines = pd.DataFrame(cosine_similarity(dist_sum_magazines))
cos_sum_magazines.columns = magazines
cos_sum_magazines.index=magazines

cos_years = pd.DataFrame(cosine_similarity(dist_years))
cos_years.columns = years
cos_years.index = years

cos_magazines.to_excel("코사인유사도.xlsx",sheet_name="잡지별")


###코사인 유사도 시각화###
seaborn.heatmap(cos_magazines)
seaborn.heatmap(cos_years)








###TfidfVectorizer
from sklearn.feature_extraction.text import TfidfVectorizer

tfidfv = TfidfVectorizer(min_df=2,ngram_range=(1,3)).fit(master['token_hanja_stopwords_str'])
print(tfidfv.transform(master['token_hanja_stopwords_str']).toarray())
print(tfidfv.vocabulary_)
