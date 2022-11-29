import pandas as pd
from apply_sgd import *
from apply_LinearRegression import *
from apply_Lasso import *
from apply_Ridge import *
from apply_k_neighbors_regression import *
from apply_LogisticRegression import *
from apply_decisionTree import *
import matplotlib.pyplot as plt
import numpy as np
import os
from apply_pyplot import *
import pathlib

try:
    if not os.path.exists('.\\result'):
        os.makedirs('.\\result')
except OSError:
    print ('Error whlie making a directory for result')

datapath=input('타자에 대한 Test data가 있는 경로를 입력하세요.(상대 경로도 괜찮습니다.) 기본 경로(.\\dataset\\TestData_Batter_csv)로 하시려면 알파벳 소문자 n을 입력하세요.')
if(datapath=='n'):
    datapath='.\\dataset\\TestData_Batter_.csv'
bat_path=pathlib.Path(datapath)
datapath=input('타자에 대한 Test data가 있는 경로를 입력하세요.(상대 경로도 괜찮습니다.) 기본 경로(.\\dataset\\TestData_Pitcher_csv)로 하시려면 알파벳 소문자 n을 입력하세요.')
if(datapath=='n'):
    datapath='.\\dataset\\TestData_Pitcher_.csv'
pit_path=pathlib.Path(datapath)

batcsv=pd.read_csv(bat_path)
pitcsv=pd.read_csv(pit_path)

#SGD
regression=apply_sgd
predict_batter=regression.batter(bat_path)
predict_pitcher=regression.pitcher(pit_path)
df=pd.DataFrame({'예상연봉': predict_batter}).to_csv('.\\result\\sgd_batter.csv')
df=pd.DataFrame({'예상연봉': predict_pitcher}).to_csv('.\\result\\sgd_pitcher.csv')
predset_batter=predict_batter   #predict set of batter
predset_pitcher=predict_pitcher #predict set of pitcher

#LinearRegression
regression=apply_LinearRegression
regression.batter(bat_path)
regression.pitcher(pit_path)
df=pd.DataFrame({'예상연봉': predict_batter}).to_csv('.\\result\\LinearRegression_batter.csv')
df=pd.DataFrame({'예상연봉': predict_pitcher}).to_csv('.\\result\\LinearRegression_pitcher.csv')
predset_batter=np.concatenate((predset_batter, predict_batter), axis=0)
predset_pitcher=np.concatenate((predset_pitcher, predict_pitcher), axis=0)

#Lasso
regression=apply_Lasso
regression.batter(bat_path)
regression.pitcher(pit_path)
df=pd.DataFrame({'예상연봉': predict_batter}).to_csv('.\\result\\Lasso_batter.csv')
df=pd.DataFrame({'예상연봉': predict_pitcher}).to_csv('.\\result\\Lasso_pitcher.csv')
predset_batter=np.concatenate((predset_batter, predict_batter), axis=0)
predset_pitcher=np.concatenate((predset_pitcher, predict_pitcher), axis=0)

#Ridge
regression=apply_Ridge
regression.batter(bat_path)
regression.pitcher(pit_path)
df=pd.DataFrame({'예상연봉': predict_batter}).to_csv('.\\result\\Ridge_batter.csv')
df=pd.DataFrame({'예상연봉': predict_pitcher}).to_csv('.\\result\\Ridge_pitcher.csv')
predset_batter=np.concatenate((predset_batter, predict_batter), axis=0)
predset_pitcher=np.concatenate((predset_pitcher, predict_pitcher), axis=0)

#Logistic Regression
regression=apply_LogisticRegression
regression.batter(bat_path)
regression.pitcher(pit_path)
df=pd.DataFrame({'예상연봉': predict_batter}).to_csv('.\\result\\LogisticRegression_batter.csv')
df=pd.DataFrame({'예상연봉': predict_pitcher}).to_csv('.\\result\\LogisticRegression_pitcher.csv')
predset_batter=np.concatenate((predset_batter, predict_batter), axis=0)
predset_pitcher=np.concatenate((predset_pitcher, predict_pitcher), axis=0)

#k-neighbors_regression
regression=apply_k_neighbors_regression
regression.batter(bat_path)
regression.pitcher(pit_path)
df=pd.DataFrame({'예상연봉': predict_batter}).to_csv('.\\result\\k_neighbors_regression_batter.csv')
df=pd.DataFrame({'예상연봉': predict_pitcher}).to_csv('.\\result\\k_neighbors_regression_pitcher.csv')
predset_batter=np.concatenate((predset_batter, predict_batter), axis=0)
predset_pitcher=np.concatenate((predset_pitcher, predict_pitcher), axis=0)

#Decision Tree
regression=apply_decisionTree
regression.batter(bat_path)
regression.pitcher(pit_path)
df=pd.DataFrame({'예상연봉': predict_batter}).to_csv('.\\result\\decisionTree_batter.csv')
df=pd.DataFrame({'예상연봉': predict_pitcher}).to_csv('.\\result\\decisionTree_pitcher.csv')
predset_batter=np.concatenate((predset_batter, predict_batter), axis=0)
predset_pitcher=np.concatenate((predset_pitcher, predict_pitcher), axis=0)

#Get Result
alg_list=['SGD', 'Linear Regression', 'Lasso', 'Ridge', 'Logistic Regression', 'k-neighbors_regression', 'Decision Tree']   #Regression Algorithm List
predset_batter=predset_batter.reshape(-1, 7)
predset_pitcher=predset_pitcher.reshape(-1, 7)
try:
    if not os.path.exists('.\\error_rate'):
        os.makedirs('.\\error_rate')
except OSError:
    print ('Error whlie making a directory for result')
df=pd.DataFrame(predset_batter, columns=alg_list).to_csv('.\\result\\total_prediction_batter.csv')
df=pd.DataFrame(predset_pitcher, columns=alg_list).to_csv('.\\result\\total_prediction_pitcher.csv')
csvcnt=len(batcsv)
bat_index=np.arange(0, csvcnt)
apply_pyplot.run(bat_index, predset_batter.T, 'batter', alg_list)
csvcnt=len(pitcsv)
pit_index=np.arange(0, csvcnt)
apply_pyplot.run(pit_index, predset_pitcher.T, 'pitcher', alg_list)