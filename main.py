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
from errrate_pyplot import *

try:
    if not os.path.exists('.\\result'):
        os.makedirs('.\\result')
except OSError:
    print ('Error whlie making a directory for result')

batcsv=pd.read_csv('.\dataset\Validation_Batter_.csv')
batcorrect=batcsv["연봉(만원)"]
pitcsv=pd.read_csv('.\dataset\Validation_Pitcher_.csv')
pitcorrect=pitcsv["연봉(만원)"]

#SGD
regression=apply_sgd
predict_batter=regression.batter()
predict_pitcher=regression.pitcher()
df=pd.DataFrame({'예상연봉': predict_batter}).to_csv('.\\result\\sgd_batter.csv')
df=pd.DataFrame({'예상연봉': predict_pitcher}).to_csv('.\\result\\sgd_pitcher.csv')
predset_batter=predict_batter   #predict set of batter
predset_pitcher=predict_pitcher #predict set of pitcher

#LinearRegression
regression=apply_LinearRegression
regression.batter()
regression.pitcher()
df=pd.DataFrame({'예상연봉': predict_batter}).to_csv('.\\result\\LinearRegression_batter.csv')
df=pd.DataFrame({'예상연봉': predict_pitcher}).to_csv('.\\result\\LinearRegression_pitcher.csv')
predset_batter=np.concatenate((predset_batter, predict_batter), axis=0)
predset_pitcher=np.concatenate((predset_pitcher, predict_pitcher), axis=0)

#Lasso
regression=apply_Lasso
regression.batter()
regression.pitcher()
df=pd.DataFrame({'예상연봉': predict_batter}).to_csv('.\\result\\Lasso_batter.csv')
df=pd.DataFrame({'예상연봉': predict_pitcher}).to_csv('.\\result\\Lasso_pitcher.csv')
predset_batter=np.concatenate((predset_batter, predict_batter), axis=0)
predset_pitcher=np.concatenate((predset_pitcher, predict_pitcher), axis=0)

#Ridge
regression=apply_Ridge
regression.batter()
regression.pitcher()
df=pd.DataFrame({'예상연봉': predict_batter}).to_csv('.\\result\\Ridge_batter.csv')
df=pd.DataFrame({'예상연봉': predict_pitcher}).to_csv('.\\result\\Ridge_pitcher.csv')
predset_batter=np.concatenate((predset_batter, predict_batter), axis=0)
predset_pitcher=np.concatenate((predset_pitcher, predict_pitcher), axis=0)

#Logistic Regression
regression=apply_LogisticRegression
regression.batter()
regression.pitcher()
df=pd.DataFrame({'예상연봉': predict_batter}).to_csv('.\\result\\LogisticRegression_batter.csv')
df=pd.DataFrame({'예상연봉': predict_pitcher}).to_csv('.\\result\\LogisticRegression_pitcher.csv')
predset_batter=np.concatenate((predset_batter, predict_batter), axis=0)
predset_pitcher=np.concatenate((predset_pitcher, predict_pitcher), axis=0)

#k-neighbors_regression
regression=apply_k_neighbors_regression
regression.batter()
regression.pitcher()
df=pd.DataFrame({'예상연봉': predict_batter}).to_csv('.\\result\\k_neighbors_regression_egression_batter.csv')
df=pd.DataFrame({'예상연봉': predict_pitcher}).to_csv('.\\result\\k_neighbors_regression__pitcher.csv')
predset_batter=np.concatenate((predset_batter, predict_batter), axis=0)
predset_pitcher=np.concatenate((predset_pitcher, predict_pitcher), axis=0)

#Decision Tree
regression=apply_decisionTree
regression.batter()
regression.pitcher()
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
df=pd.DataFrame(predset_batter, columns=alg_list).to_csv('.\\error_rate\\error_rate_batter.csv')
df=pd.DataFrame(predset_pitcher, columns=alg_list).to_csv('.\\error_rate\\error_rate_pitcher.csv')
bat_index=np.arange(0, 38)
errrate_pyplot.run(bat_index, errrate_bat)