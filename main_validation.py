import pandas as pd
from apply_sgd import *
from apply_LinearRegression import *
from apply_Lasso import *
from apply_Ridge import *
from apply_k_neighbors_regression import *
from apply_LogisticRegression import *
from apply_decisionTree import *
import numpy as np
import os
from errrate_pyplot import *
import pathlib

try:
    if not os.path.exists('.\\result'):
        os.makedirs('.\\result')
except OSError:
    print ('Error whlie making a directory for result')

bat_path=pathlib.Path('.\dataset\Validation_Batter_.csv')
pit_path=pathlib.Path('.\dataset\Validation_Pitcher_.csv')

batcsv=pd.read_csv(bat_path)
batcorrect=batcsv["연봉(만원)"]
pitcsv=pd.read_csv(pit_path)
pitcorrect=pitcsv["연봉(만원)"]

#SGD
regression=apply_sgd
predict_batter=regression.batter(bat_path)
predict_pitcher=regression.pitcher(pit_path)
df=pd.DataFrame({'예상연봉': predict_batter}).to_csv('.\\result\\sgd_batter.csv')
df=pd.DataFrame({'예상연봉': predict_pitcher}).to_csv('.\\result\\sgd_pitcher.csv')
errrate_bat=(predict_batter-batcorrect)/predict_batter
errrate_pit=(predict_pitcher-pitcorrect)/predict_pitcher

#LinearRegression
regression=apply_LinearRegression
regression.batter(bat_path)
regression.pitcher(pit_path)
df=pd.DataFrame({'예상연봉': predict_batter}).to_csv('.\\result\\LinearRegression_batter.csv')
df=pd.DataFrame({'예상연봉': predict_pitcher}).to_csv('.\\result\\LinearRegression_pitcher.csv')
errrate_bat=np.concatenate((errrate_bat, (predict_batter-batcorrect)/predict_batter), axis=0)
errrate_pit=np.concatenate((errrate_pit, (predict_pitcher-pitcorrect)/predict_pitcher), axis=0)

#Lasso
regression=apply_Lasso
regression.batter(bat_path)
regression.pitcher(pit_path)
df=pd.DataFrame({'예상연봉': predict_batter}).to_csv('.\\result\\Lasso_batter.csv')
df=pd.DataFrame({'예상연봉': predict_pitcher}).to_csv('.\\result\\Lasso_pitcher.csv')
errrate_bat=np.concatenate((errrate_bat, (predict_batter-batcorrect)/predict_batter), axis=0)
errrate_pit=np.concatenate((errrate_pit, (predict_pitcher-pitcorrect)/predict_pitcher), axis=0)

#Ridge
regression=apply_Ridge
regression.batter(bat_path)
regression.pitcher(pit_path)
df=pd.DataFrame({'예상연봉': predict_batter}).to_csv('.\\result\\Ridge_batter.csv')
df=pd.DataFrame({'예상연봉': predict_pitcher}).to_csv('.\\result\\Ridge_pitcher.csv')
errrate_bat=np.concatenate((errrate_bat, (predict_batter-batcorrect)/predict_batter), axis=0)
errrate_pit=np.concatenate((errrate_pit, (predict_pitcher-pitcorrect)/predict_pitcher), axis=0)

#Logistic Regression
regression=apply_LogisticRegression
regression.batter(bat_path)
regression.pitcher(pit_path)
df=pd.DataFrame({'예상연봉': predict_batter}).to_csv('.\\result\\LogisticRegression_batter.csv')
df=pd.DataFrame({'예상연봉': predict_pitcher}).to_csv('.\\result\\LogisticRegression_pitcher.csv')
errrate_bat=np.concatenate((errrate_bat, (predict_batter-batcorrect)/predict_batter), axis=0)
errrate_pit=np.concatenate((errrate_pit, (predict_pitcher-pitcorrect)/predict_pitcher), axis=0)

#k-neighbors_regression
regression=apply_k_neighbors_regression
regression.batter(bat_path)
regression.pitcher(pit_path)
df=pd.DataFrame({'예상연봉': predict_batter}).to_csv('.\\result\\k_neighbors_regression_egression_batter.csv')
df=pd.DataFrame({'예상연봉': predict_pitcher}).to_csv('.\\result\\k_neighbors_regression__pitcher.csv')
errrate_bat=np.concatenate((errrate_bat, (predict_batter-batcorrect)/predict_batter), axis=0)
errrate_pit=np.concatenate((errrate_pit, (predict_pitcher-pitcorrect)/predict_pitcher), axis=0)

#Decision Tree
regression=apply_decisionTree
regression.batter(bat_path)
regression.pitcher(pit_path)
df=pd.DataFrame({'예상연봉': predict_batter}).to_csv('.\\result\\decisionTree_batter.csv')
df=pd.DataFrame({'예상연봉': predict_pitcher}).to_csv('.\\result\\decisionTree_pitcher.csv')
errrate_bat=np.concatenate((errrate_bat, (predict_batter-batcorrect)/predict_batter), axis=0)
errrate_pit=np.concatenate((errrate_pit, (predict_pitcher-pitcorrect)/predict_pitcher), axis=0)

#Error Rate
alg_list=['SGD', 'Linear Regression', 'Lasso', 'Ridge', 'Logistic Regression', 'k-neighbors_regression', 'Decision Tree']   #Regression Algorithm List
errrate_bat=errrate_bat.reshape(-1, 7)
errrate_pit=errrate_pit.reshape(-1, 7)
try:
    if not os.path.exists('.\\error_rate'):
        os.makedirs('.\\error_rate')
except OSError:
    print ('Error whlie making a directory for result')
df=pd.DataFrame(errrate_bat, columns=alg_list).to_csv('.\\error_rate\\error_rate_batter.csv')
df=pd.DataFrame(errrate_pit, columns=alg_list).to_csv('.\\error_rate\\error_rate_pitcher.csv')
csvcnt=len(batcsv)
bat_index=np.arange(0, csvcnt)
errrate_pyplot.run(bat_index, errrate_bat, 'batter')
csvcnt=len(pitcsv)
pit_index=np.arange(0, csvcnt)
errrate_pyplot.run(pit_index, errrate_pit, 'pitcher')