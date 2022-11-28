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
errrate_bat=(predict_batter-batcorrect)/predict_batter
errrate_pit=(predict_pitcher-pitcorrect)/predict_pitcher
print(errrate_pit)


#LinearRegression
regression=apply_LinearRegression
regression.batter()
regression.pitcher()
df=pd.DataFrame({'예상연봉': predict_batter}).to_csv('.\\result\\LinearRegression_batter.csv')
df=pd.DataFrame({'예상연봉': predict_pitcher}).to_csv('.\\result\\LinearRegression_pitcher.csv')
errrate_bat=np.concatenate((errrate_bat, (predict_batter-batcorrect)/predict_batter), axis=0)
errrate_pit=np.concatenate((errrate_pit, (predict_pitcher-pitcorrect)/predict_pitcher), axis=0)

#Lasso
regression=apply_Lasso
regression.batter()
regression.pitcher()
df=pd.DataFrame({'예상연봉': predict_batter}).to_csv('.\\result\\Lasso_batter.csv')
df=pd.DataFrame({'예상연봉': predict_pitcher}).to_csv('.\\result\\Lasso_pitcher.csv')
errrate_bat=np.concatenate((errrate_bat, (predict_batter-batcorrect)/predict_batter), axis=0)
errrate_pit=np.concatenate((errrate_pit, (predict_pitcher-pitcorrect)/predict_pitcher), axis=0)

#Ridge
regression=apply_Ridge
regression.batter()
regression.pitcher()
df=pd.DataFrame({'예상연봉': predict_batter}).to_csv('.\\result\\Ridge_batter.csv')
df=pd.DataFrame({'예상연봉': predict_pitcher}).to_csv('.\\result\\Ridge_pitcher.csv')
errrate_bat=np.concatenate((errrate_bat, (predict_batter-batcorrect)/predict_batter), axis=0)
errrate_pit=np.concatenate((errrate_pit, (predict_pitcher-pitcorrect)/predict_pitcher), axis=0)

#Logistic Regression
regression=apply_LogisticRegression
regression.batter()
regression.pitcher()
df=pd.DataFrame({'예상연봉': predict_batter}).to_csv('.\\result\\LogisticRegression_batter.csv')
df=pd.DataFrame({'예상연봉': predict_pitcher}).to_csv('.\\result\\LogisticRegression_pitcher.csv')
errrate_bat=np.concatenate((errrate_bat, (predict_batter-batcorrect)/predict_batter), axis=0)
errrate_pit=np.concatenate((errrate_pit, (predict_pitcher-pitcorrect)/predict_pitcher), axis=0)

#k-neighbors_regression
regression=apply_k_neighbors_regression
regression.batter()
regression.pitcher()
df=pd.DataFrame({'예상연봉': predict_batter}).to_csv('.\\result\\k_neighbors_regression_egression_batter.csv')
df=pd.DataFrame({'예상연봉': predict_pitcher}).to_csv('.\\result\\k_neighbors_regression__pitcher.csv')
errrate_bat=np.concatenate((errrate_bat, (predict_batter-batcorrect)/predict_batter), axis=0)
errrate_pit=np.concatenate((errrate_pit, (predict_pitcher-pitcorrect)/predict_pitcher), axis=0)

#Decision Tree
regression=apply_decisionTree
regression.batter()
regression.pitcher()
df=pd.DataFrame({'예상연봉': predict_batter}).to_csv('.\\result\\decisionTree_batter.csv')
df=pd.DataFrame({'예상연봉': predict_pitcher}).to_csv('.\\result\\decisionTree_pitcher.csv')
errrate_bat=np.concatenate((errrate_bat, (predict_batter-batcorrect)/predict_batter), axis=0)
errrate_pit=np.concatenate((errrate_pit, (predict_pitcher-pitcorrect)/predict_pitcher), axis=0)

print(errrate_bat)

#Error Rate
alg_list=['SGD', 'Linear Regression', 'Lasso', 'Ridge', 'Logistic Regression', 'k-neighbors_regression', 'Decision Tree']   #Regression Algorithm List
df=pd.DataFrame(errrate_bat.reshape(-1, 7), columns=alg_list).to_csv('.\\result\\error_rate_batter.csv')
df=pd.DataFrame(errrate_pit).to_csv('.\\result\\error_rate_pitcher.csv')

