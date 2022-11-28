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

#SGD
regression=apply_sgd
predict_batter=regression.batter()
predict_pitcher=regression.pitcher()
df=pd.DataFrame({'예상연봉': predict_batter}).to_csv('.\\result\\sgd_batter.csv')
df=pd.DataFrame({'예상연봉': predict_pitcher}).to_csv('.\\result\\sgd_pitcher.csv')

#LinearRegression
regression=apply_LinearRegression
regression.batter()
regression.pitcher()
df=pd.DataFrame({'예상연봉': predict_batter}).to_csv('.\\result\\LinearRegression_batter.csv')
df=pd.DataFrame({'예상연봉': predict_pitcher}).to_csv('.\\result\\LinearRegression_pitcher.csv')

#Lasso
regression=apply_Lasso
regression.batter()
regression.pitcher()
df=pd.DataFrame({'예상연봉': predict_batter}).to_csv('.\\result\\Lasso_batter.csv')
df=pd.DataFrame({'예상연봉': predict_pitcher}).to_csv('.\\result\\Lasso_pitcher.csv')

#Ridge
regression=apply_Ridge
regression.batter()
regression.pitcher()
df=pd.DataFrame({'예상연봉': predict_batter}).to_csv('.\\result\\Ridge_batter.csv')
df=pd.DataFrame({'예상연봉': predict_pitcher}).to_csv('.\\result\\Ridge_pitcher.csv')

#Logistic Regression
regression=apply_LogisticRegression
regression.batter()
regression.pitcher()
df=pd.DataFrame({'예상연봉': predict_batter}).to_csv('.\\result\\LogisticRegression_batter.csv')
df=pd.DataFrame({'예상연봉': predict_pitcher}).to_csv('.\\result\\LogisticRegression_pitcher.csv')

#k-neighbors_regression
regression=apply_k_neighbors_regression
regression.batter()
regression.pitcher()
df=pd.DataFrame({'예상연봉': predict_batter}).to_csv('.\\result\\k_neighbors_regression_egression_batter.csv')
df=pd.DataFrame({'예상연봉': predict_pitcher}).to_csv('.\\result\\k_neighbors_regression__pitcher.csv')

#Decision Tree
regression=apply_decisionTree
regression.batter()
regression.pitcher()
df=pd.DataFrame({'예상연봉': predict_batter}).to_csv('.\\result\\decisionTree_batter.csv')
df=pd.DataFrame({'예상연봉': predict_pitcher}).to_csv('.\\result\\decisionTree_pitcher.csv')

#plt.scatter(predict_bat, predict_bat, c="b", marker="x")