import pandas as pd
from apply_sgd import *
from apply_LinearRegression import *
from apply_Lasso import *
from apply_Ridge import *
from apply_k_neighbors_regression import *
from apply_LogisticRegression import *
from apply_decisionTree import *

#SGD
regression=apply_sgd
predict_bat_sgd=regression.batter()
regression.pitcher()

#LinearRegression
regression=apply_LinearRegression
regression.batter()
regression.pitcher()

#Lasso
regression=apply_Lasso
regression.batter()
regression.pitcher()

#Ridge
regression=apply_Ridge
regression.batter()
regression.pitcher()

#Logistic Regression
regression=apply_LogisticRegression
regression.batter()
regression.pitcher()

#k-neighbors_regression
regression=apply_k_neighbors_regression
regression.batter()
regression.pitcher()

#Decision Tree
regression=apply_decisionTree
regression.batter()
regression.pitcher()