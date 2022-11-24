import pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn import linear_model    #Lasso
from sklearn.linear_model import SGDRegressor   #SGD
from sklearn.pipeline import make_pipeline  #SGD
from sklearn.preprocessing import StandardScaler    #SGD
from sklearn.linear_model import Ridge  #Ridge

print('\n===============BATTER=================')

#data_url='.\dataset\Dataset_Batter.csv'
csv = pd.read_csv('.\dataset\Dataset_Batter_.csv')
valcsv=pd.read_csv('.\dataset\Validation_Batter_.csv')

data = csv[["연도","생년월일","AVG","AB","R","H","HR","RBI","BB","SO","OBP","OPS","RISP"]]
label = csv["연봉(만원)"]

valdata=valcsv[["연도","생년월일","AVG","AB","R","H","HR","RBI","BB","SO","OBP","OPS","RISP"]]

#LinearRegression
lr = LinearRegression()
lr.fit(data,label)
result = lr.predict(valdata)
print('\nLinearRegression')
print(result)
print('\n===================PITCHER=================')

csvp = pd.read_csv('.\dataset\Dataset_Pitcher_.csv')
valcsvp=pd.read_csv('.\dataset\Vaildation_Pitcher_.csv')

datap = csvp[["연도","생년월일","ERA","G","W","L","SV","HLD","WPCT","H","HR","BB","HBP","SO","R","ER","WHIP"]]
labelp = csvp["연봉(만원)"]

valdatap=valcsvp[["연도","생년월일","ERA","G","W","L","SV","HLD","WPCT","H","HR","BB","HBP","SO","R","ER","WHIP"]]

#LinearRegression
lrp = LinearRegression()
lrp.fit(datap,labelp)
result = lrp.predict(valdatap)
print('\nLinearRegression')
print(result)