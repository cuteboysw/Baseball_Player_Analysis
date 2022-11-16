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

#SGD
reg = make_pipeline(StandardScaler(),
                    SGDRegressor(max_iter=1000, tol=1e-3))
reg.fit(data, label)
result=reg.predict(valdata)
print('\nSGD')
print(result)

#LinearRegression
lr = LinearRegression()
lr.fit(data,label)
result = lr.predict(valdata)
print('\nLinearRegression')
print(result)

#Lasso
clfbl = linear_model.Lasso(alpha=1.0)
clfbl.fit(data, label)
result = clfbl.predict(valdata)
print('\nLasso')
print(result)

#Ridge
clfbr=linear_model.Ridge(alpha=1.0)
clfbr.fit(data, label)
result=clfbr.predict(valdata)
print('\nRidge')
print(result)

print('\n===================PITCHER=================')

csvp = pd.read_csv('.\dataset\Dataset_Pitcher_.csv')
valcsvp=pd.read_csv('.\dataset\Vaildation_Pitcher_.csv')

datap = csvp[["연도","생년월일","ERA","G","W","L","SV","HLD","WPCT","H","HR","BB","HBP","SO","R","ER","WHIP"]]
labelp = csvp["연봉(만원)"]

valdatap=valcsvp[["연도","생년월일","ERA","G","W","L","SV","HLD","WPCT","H","HR","BB","HBP","SO","R","ER","WHIP"]]

#SGD
regp = make_pipeline(StandardScaler(),
                    SGDRegressor(max_iter=1000, tol=1e-3))
regp.fit(datap, labelp)
result=regp.predict(valdatap)
print('\nSGD')
print(result)

#LinearRegression
lrp = LinearRegression()
lrp.fit(datap,labelp)
result = lrp.predict(valdatap)
print('\nLinearRegression')
print(result)

#Lasso
clfpl = linear_model.Lasso(alpha=1.0)
clfpl.fit(datap, labelp)
result = clfpl.predict(valdatap)
print('\nLasso')
print(result)

#Ridge
clfpr=linear_model.Ridge(alpha=1.0)
clfpr.fit(datap, labelp)
result=clfpr.predict(valdatap)
print('\nRidge')
print(result)