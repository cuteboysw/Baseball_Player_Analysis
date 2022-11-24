import pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn.neighbors import KNeighborsRegressor

print('\n===============BATTER=================')

#data_url='.\dataset\Dataset_Batter.csv'
csv = pd.read_csv('.\dataset\Dataset_Batter_.csv')
valcsv=pd.read_csv('.\dataset\Validation_Batter_.csv')

data = csv[["연도","생년월일","AVG","AB","R","H","HR","RBI","BB","SO","OBP","OPS","RISP"]]
label = csv["연봉(만원)"]

valdata=valcsv[["연도","생년월일","AVG","AB","R","H","HR","RBI","BB","SO","OBP","OPS","RISP"]]

#K Neighbors Regressor
clfb = KNeighborsRegressor(n_neighbors=2)
clfb.fit(data, label)
result = clfb.predict(valdata)
print('\K Neighbors Regressor')
print(result)

print('\n===================PITCHER=================')

csvp = pd.read_csv('.\dataset\Dataset_Pitcher_.csv')
valcsvp=pd.read_csv('.\dataset\Vaildation_Pitcher_.csv')

datap = csvp[["연도","생년월일","ERA","G","W","L","SV","HLD","WPCT","H","HR","BB","HBP","SO","R","ER","WHIP"]]
labelp = csvp["연봉(만원)"]

valdatap=valcsvp[["연도","생년월일","ERA","G","W","L","SV","HLD","WPCT","H","HR","BB","HBP","SO","R","ER","WHIP"]]

#K Neighbors Regressor
clfp = KNeighborsRegressor(n_neighbors=2)
clfp.fit(datap, labelp)
result = clfp.predict(valdatap)
print('\nLasso')
print(result)