import pandas as pd
from sklearn import svm
from sklearn.linear_model import LinearRegression
from sklearn import linear_model    #Lasso

#data_url='.\dataset\Dataset_Batter.csv'

csv = pd.read_csv('.\dataset\Dataset_Batter_.csv')
#print(csv)
testcsv=pd.read_csv('.\dataset\Test_Batter_.csv')

data2 = csv[["연도","생년월일","AVG","G","PA","AB","R","H","2B","3B","HR","TB","RBI","SAC","SF","BB","IBB","HBP","SO","GDP","SLG","OBP","OPS"]]
label2 = csv["연봉(만원)"]

testdata=testcsv[["연도","생년월일","AVG","G","PA","AB","R","H","2B","3B","HR","TB","RBI","SAC","SF","BB","IBB","HBP","SO","GDP","SLG","OBP","OPS"]]

lr = LinearRegression()
lr.fit(data2,label2)
result = lr.predict(testdata)
print('LinearRegression')
print(result)

clf = linear_model.Lasso(alpha=1.0)
clf.fit(data2, label2)
result = clf.predict(testdata)
print('Lasso')
print(result)

#"이름",,"생년월일"