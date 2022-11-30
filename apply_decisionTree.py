import pandas as pd
from sklearn import tree

class apply_decisionTree:
    def batter(datapath):
        print('\n===============BATTER=================')

        csv = pd.read_csv('./dataset/Dataset_Batter_.csv')
        valcsv=pd.read_csv(datapath)

        data = csv[["연도","생년월일","AVG","AB","R","H","HR","RBI","BB","SO","OBP","OPS","RISP"]]
        label = csv["연봉(만원)"]

        valdata=valcsv[["연도","생년월일","AVG","AB","R","H","HR","RBI","BB","SO","OBP","OPS","RISP"]]

        #Decision Tree
        clf = tree.DecisionTreeRegressor()
        clf.fit(data,label)
        result = clf.predict(valdata)
        print('Decision Tree')
        print(result)
        return result

    def pitcher(datapath):
        print('\n===================PITCHER=================')

        csvp = pd.read_csv('./dataset/Dataset_Pitcher_.csv')
        valcsvp=pd.read_csv(datapath)

        datap = csvp[["연도","생년월일","ERA","G","W","L","SV","HLD","WPCT","H","HR","BB","HBP","SO","R","ER","WHIP"]]
        labelp = csvp["연봉(만원)"]

        valdatap=valcsvp[["연도","생년월일","ERA","G","W","L","SV","HLD","WPCT","H","HR","BB","HBP","SO","R","ER","WHIP"]]

        #Decision Tree
        clfp = tree.DecisionTreeRegressor()
        clfp.fit(datap,labelp)
        result = clfp.predict(valdatap)
        print('Decision Tree')
        print(result)
        return result
