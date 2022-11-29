import pandas as pd
from sklearn import linear_model
from sklearn.linear_model import Ridge  #Ridge

class apply_Ridge:
    def batter(datapath):
        print('\n===============BATTER=================')

        csv = pd.read_csv('.\dataset\Dataset_Batter_.csv')
        valcsv=pd.read_csv(datapath)

        data = csv[["연도","생년월일","AVG","AB","R","H","HR","RBI","BB","SO","OBP","OPS","RISP"]]
        label = csv["연봉(만원)"]

        valdata=valcsv[["연도","생년월일","AVG","AB","R","H","HR","RBI","BB","SO","OBP","OPS","RISP"]]

        #Ridge
        clfb=linear_model.Ridge(alpha=1.0)
        clfb.fit(data, label)
        result=clfb.predict(valdata)
        print('\nRidge')
        print(result)
        return result

    def pitcher(datapath):
        print('\n===================PITCHER=================')

        csvp = pd.read_csv('.\dataset\Dataset_Pitcher_.csv')
        valcsvp=pd.read_csv(datapath)

        datap = csvp[["연도","생년월일","ERA","G","W","L","SV","HLD","WPCT","H","HR","BB","HBP","SO","R","ER","WHIP"]]
        labelp = csvp["연봉(만원)"]

        valdatap=valcsvp[["연도","생년월일","ERA","G","W","L","SV","HLD","WPCT","H","HR","BB","HBP","SO","R","ER","WHIP"]]

        #Ridge
        clfp=linear_model.Ridge(alpha=1.0)
        clfp.fit(datap, labelp)
        result=clfp.predict(valdatap)
        print('\nRidge')
        print(result)
        return result