import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

df = pd.read_excel("Dataset_Batter_final_excelVer_.xlsx")

dfS = df[["연봉"]]
dfAVG = df[["AVG"]]
dfH = df[["H"]]
dfHR = df[["HR"]]
dfAB = df[["AB"]]
dfRBI = df[["RBI"]]
dfBB = df[["BB"]]
dfSO = df[["SO"]]
dfOBP = df[["OBP"]]
dfOPS = df[["OPS"]]
dfRISP = df[["RISP"]]

plt.scatter(dfAVG, dfS, c = "blue", marker = "x")
plt.title("Batter's Study Data Scatter plot (AVG, Salary)")
plt.xlabel('AVG')
plt.ylabel('Salary')
plt.show()
plt.scatter(dfH, dfS, c = "red", marker = "x")
plt.title("Batter's Study Data Scatter plot (Hits(H), Salary)")
plt.xlabel('H')
plt.ylabel('Salary')
plt.show()
plt.scatter(dfHR, dfS, c = "green", marker = "x")
plt.title("Batter's Study Data Scatter plot (Homeruns(HR), Salary)")
plt.xlabel('HR')
plt.ylabel('Salary')
plt.show()
plt.scatter(dfAB, dfS, c = "purple", marker = "x")
plt.title("Batter's Study Data Scatter plot (At bats(AB), Salary)")
plt.xlabel('AB')
plt.ylabel('Salary')
plt.show()
plt.scatter(dfRBI, dfS, c = "indigo", marker = "x")
plt.title("Batter's Study Data Scatter plot (Runs batted in(RBI), Salary)")
plt.xlabel('RBI')
plt.ylabel('Salary')
plt.show()
plt.scatter(dfBB, dfS, c = "blue", marker = "x")
plt.title("Batter's Study Data Scatter plot (Base on balls(BB), Salary)")
plt.xlabel('BB')
plt.ylabel('Salary')
plt.show()
plt.scatter(dfSO, dfS, c = "red", marker = "x")
plt.title("Batter's Study Data Scatter plot (Strike outs(SO), Salary)")
plt.xlabel('SO')
plt.ylabel('Salary')
plt.show()
plt.scatter(dfOBP, dfS, c = "green", marker = "x")
plt.title("Batter's Study Data Scatter plot (On base percentage(OBP), Salary)")
plt.xlabel('ERA')
plt.ylabel('Salary')
plt.show()
plt.scatter(dfOPS, dfS, c = "purple", marker = "x")
plt.title("Batter's Study Data Scatter plot (On base plus slugging(OPS), Salary)")
plt.xlabel('OPS')
plt.ylabel('Salary')
plt.show()
plt.scatter(dfRISP, dfS, c = "indigo", marker = "x")
plt.title("Batter's Study Data Scatter plot (RISP, Salary)")
plt.xlabel('RISP')
plt.ylabel('Salary')
plt.show()