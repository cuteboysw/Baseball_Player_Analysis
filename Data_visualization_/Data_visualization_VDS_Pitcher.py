import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

df = pd.read_excel("Validation_Pitcher_excelVer_.xlsx")

dfS = df[["연봉"]]
dfERA = df[["ERA"]]
dfG = df[["G"]]
dfW = df[["W"]]
dfL = df[["L"]]
dfH = df[["H"]]
dfHR = df[["HR"]]
dfWHIP = df[["WHIP"]]
dfSO = df[["SO"]]
dfBB = df[["BB"]]
dfHBP = df[["HBP"]]
dfR = df[["R"]]

plt.scatter(dfERA, dfS, c = "blue", marker = "x")
plt.title("Pitcher's Validation Data Scatter plot (ERA, Salary)")
plt.xlabel('ERA')
plt.ylabel('Salary')
plt.show()
plt.scatter(dfG, dfS, c = "red", marker = "x")
plt.title("Pitcher's Validation Data Scatter plot (Games(G), Salary)")
plt.xlabel('G')
plt.ylabel('Salary')
plt.show()
plt.scatter(dfW, dfS, c = "green", marker = "x")
plt.title("Pitcher's Validation Data Scatter plot (Wins(W), Salary)")
plt.xlabel('W')
plt.ylabel('Salary')
plt.show()
plt.scatter(dfL, dfS, c = "purple", marker = "x")
plt.title("Pitcher's Validation Data Scatter plot (Loses(L), Salary)")
plt.xlabel('L')
plt.ylabel('Salary')
plt.show()
plt.scatter(dfH, dfS, c = "indigo", marker = "x")
plt.title("Pitcher's Validation Data Scatter plot (Hits(H), Salary)")
plt.xlabel('H')
plt.ylabel('Salary')
plt.show()
plt.scatter(dfHR, dfS, c = "black", marker = "x")
plt.title("Pitcher's Validation Data Scatter plot (Homeruns(HR), Salary)")
plt.xlabel('HR')
plt.ylabel('Salary')
plt.show()
plt.scatter(dfWHIP, dfS, c = "blue", marker = "x")
plt.title("Pitcher's Validation Data Scatter plot (WHIP, Salary)")
plt.xlabel('WHIP')
plt.ylabel('Salary')
plt.show()
plt.scatter(dfSO, dfS, c = "red", marker = "x")
plt.title("Pitcher's Validation Data Scatter plot (Strike outs(SO), Salary)")
plt.xlabel('SO')
plt.ylabel('Salary')
plt.show()
plt.scatter(dfBB, dfS, c = "green", marker = "x")
plt.title("Pitcher's Validation Data Scatter plot (Base on balls(BB), Salary)")
plt.xlabel('BB')
plt.ylabel('Salary')
plt.show()
plt.scatter(dfHBP, dfS, c = "purple", marker = "x")
plt.title("Pitcher's Validation Data Scatter plot (Hit by pitch balls(HBB), Salary)")
plt.xlabel('HBB')
plt.ylabel('Salary')
plt.show()
plt.scatter(dfR, dfS, c = "indigo", marker = "x")
plt.title("Pitcher's Validation Data Scatter plot (Runs(R), Salary)")
plt.xlabel('R')
plt.ylabel('Salary')
plt.show()
