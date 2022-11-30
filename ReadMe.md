# Baseball Player Analysis

## 1. How to Install

### 1.1. Project Clone

#### 1.1.1. Windows

https://github.com/cuteboysw/Baseball_Player_Analysis/archive/refs/heads/main.zip

Click this link.

#### 1.1.2. Linux Ubuntu

Type the commands.

```
apt-get install git
clone https://github.com/cuteboysw/Baseball_Player_Analysis/archive/refs/heads/main.zip
cd ./Baseball_Player_Analysis
git checkout ubuntu
```

### 1.2. Intall package needed
```
pip install pandas
pip install scikit-learn
pip install matplotlib
```

## 2. How to Use
* main.py: Main program for user
* main_validation.py: The program for developer

If you run this program using your own data, excute **main.py**.

```
python ./main.py
```

타자에 대한 Test data가 있는 경로를 입력하세요.(상대 경로도 괜찮습니다.) 기본 경로(./dataset/TestData_Batter_.csv)로 하시려면 알파벳 소문자 n을 입력하세요.

투수에 대한 Test data가 있는 경로를 입력하세요.(상대 경로도 괜찮습니다.) 기본 경로(./dataset/TestData_Pitcher_.csv)로 하시려면 알파벳 소문자 n을 입력하세요.

Then, the csv files of result is saved in **result** folder. You can watch total prediction bar graph with your screen(GUI). The graphs of each algorithm are saved in **./Data_visualization_/result_visual**.