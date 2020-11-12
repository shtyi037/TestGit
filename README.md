# Machine Learning Introduction(機器學習簡介)

`目錄`
* [`Machine Learning concept(機器學習概念)`](#機器學習概念)
    * [四種常見的學習方法](#四種常見的學習方法)
        * [監督式學習]
unsupervised(非監督式學習)
semi-superviesed(半監督式學習)
reinforcement (強化學習)
* [`Applications in Machine Learning(機器學習中的應用)`](#機器學習中的應用)
* [`scikit-learn Introduction(scikit-learn簡介)`](#scikit-learn簡介)
* [`Different Competition Platform(不同的比賽平台)`](#不同的比賽平台)


# 機器學習概念

機器學習只是AI的其中一部份

![](image/Machine_Learning_001.jpg)


## Machine Learning knowledge(機器學習知識)

* artificial intelligence(人工智能)
* statistics(統計)
* computer science(計算機科學)


## 四種常見的學習方法
* supervised(監督式學習)
* unsupervised(非監督式學習)
* semi-superviesed(半監督式學習)
* reinforcement (強化學習)

有監督：事先已知類別。無監督：事先未知類別。
### Supervised Learning(監督式學習)
提供機器數據和相應的標籤(label)

![](image/Machine_Learning_002.jpg)

### unsupervised(非監督式學習)
群集未標記的數據

![](image/Machine_Learning_003.jpg)

### semi-superviesed(半監督式學習)
有些數據帶有標籤，有些則沒有
有監督+無監督學習

![](image/Machine_Learning_004.jpg)

### reinforcement (強化學習)
給定未標記的數據，使機器進行自學習

![](image/Machine_Learning_005.jpg)
 
## Classification (分類) V.S. Regression (回歸)

![](image/Machine_Learning_006.jpg)

Classification會比Regression容易些，
舉例股價預測Classification分辨漲或跌，而Regression需算出價格。
 
Regressionu也較常遇到房價預測

![](image/Machine_Learning_007.jpg)

一般流程：
蒐集的資料 -> 特徵工程 -> 機器學習模型 -> 預測出的結果

![](image/Machine_Learning_008.jpg)

## Feature engineer(特徵工程)
* 填寫缺失值
    * 數值
        * 刪除丟失值的數據
        * 將中位數/平均值填寫為缺失值
    * 分類
        * 填充模式為缺失類別
        * 將“其他”填寫為缺失類別
* 離群值檢測

更多參考
[Feature Engineering 特徵工程中常見的方法](https://vinta.ws/code/feature-engineering.html)


## 分割資料

* 將總房價數據分為兩部分
* 80％作為訓練數據
* 20％作為測試數據
* 訓練數據用於使機器學習
* 測試數據用於驗證機器是否學習得很好

# Data Normalization (數據歸一化)

* 嘗試將所有功能縮放為[0，1]或[-1，1]
* 一些常見的歸一化方法

![](image/Machine_Learning_009.jpg)


![](image/Machine_Learning_010.jpg)

## Select model

在本課程中我們將重點關注什麼?

![](image/Machine_Learning_011.jpg)

## validate trained model - regression (驗證經過訓練的模型-回歸)

* MSE(Mean-Square Error 均方誤差)
    * 誤差/偏差平方的平均值（估計量與估計量之間的差）
    * 值越小代表預測出來的值與實際的值很接近，代表模型越好。

* R squared (R平方 決定係數）
    * 可從自變量預測的因變量中方差的比例
    * R平方接近1表示模型更適合，有可能會是負號。

![](image/Machine_Learning_012.jpg)


## validate trained model – classification (驗證經過訓練的模型-分類)
* The most common way to measure of classification performance(衡量分類績效的最常用方法)
    * Note that please use testing data to evaluate(請注意，請使用測試數據進行評估)

假設100筆資料猜對了80筆，可以說準度為80%

![](image/Machine_Learning_013.jpg)

如果是分類的問題，除了準度以外，也可以拆分成更細部，分四個矩陣，正、負樣本預測成功多少，誤判的個數多少。

`補充:`
>正樣本和負樣本
簡單來說，和概率論中類似，一般我們看一個問題時，只關註一個事件（希望它發生或者成功，並對其進行分析計算），而正樣本就是屬於我們關注的這一類別的樣本，負樣本就是指不屬於該類別的樣本

![](image/Machine_Learning_014.jpg)

## accuracy paradox (準確性悖論)
假設我們有一個分類器來識別垃圾郵件，並顯示以下結果

![](image/Machine_Learning_015.jpg)


如果結果更改如下所示怎麼辦？

![](image/Machine_Learning_016.jpg)

>如果是進行分類的問題，不要只看準確度，應看更多的指標。



可以參考的指標如下圖:
![](image/Machine_Learning_017.jpg)

也可以參考
* F1分數
    * 精確度和召回率的諧波平均值
    * 最佳值為1（精確度和查全率），最差值為0

![](image/Machine_Learning_018.jpg)



* ROC curve (ROC曲線，接收器工作特性曲線）說明了二元分類器系統在不同閾值下的診斷能力
    * 通過繪製各種閾值設置下的真實陽性率（TPR）與陽性陽性率（FPR）來創建

![](image/Machine_Learning_020.jpg)


* AUC（曲線下面積）值是隨機選擇的正例的排名高於隨機選擇的負例的概率
    * 通常在[0.5，1]範圍內
    * AUC值越大表示分類器性能越好，越偏左上越好。

![](image/Machine_Learning_021.jpg)

![](image/Machine_Learning_022.jpg)

## Bias-Variance Tradeoff (偏差-偏差權衡)

* Low Variance = 低方差
* High Variance = 高差異
* Low Bias = 低偏見
* High Bias = 高偏見(整體很集中，但離紅心很遠)

左上圖 - 很集中，都離紅心很近

右上圖 - 較分散，不穩定，有時候離紅心較遠

左下圖 - 很集中但離紅心很遠

右下圖 - 不集中，離紅心遠。

![](image/Machine_Learning_023.jpg)

* X軸 = 模型的複雜度
* Y軸 = 錯誤率

一般模型會希望選在High Variance前一小段的區域。

![](image/Machine_Learning_024.jpg)

## No free lunch theory (沒有免費的午餐理論)

* No free lunch theory
* http://www.no-free-lunch.org/

簡言:沒有一個演算法可以通吃每個情況，也沒有一個演算法可以贏過所有演算法。

![](image/Machine_Learning_025.jpg)

# 機器學習中的應用

* auto vehicle (汽車自動駕駛)
* payment (人臉付款)
* medical treatment (藥物治療)
* robot advisor (機器人顧問)
* drone (無人機)
* precision marketing (精準行銷)
* smart factory (智能工廠)
* voice assistant (語音助手)

# scikit-learn簡介

## What’s Scikit-learn

* scikit-learn是用於python編程語言的機器學習庫
* http://scikit-learn.org/stable/
* https://github.com/scikit-learn/scikit-learn
* 支持許多著名的機器學習算法
* 分類，回歸，聚類
* 是一個Open Source

# Different Competition Platform(不同的比賽平台)

## What’s Kaggle?
* Kaggle是一個平台，統計學家和數據挖掘人員可以競爭該平台以生成最佳的預測模型
* https://www.kaggle.com/
* 數據集由公司和用戶上傳
* Google於2017年3月8日收購Kaggle


## What’s “天池”

* 天池是中文版的kaggle
* https://tianchi.aliyun.com/index.htm?spm=5176.100066.5610778.10.5198d780qaVmpq
* 阿里雲託管的數據平台

# pandas

```python
# use pandas to read csv file
import pandas as pd
df = pd.read_csv('RegularSeasonCompactResults.csv')

# print first five row
print(df.head())  # 顯示前面5筆資料

# print last five row
print(df.tail())  # 顯示最後五筆資料

# statistics on the dataframe
print(df.describe())  # 回傳描述性統計 

# print max value of each column
print(df.max())  # 回傳各欄位的最大值

# print Wscore that is greater than 150
print(df[df['Wscore'] > 150])  # 顯示Wscore大於150的資料

# drop rows and reset index
df_drop_row = df.drop(df.index[0])  # 刪除index為0的資料
df_reset_index1 = df_drop_row.reset_index(drop=True)  # 重新排列index
print(df_reset_index1.head())

# drop columns
df_drop_column = df.drop('Season', axis=1)  # 刪除整欄的名為Season資料
print(df_drop_column.head())
```

```python
df = pd.read_csv('RegularSeasonCompactResults.csv')

# select two origin column as new dataframe
df_new = df[['Season', 'Daynum']]  # 只取出'Season', 'Daynum'欄位的值
print(df_new.head())

# save dataframe to a csv file
df_new.to_csv('dfnew.csv', index=False)  # to_csv為儲存成csv檔，第一個參數為儲存的名稱與格式，index為False時則不需儲存。
#df_new.to_csv('dfnew.csv', index=False, header=False)

# apply some logical operation to manipulate data
df_daynum = df_new['Daynum'].apply(lambda x: 1 if x> 23 else 0)  # 將欄位Daynum值大於23的資料改為1，小於23則為0
print(df_daynum.head())
```

## Missing value

遇到缺失值時，處理方法兩種
* dropna 移除整列
* fillna 對資料進行填充，將該欄位填入平均值
```python
import pandas as pd

# load csv file
df = pd.read_csv('demo.csv')

print('origin dataframe')
print(df)

print('drop row that contain any missing value')
# drop row that contain any missing value
df_no_missing = df.dropna()  # 移除欄位有Nan的資料
print(df_no_missing)

print('fill missing value with mean')
# fill missing value with mean 
df["size"].fillna(df["size"].mean(), inplace=True)  # fillna為對資料進行填充，Nan的值填入該欄位的平均
print(df)
```

## Encoding categorical features

在進行分析時都是只有數字不會有英文，所以要將英文處理為數字。
* Categorical 將某的欄位進形編碼

```python
import pandas as pd

# load csv file
df = pd.read_csv('demo.csv')

print('origin dataframe')
print(df)

print('encode category')
df['house_type'] = pd.Categorical(df['house_type']).codes
print(df)
```

## Change dataframe into numpy arrray

dataframe與numpy arrray互相轉換的方式。

```python
import pandas as pd

# load csv file
df = pd.read_csv('demo.csv')

print('change dataframe to numpy array')
numpy_array = np.array(df)
print(numpy_array)

print('change numpy array to dataframe')
df_from_numpy = pd.DataFrame(numpy_array)
print(df_from_numpy)
```


## Split data

下列程式範例
* x = 0~19分成10組，每組2個值
* y = 0~9

test_size=0.3，3成的資料當testData，7成是trainData
`補充:`
>機器學習（machine learning）的相關研究中，經常會將數據集（dataset）分為訓練集（training set）跟測試集（testing set）這兩個子集，前者用以建立模型（model），後者則用來評估該模型對未知樣本進行預測時的精確度，正規的說法是泛化能力（generalization ability）

* 只有訓練集才可以用在模型的訓練過程中，測試集則必須在模型完成之後才被用來評估模型優劣的依據。
* 訓練集中樣本數量必須夠多，一般至少大於總樣本數的50%。
* 兩組子集必須從完整集合中均勻取樣。


```python
import numpy as np
from sklearn.model_selection import train_test_split

x, y = np.arange(20).reshape((10, 2)), np.arange(10)

print('before splitting......')

print("x: {}\n".format(x))
print("y: {}\n".format(y))

print("shape of x: {}".format(x.shape))
print("shape of y: {}\n".format(y.shape))

x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.3) # test_size=0.3，3成的資料當testData，7成是trainData
print('after splitting......')

print("x_train: {}\n".format(x_train))
print("x_test: {}\n".format(x_test))

print("y_train: {}\n".format(y_train))
print("y_test: {}\n".format(y_test))
```


## Preprocessing Data (預處理數據)
* Standardize data into zero mean and unit std (將數據標準化為零均值和標準房)

使用sklearn.preprocessing.StandardScaler類，使用該類的好處在於可以保存訓練集中的參數（均值、方差）直接使用其對象轉換測試集數據。

fit() 通常是做模型參數的優化，或是算統計變量，實際丟入的數據沒有變。
transform() 才是真正的做轉換。

```python
from sklearn import preprocessing
import numpy as np

x_train = np.array([[ 100., -1.,  2.],
                    [ 900.,  0.,  0.],
                    [ 200.,  1., -1.]])


print("mean of x_train: {}".format(x_train.mean(axis=0)))
print("std of x_train: {}\n".format(x_train.std(axis=0)))


scaler = preprocessing.StandardScaler().fit(x_train)

print("mean of x_scale: {}".format(scaler.mean_))
print("std of x_scale: {}\n".format(scaler.scale_))

# apply mean and std to standardize data
x_train = scaler.transform(x_train)

print("after standardiztion......")
print('x_train: {}'.format(x_train))


x_test = np.array([[-1., 1., 0.]])
print("apply same mean and std to new data(test data)\n")

x_test = scaler.transform(x_test)
print('x_test: {}'.format(x_test))
```

## Standardize data into a range (將數據標準化到範圍內)

* 與上方的Preprocessing Data (預處理數據)相同，只是將`StandardScaler`改成`MinMaxScaler`

```python
from sklearn import preprocessing
import numpy as np

x_train = np.array([[ 1., -1.,  2.],
                    [ 2.,  0.,  0.],
                    [ 0.,  1., -1.]])

scaler = preprocessing.MinMaxScaler().fit(x_train)
x_train = scaler.transform(x_train)

print("after standardiztion......")
print('x_train: {}'.format(x_train))


x_test = np.array([[ -3., -1.,  4.]])
print("apply same transformation to new data(test data)\n")

x_test = scaler.transform(x_test)
print('x_test: {}'.format(x_test))
```

## Evaluate Result (評估結果)

```python
from sklearn.metrics import mean_squared_error, r2_score, accuracy_score, confusion_matrix

y_test = [0, 1, 0 , 1, 0]
y_pred = [1, 0, 0 , 1, 0]

mse = mean_squared_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)
accuracy = accuracy_score(y_test, y_pred)
num_correct_samples = accuracy_score(y_test, y_pred, normalize=False)
con_matrix = confusion_matrix(y_test, y_pred)


print("Mean squared error: {}".format(mse))
print('r2 score: {}'.format(r2))
print('number of correct sample: {}'.format(num_correct_samples))
print('accuracy: {}'.format(accuracy))
print('confusion matrix: {}'.format(con_matrix))
```


# Regression Supervised Learning (回歸監督學習)

* Linear Regression (線性回歸)
    * 最基礎的一個Regression的一個方法，也是最常見的方法。
* Polynomial Regression (多項式回歸)
* Logistic Regression (邏輯回歸)

## Linear Regression (線性回歸)

什麼是線性回歸?
* 一種線性方法，用於建模標量因變量y和一個或多個自變量X之間的關係。

下圖的紅色線為藍色點的趨勢線。


![](image/Machine_Learning_026.jpg)

`補充:`
>趨勢線是用圖形的方式顯示資料的趨勢，可以用它來預測分析。這種分析也稱為迴歸分析(Regression Analysis)。利用迴歸分析，可以在圖形中延伸趨勢線，根據現在實際已獲取的資料來預測未來資料。例如，利用趨勢線來預測系統瓶頸或者資源使用量。


如何預測房價？

![](image/Machine_Learning_027.jpg)

發現特徵數量級不一樣時，需要做Data normalization(數據標準化)

簡單回歸案例:
* 假設所有（𝑥𝑖，𝑦𝑖）對如下
    * 𝑥𝑖是第i個數據上的房屋年齡
    * 𝑦𝑖是第i個數據上的房價

![](image/Machine_Learning_028.jpg)

* 我們想找到最適合這些數據的線
    * 著名的解決方案之一是線性回歸

![](image/Machine_Learning_029.jpg)

* `最合適`是指實際y值和預測y值之間的差最小
    * 我們通常使用最小二乘，以最小化平方差之和

![](image/Machine_Learning_030.jpg)


* 我們可以使用微積分找到最合適的線

![](image/Machine_Learning_031.jpg)

### 數據標準化的重要性

左圖為沒有數據標準化，右圖為數據歸一化
![](image/Machine_Learning_032.jpg)

### 多元線性回歸模型

![](image/Machine_Learning_033.jpg)


![](image/Machine_Learning_034.jpg)

* 查找函數最小值的算法

𝜂 = 學習率
![](image/Machine_Learning_035.jpg)

![](image/Machine_Learning_036.jpg)
## Polynomial Regression (多項式回歸)


* Gradient Descent
    * 需要選擇學習率
    * 需要多次迭代
    * 當n大時效果很好

* Normal Equation
    * 無需選擇學習率
    * 需要計算矩陣逆
    * 如果n大，則非常慢

### Overfitting

![](image/Machine_Learning_037.jpg)
變量太多會導致過度擬合

### Example and Practice
* Example
    * 線性回歸
    * 示例/回歸
* Practice
    * 嘗試使用線性回歸來預測房價
        * 數據集/house.csv
        * 實踐/回歸
    * 有關數據集的更多信息
        * [Boston Housing](https://www.kaggle.com/c/boston-housing)


## Logistic Regression (邏輯回歸)




# 不同的比賽平台

![Uploading file..._so7m1tcw9]()
