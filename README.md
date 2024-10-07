# BLENDED_LEARNING
# Implementation-of-Linear-Regression-for-Predicting-Car-Prices
## AIM:
To write a program to predict car prices using a linear regression model and test the assumptions for linear regression.

## Equipments Required:
1. Hardware – PCs
2. Anaconda – Python 3.7 Installation / Jupyter notebook

## Algorithm
1. Collect and import the dataset (features and car prices).
2. Preprocess data by handling missing values and scaling features.
3. Split data into training and testing sets.
4. Select relevant features for prediction.
5. Fit a linear regression model using the training data.
6. Evaluate model performance using metrics like RMSE on test data.
7. Adjust the model (if needed) by tuning hyperparameters or feature selection.
8. Predict car prices using the trained model.
## Program:
```
/*
 Program to implement linear regression model for predicting car prices and test assumptions.
Developed by: Priyadharshan S
RegisterNumber:  212223240127


from tqdm import tqdm

import numpy as np
import pandas as pd
from itertools import accumulate
import matplotlib.pyplot as plt
import seaborn as sns
%matplotlib inline
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.datasets import load_digits, load_wine
from scipy.stats import boxcox
from scipy.stats.mstats import normaltest
from sklearn.metrics import r2_score 

from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score 
from sklearn.metrics import mean_squared_error
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder
from sklearn.compose import ColumnTransformer

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

from pyodide.http import pyfetch
 
async def download(url, filename):
    response = await pyfetch(url)
    if response.status == 200:
        with open(filename, "wb") as f:
            f.write(await response.bytes())
 
path = "https://cf-courses-data.s3.us.cloud-object-storage.appdomain.cloud/IBM-ML240EN-SkillsNetwork/labs/data/CarPrice_Assignment.csv"
 
await download(path, "CarPrice_Assignment.csv")

import pandas as pd 
data = pd.read_csv("CarPrice_Assignment.csv") 
data.head(5)

data['brand'] = data.CarName.str.split(' ').str.get(0).str.lower()

data['brand'] = data['brand'].replace(['vw', 'vokswagen'], 'volkswagen')
data['brand'] = data['brand'].replace(['maxda'], 'mazda')
data['brand'] = data['brand'].replace(['porcshce'], 'porsche')
data['brand'] = data['brand'].replace(['toyouta'], 'toyota')

data.drop(['car_ID', 'symboling', 'CarName'],axis = 1, inplace = True)

data_comp_avg_price = data[['brand','price']].groupby('brand', as_index = False).mean().rename(columns={'price':'brand_avg_price'})


data = data.merge(data_comp_avg_price, on = 'brand')

data['brand_category'] = data['brand_avg_price'].apply(lambda x : "Budget" if x < 10000 
                                                     else ("Mid_Range" if 10000 <= x < 20000
                                                           else "Luxury"))

columns=['price', 'fueltype', 'aspiration','carbody', 'drivewheel','wheelbase', 'brand_category',
                  'curbweight', 'enginetype', 'cylindernumber', 'enginesize', 'boreratio','horsepower', 'carlength','carwidth','citympg','highwaympg']



selected = data[columns]
selected.info()

categorical_columns = [col for col in selected.columns if selected[col].dtype == 'object']  


numeric_columns=list(set(columns)-set(categorical_columns))
X = selected.drop("price", axis=1)
y = selected["price"].copy()


one_hot = ColumnTransformer(transformers=[("one_hot", OneHotEncoder(), categorical_columns) ],remainder="passthrough")
X=one_hot.fit_transform(X)

names=one_hot.get_feature_names_out()


colunm_names=[name[name.find("_")+1:] for name in  [name[name.find("__")+2:] for name in names]]


df=pd.DataFrame(data=X,columns=colunm_names)


X_train, X_test, y_train, y_test = train_test_split( df, y, test_size=0.30, random_state=0)

ss=StandardScaler()
ss

X_train=ss.fit_transform(X_train)

lm = LinearRegression()
lm.fit(X_train,y_train)

X_test=ss.transform(X_test)
car_price_predictions = lm.predict(X_test)
car_price_predictions

mse = mean_squared_error(y_test, car_price_predictions)
mse

r2_score(y_test,car_price_predictions)
*/
```

## Output:
![simple linear regression model for predicting the marks scored](sam.png)


## Result:
Thus, the program to implement a linear regression model for predicting car prices is written and verified using Python programming, along with the testing of key assumptions for linear regression.
