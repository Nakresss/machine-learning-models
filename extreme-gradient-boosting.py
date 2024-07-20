import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split,ShuffleSplit,GridSearchCV
from sklearn.metrics import mean_squared_error,r2_score
import matplotlib.pyplot as plt
from sklearn.preprocessing import scale
from sklearn import model_selection
from sklearn.tree import DecisionTreeClassifier, DecisionTreeRegressor
from sklearn.neighbors import KNeighborsRegressor
from sklearn.ensemble import BaggingRegressor
from sklearn.ensemble import GradientBoostingRegressor
import xgboost as xgb
from xgboost import XGBRegressor

hit = pd.read_csv("hitters.csv")
df = hit.copy()
df = df.dropna()
dms = pd.get_dummies(df[['League','Division','NewLeague']])
y = df["Salary"]
x_ = df.drop(['Salary','League','Division','NewLeague'],axis=1).astype('float64')
x = pd.concat([X_, dms[["League_N","Division_W","NewLeague_N"]]], axis = 1)
X_train,X_test,y_train, y_test = train_test_split(X,y,test_size=0.25, random_state=42)

DM_train = xgb.DMatrix(data = X_train, label = y_train)
DM_test = xgb.DMatrix(data = X_test, label = y_test)

xgb_model = XGBRegressor().fit(X_train,y_train)
y_pred = xgb_model.predict(X_test)
np.sqrt(mean_squared_error(y_test,y_pred))

xgb_grid = {'colsample_bytree':[0.4,0.5,0.6,0.9,1],'n_estimators':[100,200,500,1000],'max_depth':[2,3,4,5,6],'learning_rate':[0.1,0.01,0.5]}
 
xgb = XGBRegressor()
xgb = GridSearchCV(xgb,param_grid=xgb_grid,cv=10,n_jobs=-1,verbose=2)

xgb_cv.fit(X_train,y_train)
xgb_cv.best_params_

y_pred = xgb_tuned.predict(X_test)
np.sqrt(mean_squared_error(y_test,y_pred))




