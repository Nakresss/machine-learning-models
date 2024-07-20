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
from lightgbm import LGBMRegressor #eğer bununla iligili hata olursa conda install -c conda-forgelightgbm yazarsanız cmd ye düzelir

hit = pd.read_csv("hitters.csv")
df = hit.copy()
df = df.dropna()
dms = pd.get_dummies(df[['League','Division','NewLeague']])
y = df["Salary"]
x_ = df.drop(['Salary','League','Division','NewLeague'],axis=1).astype('float64')
x = pd.concat([X_, dms[["League_N","Division_W","NewLeague_N"]]], axis = 1)
X_train,X_test,y_train, y_test = train_test_split(X,y,test_size=0.25, random_state=42)

lgbm = LGBMRegressor()
lgbm_model = lgbm.fit(X_train,y_train)

lgbm_model.predict(X_test,num_iteration = lgbm_model.best_iteration_)
np.sqrt(mean_squared_error(y_test,y_pred))

lgbm_grid={'learning _rate':[0.01,0.1,0.5,1],'n_estimators':[20,40,100,200,500,1000],'max_depth':[1,2,3,4,5,6,7,8]}
lgbm = LGBMRegressor
lgbm_cv_model = GridSearchCV(lgbm,lgbm_grid, cv=10, n_jobs=-1, verbose=2)
lgbm_cv_model.fit(X_train,y_train)


















