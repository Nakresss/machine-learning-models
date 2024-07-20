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

hit = pd.read_csv("hitters.csv")
df = hit.copy()
df = df.dropna()
dms = pd.get_dummies(df[['League','Division','NewLeague']])
y = df["Salary"]
x_ = df.drop(['Salary','League','Division','NewLeague'],axis=1).astype('float64')
x = pd.concat([X_, dms[["League_N","Division_W","NewLeague_N"]]], axis = 1)
X_train,X_test,y_train, y_test = train_test_split(X,y,test_size=0.25, random_state=42)

gbm_model = GradientBoostingRegressor()
gbm_model.fit(X_train,y_train)

y_pred = gbm_model.predict(X_test)
np.sqrt(mean_squared_error(y_test,y_pred))

gbm_params = {'learnin_rate':[0.001,0.01,0.1,0.2], 'max_depth':[3,5,8,50,100],'n_estimators':[200,500,1000,2000],'subsample':[1,0.5,0.75],}
gbm = GradientBoostingRegressor()
gbm_cv_model = GridSearchCV(gbm,gbm_params,cv=10,n_jobs = -1,verbose=2)
gbm_cv_model.fit(X_train,y_train)

gbm_tuned = {'learnin_rate':[0.1], 'max_depth':[5],'n_estimators':[200],'subsample':[0.5],}
gbm_tuned = gbm_tuned.fit(X_train,y_train)

y_pred = gbm_tuned.predict(X_test)
np.sqrt(mean_squared_error(y_test,y_pred))







