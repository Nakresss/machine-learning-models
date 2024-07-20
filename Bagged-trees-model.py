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
from sklearn.svm import SVR
from sklearn.preprocessing import StandardScaler
from sklearn.neural_network import MLPRegressor

hit = pd.read_csv("hitters.csv")
df = hit.copy()
df = df.dropna()
dms = pd.get_dummies(df[['League','Division','NewLeague']])
y = df["Salary"]
x_ = df.drop(['Salary','League','Division','NewLeague'],axis=1).astype('float64')
x = pd.concat([X_, dms[["League_N","Division_W","NewLeague_N"]]], axis = 1)
X_train,X_test,y_train, y_test = train_test_split(X,y,test_size=0.25, random_state=42)


bag_model = BaggingRegressor(bootstrap_features=True)
bag_model.fit(X_train,y_train)
bag_model.n_estimators
bag_model.estimator_
bag_model.estimators_samples_

y_pred = bag_model.predict(X_test)
np.sqrt(mean_squared_error(y_test,y_pred))

iki_y_pred = bag_model.estimators_[1].fit(X_train, y_train).predict(X_test)
np.sqrt(mean_squared_error(y_test,y_pred))

dort_y_pred = bag_model.estimators_[4].fit(X_train, y_train).predict(X_test)
np.sqrt(mean_squared_error(y_test,y_pred))

bag_params = {"n_estimators": range(2,20)}
bag_cv_model = GridSearchCV(bag_model,bag_params,cv=10)
bag_cv_model.fit(X_train,y_train)
bag_cv_model.best_params_


bag_tuned = BaggingRegressor(n_estimators=14, random_state=45)
bag_tuned.fit(X_train,y_train)
y_pred = bag_tuned.predict(X_test)
np.sqrt(mean_squared_error(y_test,y_pred))






















