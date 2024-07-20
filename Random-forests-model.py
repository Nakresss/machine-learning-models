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
from sklearn.ensemble import RandomForestRegressor

hit = pd.read_csv("hitters.csv")
df = hit.copy()
df = df.dropna()
dms = pd.get_dummies(df[['League','Division','NewLeague']])
y = df["Salary"]
x_ = df.drop(['Salary','League','Division','NewLeague'],axis=1).astype('float64')
x = pd.concat([X_, dms[["League_N","Division_W","NewLeague_N"]]], axis = 1)
X_train,X_test,y_train, y_test = train_test_split(X,y,test_size=0.25, random_state=42)

rf_model = RandomForestRegressor(random_state = 42)
rf_model.fit(X_train, y_train)
y_pred = rf_model.predict(X_test)[0:5]
np.sqrt(mean_squared_error(y_test,y_pred))
rf_params = {'max_depth': list(range(1,10)), 'max_features':[2,3,10,15],'n_estimators':[200,500,1000,2000]}
rf_model = RandomForestsRegressor(random_state=42)
rf_cv_model = GridSearchCV(rf_model,rf_params,cv=10,n_jobs=-1)
rf_cv_model.fit(X_train,y_train)
rf_cv_model.best_params_
rf_tuned = RandomForestRegressor(max_depth=8,max_feature=3,n_estimators=200)
rf_tuned.fit(X_train,y_train)
y_pred = rf_tuned.predict(X_test)
np.sqrt(mean_squared_error(y_test,y_pred))

Importance = pd.DataFrame({"Importance":rf_tuned.feature_importances_*100},index = X_train.columns)
Importance.sort_values(by ="Importance",axis=0,ascending=True).plot(kind='barh',color="r")
plt.xlabel("Değişken önem düzeyleri")























