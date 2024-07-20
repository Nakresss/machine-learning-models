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

np.random.seed(3)

x_sim = np.random.uniform(2,10,145)
y_sin = np.sin(x_sim) + np.random.normal(0,0.4,145)

x_outliers = np.arange(2.5,5,0.5)
y_outliers = 5*np.ones(5)

x_sim_idx = np.argort(np.concatenate([x_sim,x_outliers]))
x_sim = np.concatenate([x_sim, x_outliers])[x_sim_idx]
y_sim = np.concatenate([y_sim, y_outliers])[x_sim_idx]

ols = LinearRegression()
ols.fit(np.sin(x_sim[:,np.newaxis]), y_sim)
ols_pred = ols.predict(np.sin(x_sim[:,np.newaxis]))

eps = 0.1
svr=SVR('rbf',epsilon = eps)
svr.fit(x_sim[:np.newaxis],y_sim)
svr_pred = svr.predict(x_sim[:np.newaxis])

plt.scatter(x_sim,y_sim,alpha=0.5,s=26)
plt_ols, = plt.plot(x_sim, ols_pred,'g')
plt_svr, = plt.plot(x_sim, svr_pred, color='r')
plt.xlabel("Bağımsız değişken")
plt.ylabel("bağımlı değişken")
plt.ylim(-5.2,2.2)
plt.legend([plt_ols,plt_svr],['EKK','SVR'],loc = 4)

hit = pd.read_csv("hitters.csv")
df = hit.copy()
df = df.dropna()
ms = pd.get_dummies(df[['League','Division','NewLeague']])
y = df["Salary"]
x_ = df.drop(['Salary','League','Division','NewLeague'],axis=1).astype('float64')
x = pd.concat([X_, dms[["League_N","Division_W","NewLeague_N"]]], axis = 1)
X_train,X_test,y_train, y_test = train_test_split(X,y,test_size=0.25, random_state=42)

svr_rbf = SVR("rbf").fit(X_train,y_train)

y_pred = svr_rbf.predict(X_test)
np.sqrt(mean_squared_error(y_test,y_pred))

svr_params = {"C":[0.1,0.4,5,10,20,30,40,50]}
svr_cv_model = GridSearchCV(svr_rbf,svr_params, cv=10)
svr_cv_model.fit(X_train,y_train)
pd.Series(svr_cv_model.best_params_)[0]
svr_tuned = SVR("rbf", C=pd.Series(svr_cv_models.best_params_)[0]).fit(X_train,y_train)

y_pred = svr_tuned.predict(X_train)
np.sqrt(mean_squared_error(y_test,y_pred))
















