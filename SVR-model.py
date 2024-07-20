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

hit = pd.read_csv("hitters.csv")
df = hit.copy()
df = df.dropna()
dms = pd.get_dummies(df[['League','Division','NewLeague']])
y = df["Salary"]
x_ = df.drop(['Salary','League','Division','NewLeague'],axis=1).astype('float64')
x = pd.concat([X_, dms[["League_N","Division_W","NewLeague_N"]]], axis = 1)
X_train,X_test,y_train, y_test = train_test_split(X,y,test_size=0.25, random_state=42)

X_train = pd.DataFrame(X_train["hits"])
X_test = pd.DataFrame(Xtest["hits"])


svr_model = SVR("linear").fit(X_train,y_train)
svr_model.predict(X_train)[0:5]

print("y= {0}+{1} x".format(svr_model.intercept_[0],svr_model.coef_[0][0]))
X_train["hits"][0:1]

plt.scatter(X_train,y_train)
plt.plot(X_train,y_pred, color="r")

lm_model = LinearRegression().fit(X_train,y_train)
lm_pred = lm_model.predict(X_train)
print("y = {0}+{1} x".format(lm_model.intercept_,lm_model.coef_[0]))

svr_model.predict([[91]])
y_pred = svr_model.predict(X_test)

np.sqrt(mean_squared_error(y_test,y_pred))

print(svr_model)

svr_params = {"c": np.arange(0.1,2,0.1)}
svr_cv_model = GridSearchCV(svr_modelisvr_params,cv=10).fit(X_train,y_train)
svr_cv_model.best_params_
svr_tuned = SVR("linear", C=pd.Series(svr_cv_model.best_params_).fit(X_train,y_train))

y_pred = svr_tuned.predict(X_test)
np.sqrt(mean_squared_error(y_test,y_pred))




