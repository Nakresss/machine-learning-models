from sklearn.linear_model import Elasticnet

hit = pd.read_csv("hitters.csv")
df = hit.copy()
df = df.dropna()
ms = pd.get_dummies(df[['League','Division','NewLeague']])
y = df["Salary"]
x_ = df.drop(['Salary','League','Division','NewLeague'],axis=1).astype('float64')
x = pd.concat([X_, dms[["League_N","Division_W","NewLeague_N"]]], axis = 1)
X_train,X_test,y_train, y_test = train_test_split(X,y,test_size=0.25, random_state=42)


enet_model = Elasticnet().fit(X_train,y_train)
enet_model.coef_
enet_model.intercept_

enet_model.predict(X_test)
y_pred = enet_tuned.predict(X_test)
np.sqrt(mean_squared_error(y_test,y_pred))

r2_score(y_test,y_pred)

from sklearn.linear_model import ElastikNetCV
enet_cv_model = ElastikNetCV(cv=10, random_state = 0).fit(X_train,y_train)
enet_cv_model.alpha_
enet_tuned = ElastikNet(alpha = enet_cv_model.alpha_).fit(X_train,y_train)
y_pred = enet_tuned.predict(X_test)
np.sqrt(mean_squared_error(y_test,y_pred))















