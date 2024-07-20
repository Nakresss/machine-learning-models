hit = pd.read_csv("hitters.csv")
df = hit.copy()
df = df.dropna()
ms = pd.get_dummies(df[['League','Division','NewLeague']])
y = df["Salary"]
x_ = df.drop(['Salary','League','Division','NewLeague'],axis=1).astype('float64')
x = pd.concat([X_, dms[["League_N","Division_W","NewLeague_N"]]], axis = 1)
X_train,X_test,y_train, y_test = train_test_split(X,y,test_size=0.25, random_state=42)

from sklearn.linear_model import Lasso
lasso_model = Lasso(alpha=0.1).fit(X_train,y_train)
lasso_model.coef_

lambdalar = 10**np.linspace(10,-2,100)*5

lasso = Ridge
katsayılar = []


for i in lambdalar:
    lasso.setparams(alpha = i)
    lasso.fit(X_train, y_train)
    katsayılar.append(lasso.coef_)

ax = plt.gca()
ax.plot(lambdalar*2,katsayılar)
ax.setxscale('log')

plt.xlabel('tight')
plt.ylabel('alpha')
plt.title('weights')
#Tahmin
lasso_model.predict(X_test)

y_pred = lasso_model.predict(X_test)
np.sqrt(mean_squared_error(y_test,y_pred))
#tuning
from sklearn.linear_model import LassıCV

lasso_cv_model = LassoCV(alphas= None, cv =10, max_iter = 10000, normalize = True)
lasso_cv_model.fit(X_train,y_train)
lasso_cv_model.alpha_

lasso_tuned = Lasso(alpha = lasso_cv_model.alpha_)
lasso_tuned.fit(X_train,y_train)
y_pred = lasso_tuned.predict(X_test)

np.sqrt(mean_squared_error(y_test,y_pred))





