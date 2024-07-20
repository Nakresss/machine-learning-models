from sklearn.Linear_model import Ridge


hit = pd.read_csv("hitters.csv")
df = hit.copy()
df = df.dropna()
ms = pd.get_dummies(df[['League','Division','NewLeague']])
y = df["Salary"]
x_ = df.drop(['Salary','League','Division','NewLeague'],axis=1).astype('float64')
x = pd.concat([X_, dms[["League_N","Division_W","NewLeague_N"]]], axis = 1)
X_train,X_test,y_train, y_test = train_test_split(X,y,test_size=0.25, random_state=42)

ridge_model = ridge(alpha = 0.1).fit(X_train,y_train)

print(ridge_model.coef_)

lambdalar = 10**np.linspace(10,-2,100)*5

ridge_model = Ridge
katsayılar = []


for i in lambdalar:
    ridge_model.setparams(alpha = i)
    ridge_model.fit(X_train, y_train)
    katsayılar.append(ridge_model.coef_)

ax = plt.gca()
ax.plot(lambdalar,katsayılar)
ax.setxscale('log')

plt.xlabel('Lambda(alpha) değeri')
plt.ylabel('katsayılar/ağırlıklar')
plt.title('Düzenlileştirmenin bir fonksiyonu olarak ridge katsayıları');

y_pred = ridge_model.predict(X_test)
np.sqrt(mean_squared_error(y_test,y_pred))

from sklearn.linear_model import RidgeCV
ridge_cv = RidgeCV(alphas = lambdalar, scoring = "neg_mean_squared_error", normalize = True)

ridge_cv.fit (X_train, y_train)
ridge.cv.alpha_
ridge_tuned = ridge(alpha = ridge_cv.alpha_, normalize = True).fit(X_train, y_train)
np.sqrt(mean_squared_error(y_test,ridge_tuned.predict(X_test)))
























