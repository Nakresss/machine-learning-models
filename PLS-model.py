from sklearn.cross_decomposition import PLSRegression, PLSSVD

hit = pd.read_csv("hitters.csv")
df = hit.copy()
df = df.dropna()
ms = pd.get_dummies(df[['League','Division','NewLeague']])
y = df["Salary"]
x_ = df.drop(['Salary','League','Division','NewLeague'],axis=1).astype('float64')
x = pd.concat([X_, dms[["League_N","Division_W","NewLeague_N"]]], axis = 1)
X_train,X_test,y_train, y_test = train_test_split(X,y,test_size=0.25, random_state=42)


pls_model = PLSRegression(n_components=6).fit(X_train, y_train)
pls_model.coef_

#Tahmin için

pls_model.predict(X_train)[0:10]
y_pred = pls_model.predict(X_train)
np.sqrt(mean_squared_error(y_train,y_pred))
r2_score(y_train,y_pred)

pls_model.predict(X_test)
np.sqrt(mean_squared_error(y_test,y_pred))


#model tuning

cv_10 = moıdel_selection.KFold(n_splits = 10, shuffle = True, random_state = 1)
RMSE = []


for i in np.qrqnge(1, X_reduced_train.shape[1] + 1):
    score=np.sqrt(-1*model_selection.cross_val_score(lm,X_reduced_train[:,:i],y_train.ravel(),cv=cv_10,scoring='neg_mean_squared_error').mean())
    RMSE.append(score)


plt.plot(RMSE,'-v')
plt.xlabel('Bileşen Sayısı')
plt.ylabel('RMSE')
plt.title('Salary');

