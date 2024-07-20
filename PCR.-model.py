hit = pd.read_csv("Hitters.csv")
df = hit.copy()
df = df.dropna()


df.head()
df.info()
df.describe().T

#Katagorik değişkenleri dummi değişken formatına çeviriyoruz

dms = pd.get_dummies(df[['League','division', 'NewLeague']])
dms.head()
#modellemeye hazır hale getirelim
y = df ["Salary"]
X_ = df.drop(["Salary","League","Division","NewLeague"],axis = 1).astype("float64")
x = pd.concat([X_, dms[["League_N","Division_W","NewLeague_N"]]], axis = 1)

#test train için hazırlıyoruz

X_train,X_test,y_train,y_test = train_test_split(X,y,test_size=0.25,random_state=42)
#Oranları alalım
print("X_train", X_train.shape)
print("y_train", y_train.shape)
print("X_test", X_test.shape)
print("y_test", y_test.shape)
training = df.copy()
print("training", training.shape)


from sklearn.decomposition import PCA
from sklearn.preprocessing import scale
pca = PCA()
X_reduced_train = pca.fit_transform(scale(X_train))#dönüştürme işlemi ile ilgili uyarı verebilir çok onunla ilgilenmeyin

#gözlemler
X_reduced_train[0:1:]

#Açıklanan varyans
np.cumsum(np.round(pca.explainined_variance_ratio_, decimals=4)*100[0:10])

#Tüm bileşenleri kullanarak modeli fit etme işlemi
lm = LinearRegression()
pcr_model = lm.fit(X_reduced_train[:,0:1], y_train)
pcr_model.intercept_
pcr_model.coef_

#Kurmuş olduğumuz model ile tahmin işlemi gerçekleştiricez

y_pred = pcr_model.predict(X_reduced_train[:,0:1])
y_pred[0:5]
np.sqrt(mean_squared_error(y_train, y_pred))
df["Salary"].mean()
r2_score(y_train,y_pred)


pca2 = pca()
X_reduced_test = pca2.fit_transform(scale(X_test))

y_pred = pcr_model.predict(X_reduced_test)
print(np.sqrt(mean_squared_error(y_test,y_pred)))

from sklearn import model_selection

cv_10 = moıdel_selection.KFold(n_splits = 10, shuffle = True, random_state = 1)
lm = LinearRegression()
RMSE = []

for i in np.qrqnge(1, X_reduced_train.shape[1] + 1):
    score=np.sqrt(-1*model_selection.cross_val_score(lm,X_reduced_train[:,:i],y_train.ravel(),cv=cv_10,scoring='neg_mean_squared_error').mean())
    RMSE.append(score)
plt.plot(RMSE,'-v')
plt.xlabel('Bileşen Sayısı')
plt.ylabel('RMSE')
plt.title('Maaş Tahmin Modeli için PCR model tuning');


lm = LinearRegrassion()
pcr_model = pcr_model.predict(X_reduced_train[:,0:6])
print(np.sqrt(mean_squared_error(y_train,y_pred)))
pcr_model = pcr_model.predict(X_reduced_test[:,0:6])
print(np.sqrt(mean_squared_error(y_test,y_pred)))


















