import pandas as pd
from sklearn.model_selection import train_test_split,cross_val_score, cross_val_predict


ad = pd.read_csv("Advertising.csv",usecols = [1,2,3,4,])
df = ad.copy()
df.head()

#Sales değişkenini dışarda bırakıp diğer tüm bağımsız değişkenleri seçme işlemi
X = df.drop("sales",axis = 1)
y = df["sales"]
#veri setini ayırma işlemi
X_train, X_test, y_train, y_test = train_test_split(X,y, test_size = 0.20, random_state=42)#her üretmede farklı değerler çıkmaması için random state argümanı girdim

#Train setlerimize bakmak için
X_train.shape
y_train.shape

#test setimize bakmak için
X_test.shape
y_test.shape

#veri setnin tüm halini barındırması için
training = df.copy()
#veri setimizin ilk hali için
training.shape

#Çoklu doğrusal regresyon modelini kurma işlemi

lm = sm. OLS(y_train, X_train)
model = lm.fit()
model.summary()

#model çıktısından bağzı sonuçlara erişmek için
model.summary().table[1]

#modelimizin tahmin başarısı
#BUNUN İÇİN SCİKİT-LEARN İLE MODEL KURUYORUZ
lm = LinearRegression()
mode = lm.fit(X_train, y_train)

#sabit katsayı için
model.intercept_

#diğer tüm katsayılar
model.coef_


#TAHMİN
# model denklemi: Sales = 2.97 + TV0.04 + radio0.18 + newpaper*0.002
#Örneğin 30 birim TV harcaması, 10 birim radio harcaması 40 birimde gazete harcaması olduğunda satışların tahmini değeri ne olur?

yeni_veri = [[30],[10],[40]] 
yeni_veri = pd.DataFrame(yeni_veri).T

model.predict(yeni_veri) #tahmin değeri için

#Tahmin başarısı için
rmse = np.sqtr(mean_squared_error(y_train,model.predict(X_train)))
print(rmse)

rmse = np.sqtr(mean_squared_error(y_test,model.predict(X_test)))

print(rmse)

model.score(X_train, y_train)

#Model tuning (doğrulama) işlemi
cross_val_score(model, X,y, cv = 10, scoring = "r2").mean() 
cross_val_score(model, X_train,y_train, cv = 10, scoring = "r2").mean() 
cross_val_score(model, X_test,y_test, cv = 10, scoring = "r2").mean() 
np.sqrt(-cross_val_score(model, X_train,y_train, cv = 10, scoring = "neg_mean_squared_error")).mean()












