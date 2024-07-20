from sklearn.metrics import mean_squared_error, r2_score

ad = pd.read_csv("Advertising.csv")
df = ad.copy()

#modelimizi hazırlıyoruz
#Statsmodels
X = df[["TV"]]
X[0:5]
#Matris işlemi için gerekenler
X = sm.add_constant(X)
X[0:5]
y = df["sales"]
lm = sm. OLS(y,X)
model = lm.fit()

lm = smf.ols("sales ~TV", df)
model = lm.fit()
#Tahmin edilen değerler ile gerçek değerlerin arasındaki farkların karelerinin ortalaması için
mse = mean_squared_error(y, model.fittedvalues)
print(mse)

import numpy as np

rmse = np.sqrt(sme)
reg.predict(X)[0:10]

#Karşılaştırma tablosu için
k_t = pd.DataFrame({"Gerçek_y": y[0:10],
"Tahmin_y": reg.predict(X)[0:10]})
print(k_t)

#Hatalarımızı görmek için

k_t["hata"] = k_t["gerçek_y"]-k_t["Tahmin_y"]
print(k_t)

#Toplam hata için
np.sum(k_t["hata_kare"])

#Hataların ortalaması için
np.mean(k_t["Hata_kare"])

#Hata ortalamasının karekökü için
np.sqrt(np.mean(k_t["Hata_kare"]))

#Modelin artıklarını ifade etmek için
mode.resid[0:10]

plt.plot(model.resid)




