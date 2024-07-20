import pandas as pd
import seaborn as sns
import statsmodels.api as sm
import statsmodels.formula.api as smf



ad = pd.read_csv("Advertising.csv")
df = ad.copy()
df.head()

#index değişken olarak aldığında çözüm için
df = df.iloc[:,1:len(df)]
df.info()

#Datayı incelemek için
df.describe().T
#Eksik değer olup olmadığına bakalım
df.isnull().values.any()
#değişkenlerin dağılımlarını inceleyelim
df.corr()

#grafiğimizde görelim
sns.pairplot(df,kind ="reg");

#Örnek olarak TV ve Satışın grafiğini dağılımını görmek için
sns.joinplot(x = "TV", y = "sales", data = df, kind = "reg")

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
model.summary()

#değişkenleri farklı bir şekilde ifade edelim
lm = smf.ols("sales ~ TV",df)
model = lm.fit()
model.summary()

#modelin parametrelerini görmek için
model.params

#katsayı detayları için
model.summary().tables[1]

#Katsayıların güven aralıklarına erişmek için
model.conf_int()

#modelin anlamlılığına ilişkin pvalue ifadesi
model.f_pvalue


#Pvalue değerini yazdırmak için
print("f_pvalue:","%.4f" % model.f_pvalue)


#F istatistiğine erişmek için
print("fvalue:","%.2f" % model.fvalue)


#Tvalues değeri için
print("tvalue:","%.2f" % model.tvalues[0:1])

#modelin anlamlılığına ilişkin model değerlendirme istatistiklerine erişmek için
model.mse_model#burda çıkacak değer şuan için çok korkunç bir değer çıkıcaktır.

#Açıklanabilirlik oranı için
model.rsquared_adj

#modelin tahmin ettiği değerlere erşmek için
model.fittedvalues[0:5]


#modelin denklemini yazmak Mülakat sorusu genelde
print("Sales =" + str("%.2f"% model.params[0])+"TV"+"*"+str("%.2f"%model.params[1]))

#görsel olarak ifadesine bakalım modelimizin

g = sns.regplot(df["TV"], df["sales"], ci=None, scatter_kws={'color':'r','s':9})
g.set_title("Model denklemi: Sales = 7.03 + TV*0.05")
get.set_ylabel("Satış Sayısı")
g.set_xlabel("TV Harcamaları")
plt.xlim(-10,310)
plt.ylim(bottomm =0);

#Skitlearn ile kuralım modelimizi

from sklearn.linear_model import LinearRegression

X = df[["TV"]]
y = df["sales"]
reg = LinearRegression()
model = reg.fit(X,y)
model.intercept_
model.coef_

#modelin scoru için
model.score(X,y)

#Tahmin edilen değerler
model.predict(X)[0:10]


#Modele bir kaç tahmin gerçekleştirelim

#Örneğin 30 birim TV harcaması olduğunda satışların tahmini değeri ne olur ?
7.03 + 30*0.4
X = df [["TV"]]
y = df["sales"]
reg = LinearRegression()
model = reg.fit(X,y)
model.predict([[30]])


#Satış için 3 farklı departmandan tahmin geldiğini düşünelim
yeni_veri = [[5],[90],[200]]
model.predict(yeni_veri)







