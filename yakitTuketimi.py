import pandas as pd
import seaborn as sns
import numpy as np
import matplotlib.pyplot as plt

from scipy import stats
from scipy.stats import norm, skew

from sklearn.preprocessing import RobustScaler, StandardScaler
from sklearn.linear_model import LinearRegression, Ridge, Lasso, ElasticNet
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.metrics import mean_squared_error
from sklearn.base import BaseEstimator, TransformerMixin, RegressorMixin, clone


import xgboost as xgb

import warnings
warnings.filterwarnings('ignore')

# DATAYI İMPORT ETMEK VE DATA İÇİNİ İNCELEMEK

# data import edildi
column_name = ["MPG","Cylinders","Displacement","Horsepower","Weight","Acceleration","Model Year","Origin"]
data = pd.read_csv("auto-mpg.data",names=column_name,na_values="?",comment="\t",sep=" ",skipinitialspace=True)

data = data.rename(columns = {"MPG" : "Target"}) 
print(data.head())  # ilk 5 satırına bakmak
print("Data Boyut : ",data.shape)  # datanın boyutu
data.info()  # veri kümesine hızlı bir genel bakış
describe = data.describe()  # datanın istatiksel değerlerini döndürür
 
print("-------------------------------------------------------")
# EKSİK VERİLERİ TESPİT EDİP EKSİK VERİLERİN İÇİNİ DOLDURMA

print(data.isna().sum()) 
data["Horsepower"] = data["Horsepower"].fillna(data["Horsepower"].mean())  # mean deki değerleri horsepowerda boş olan kısımlara doldur
print(data.isna().sum())  # boş veri kalmadı
sns.distplot(data.Horsepower)  # Horsepower in dağılımı


corr_matrix = data.corr()  
sns.clustermap(corr_matrix, annot = True, fmt = ".2f")
plt.title("Özellik Arasındaki İlişki")
plt.show()


threshold = 0.75
filtre = np.abs(corr_matrix["Target"])>threshold # threshold u 0.75 ten büyük olanları al
corr_features = corr_matrix.columns[filtre].tolist() 
sns.clustermap(data[corr_features].corr(), annot = True, fmt = ".2f")
plt.title("Özellik Arasındaki İlişki")
plt.show()

## Eş düzlemlilik vardır

sns.pairplot(data, diag_kind = "kde", markers = "+") # Histogram olarak düşünelim
plt.show()

plt.figure()
sns.countplot(data["Cylinders"])
print(data["Cylinders"].value_counts())

plt.figure()
sns.countplot(data["Origin"])
print(data["Origin"].value_counts())

print("-------------------------------------------------------")
# Outlier (Aykırı Durumlar)
for c in data.columns:
    plt.figure()
    sns.boxplot(x = c, data = data, orient = "v") # v , H 
    

# Aykırı Durumların tespit Edilip Çıkarılması

#Hosepower için

thr = 2
horsepower_desc = describe["Horsepower"]
q3_hp = horsepower_desc[6]
q1_hp = horsepower_desc[4]
IQR_hp = q3_hp -q1_hp
top_limit_hp = q3_hp + thr*IQR_hp
bottom_limit_hp = q1_hp - thr*IQR_hp
filter_hp_bottom = bottom_limit_hp < data["Horsepower"]
filter_hp_top = data["Horsepower"] < top_limit_hp
filter_hp = filter_hp_bottom & filter_hp_top

data = data[filter_hp]

# Accelaration için

acceleration_desc = describe["Acceleration"]
q3_acc = acceleration_desc[6]
q1_acc = acceleration_desc[4]
IQR_acc = q3_acc -q1_acc
top_limit_acc = q3_acc + thr*IQR_acc
bottom_limit_acc = q1_acc - thr*IQR_acc
filter_acc_bottom = bottom_limit_acc < data["Acceleration"]
filter_acc_top = data["Acceleration"] < top_limit_acc
filter_acc = filter_acc_bottom & filter_acc_top

data = data[filter_acc]
print("-------------------------------------------------------")

# Skewness (Çarpıklık) İnceleme ve Ortadan Kaldırma

# Hedefe Bağlı Değişken

sns.distplot(data.Target, fit =norm)
(mu, sigma) = norm.fit(data["Target"])
print("mu : {}, sigma = {},".format(mu,sigma))

# qq plot
plt.figure()
stats.probplot(data["Target"], plot =plt)
plt.show()

data["Target"] = np.log1p(data["Target"])

plt.figure()
sns.distplot(data.Target, fit =norm)

(mu, sigma) = norm.fit(data["Target"])
print("mu : {}, sigma = {},".format(mu,sigma))

# qq plot
plt.figure()
stats.probplot(data["Target"], plot = plt)
plt.show()

# Özellikten Bağımsız Değişken
skewed_feats = data.apply(lambda x: skew(x.dropna())).sort_values(ascending = False)
skewness = pd.DataFrame(skewed_feats, columns = ["skewed"])                                                                
print("-------------------------------------------------------")

# ONE HOT ENCODING
data["Cylinders"] = data["Cylinders"].astype(str)
data["Origin"] = data["Origin"].astype(str)

data = pd.get_dummies(data)

print("-------------------------------------------------------")
#  Standardizasyon İşlemi
x = data.drop(["Target"],axis = 1)
y = data.Target

test_size = 0.9
X_train, X_test, Y_train, Y_test =train_test_split(x,y, test_size = test_size,random_state = 42)
# Standardizasyon
scaler = RobustScaler()  # RobustScaler
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)  # mean 0, standart sapma 1 olarak ayarlanmış oldu

print("-------------------------------------------------------")
# Regresyon Modeli
# Doğrusal Regresyon (Linear Regression)

lr = LinearRegression()
lr.fit(X_train, Y_train)
print("LR Coef : ",lr.coef_)
y_predicted_dummy = lr.predict(X_test)
mse = mean_squared_error(Y_test, y_predicted_dummy)
print("Linear Regression MSE : ",mse)

print("-------------------------------------------------------")
# Ridge Regression

ridge = Ridge(random_state = 42, max_iter = 10000)

alphas = np.logspace(-4,-0.5,30)

tuned_parameters = [{'alpha' : alphas}]
n_folds = 5

clf = GridSearchCV(ridge, tuned_parameters, cv = n_folds, scoring = "neg_mean_squared_error", refit = True)
clf.fit(X_train, Y_train)
scores = clf.cv_results_["mean_test_score"]
scores_std = clf.cv_results_["std_test_score"]

print("Ridge Coef : ",clf.best_estimator_.coef_)

ridge = clf.best_estimator_  # refit = True olduğu için clf kullanabiliyoruz , False olsaydı kullanamazdık

print("Ridge Best Estimator : ",ridge)

y_predicted_dummy = clf.predict(X_test)
mse = mean_squared_error(Y_test, y_predicted_dummy)
print("Ridge Regression MSE : ",mse)
print("-------------------------------------------------------")
plt.figure()
plt.semilogx(alphas, scores)
plt.xlabel("alpha")
plt.ylabel("score")
plt.title("Ridge")


# Lasso Regression

lasso = Lasso(random_state = 42, max_iter = 10000)

alphas = np.logspace(-4,-0.5,30)

tuned_parameters = [{'alpha' : alphas}]
n_folds = 5

clf = GridSearchCV(lasso, tuned_parameters, cv = n_folds, scoring = "neg_mean_squared_error", refit = True)
clf.fit(X_train, Y_train)
scores = clf.cv_results_["mean_test_score"]
scores_std = clf.cv_results_["std_test_score"]

print("Lasso Coef : ",clf.best_estimator_.coef_)

ridge = clf.best_estimator_  # refit = True olduğu için clf kullanabiliyoruz , False olsaydı kullanamazdık

print("Lasso Best Estimator : ",lasso)

y_predicted_dummy = clf.predict(X_test)
mse = mean_squared_error(Y_test, y_predicted_dummy)
print("Lasso Regression MSE : ",mse)
print("-------------------------------------------------------")
plt.figure()
plt.semilogx(alphas, scores)
plt.xlabel("alpha")
plt.ylabel("score")
plt.title("Lasso")


# ElasticNet
parametersGrid = {"alpha" : alphas,
                  "l1_ratio" : np.arange(0.0, 1.0, 0.05)}

eNet = ElasticNet(random_state=42, max_iter=10000)
clf = GridSearchCV(eNet, parametersGrid, cv = n_folds, scoring = "neg_mean_squared_error", refit = True)
clf.fit(X_train, Y_train)

print("ElasticNet Coef : ",clf.best_estimator_.coef_)
print("ElasticNet Best Estimator : ",clf.best_estimator_)

y_predicted_dummy = clf.predict(X_test)
mse = mean_squared_error(Y_test, y_predicted_dummy)
print("ElasticNet Regression MSE : ",mse)



# XGBoost

parametersGrid = { 'nthread':[4],
                  'objective':['reg:linear'],
                  'learning_rate':[.03, 0.05, .07],
                  'max_depth': [5, 6, 7],
                  'min_child_weight':[4],
                  'silent':[1],
                  'subsample':[0.7],
                  'colsample_bytree':[0.7],
                  'n_estimators':[500,1000]  }

model_xgb = xgb.XGBRegressor()

clf = GridSearchCV(model_xgb, parametersGrid, cv = n_folds, scoring = 'neg_mean_squared_error', refit = True, n_jobs = 5, verbose = True)

clf.fit(X_train, Y_train)

y_predicted_dummy = clf.predict(X_test)
mse = mean_squared_error(Y_test, y_predicted_dummy)
print("XGB Regressor MSE : ",mse)

# Modellerin Ortalaması

class AveragingModels():
    def __init__(self, models):
        self.models = models
        
    def fit(self, X, y):
        self.models_ = [clone(x) for x in self.models]
        
        for model in self.models_:
            model.fit(X, y)
            
        return self
 
    def predict(self, X):
        predictions = np.column_stack([model.predict(X) for model in self.models_])
        return np.mean(predictions, axis=1)
    
averaged_models = AveragingModels(models = (model_xgb, lasso))
averaged_models.fit(X_train, Y_train)

y_predicted_dummy = averaged_models.predict(X_test)
mse = mean_squared_error(Y_test, y_predicted_dummy )
print("Averaged Models MSE : ",mse)


"""
Linear Regression MSE :  0.020984711065869636
Ridge MSE : 0.018839299330570596
Lasso MSE :  0.016597127172690827
ElasticNet MSE :  0.017234676963922276
XGBRegressor MSE :  0.017444718427058307
Averaged Models MSE :  0.037661199313583534 -> Modellerin ortalaması
"""




