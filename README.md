🏠 House Price Prediction - Advanced ML Pipeline (Hilal Ay)

🔗 Veri Seti Linki (Kaggle)
Kullanılan veri setine ve detaylarına Kaggle üzerinden erişebilirsiniz:
House Prices: Advanced Regression Techniques 

https://www.kaggle.com/c/house-prices-advanced-regression-techniques/data

🚀Kurulum ve Ortam Hazırlığı
Projenin taşınabilirliği ve sürümler arası uyumluluğu için version pinning (sürüm sabitleme) uygulanmıştır.

Terminalinizi açın ve proje ana dizinine gidin.

Gerekli tüm kütüphaneleri (XGBoost, Scikit-learn, Joblib vb.)  kurmak için şu komutu çalıştırın:

pip install -r requirements.txt

Projenin ana dosyası main.py'dir. 

🛠️ Teknik Mimari ve Mühendislik Kararları
Proje şu 5 kritik sütun üzerine inşa edilmiştir:

1. Modüler Klasör Yapısı

Proje, monolitik yapıdan kurtarılarak endüstri standardı olan modüler yapıya geçirilmiştir
:src/preprocessing.py: ColumnTransformer ile otomatik veri temizleme ve dönüşüm.
src/model.py: Model mimarisi, GridSearchCV optimizasyonu ve log-transform mantığı.
src/evaluation.py: Performans analizleri, RMSE/MAE hesaplamaları ve görsel çıktı yönetimi.
src/main.py: Tüm pipeline'ı tek komutla çalıştıran orkestra şefi.
src/predict.py: Bağımsız tahmin (inference) aracı.

2. Pipeline Mimarisi & ColumnTransformer

Veri sızıntısını (Data Leakage) önlemek için tüm adımlar bir Scikit-learn Pipeline içine alınmıştır.Sayısal Veriler: SimpleImputer(strategy='median') + StandardScaler()Kategorik Veriler: SimpleImputer(strategy='constant') + OneHotEncoder(handle_unknown='ignore')Bu yapı sayesinde, eğitim setindeki istatistiklerin test setine sızması engellenmiş ve yeni gelen veriler için tam güvenlik sağlanmıştır.

3. Log-Target Modeling (Target Engineering)

Ev fiyatlarının sağa çarpık (right-skewed) dağılımını normalize etmek amacıyla hedef değişkene (SalePrice) Log1p dönüşümü uygulanmış; eğitim bu ölçekte yapılmıştır. Tahmin aşamasında sonuçlar Expm1 ile orijinal dolar birimine geri döndürülmüştür.

4. Hiperparametre Optimizasyonu & Cross-Validation

Modelin genelleme yeteneğini ispatlamak için tek split yerine 5-Fold Cross-Validation kullanılmıştır. Ridge ve XGBoost gibi modellerin en iyi parametreleri (alpha, learning_rate vb.) GridSearchCV ile bilimsel olarak belirlenmiştir.

5. Model Persistence (Kalıcılık)

Eğitilen şampiyon model, joblib kütüphanesi ile paketlenerek final_model.joblib olarak kaydedilmiştir.

📊 Güncel Performans Metrikleri

Yeni mimaride elde edilen dürüst doğrulama sonuçları:
Model Name,         R² (%),     MAE ($),            RMSE ($)
Ridge (Tuned) 🏆    92.70,    "13,926.36",      "19,567.50"
XGBoost (Best),      91.99,   "14,456.14",      "20,498.07"
Linear Regression,   91.60,    "14,513.31",     "20,994.44"

🧠 Design Choices & FAQ (Teknik Kararlar)

1. Neden Şampiyon Ridge (Tuned) Oldu?

XGBoost gibi kompleks modeller genellikle öne çıksa da, Ames Housing gibi veri setlerinde verinin doğrusal (linear) doğası ve başarılı Target Engineering (log-transform) sayesinde Ridge Regresyon, aşırı öğrenmeyi (overfitting) daha iyi domine ederek hem R² hem de RMSE metriklerinde en dengeli sonucu vermiştir.

2. Neden Log-Target Transformation Uygulandı?

Problem: Yüksek fiyatlı uç değerler (outliers), hataların karesini alan RMSE metriklerini domine ederek modeli yanıltıyordu.Çözüm: Log dönüşümü, fiyat dağılımını "Normal Dağılım"a yaklaştırarak modelin daha stabil katsayılar öğrenmesini sağlamış, varyansı stabilize etmiştir.

