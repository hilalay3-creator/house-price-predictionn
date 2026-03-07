🏠 House Price Prediction - Bitirme Projesi (Hilal Ay)
📖 Proje Özeti
Bu çalışma, Ames Housing veri seti üzerinde ev fiyatlarını tahmin etmek için geliştirilmiş uçtan uca bir makine öğrenmesi projesidir. Proje kapsamında ham veriler üzerinde derinlemesine analizler (EDA) yapılmış, veriler temizlenmiş ve Lasso/Ridge regresyon modelleri ile fiyat tahmini gerçekleştirilmiştir.

🔗 Veri Seti Linki (Kaggle)
Kullanılan veri setine ve detaylarına Kaggle üzerinden erişebilirsiniz:
House Prices: Advanced Regression Techniques 
https://www.kaggle.com/c/house-prices-advanced-regression-techniques/data


🛠️ Kurulum
Projenin çalışması için VS Code veya herhangi bir Python editöründe aşağıdaki kütüphanelerin yüklü olması gerekmektedir:

Terminalinizi açın.
Şu komutu yazarak gerekli tüm kütüphaneleri (Pandas, Scikit-learn vb.) tek seferde kurun:
Bash
pip install -r requirements.txt

🚀 Nasıl Çalıştırılır?
Proje dosya yapısı birbirine bağımlı şekilde kurgulanmıştır:

İndirdiğiniz train.csv dosyasını data/ klasörünün içine koyun.

notebooks/01_modeling.py dosyasını VS Code ile açın.

Dosya, veriyi otomatik olarak ../data/train.csv yolundan okuyacak şekilde ayarlanmıştır.

Dosyayı çalıştırdığınızda analiz raporlarını ve modellerin başarı skorlarını terminalde görebilirsiniz.

📊 Sonuç Metrikleri (Şampiyon Model: Lasso)
R² Skoru: %90.54 (Doğruluk oranı)

MAE (Ortalama Mutlak Hata): 16.045,91 $

RMSE (Kök Ortalama Kare Hata): 22.277,02 $

💡 Kısa Yorum ve Çıkarımlar
Proje sırasında model başarısını artıran kritik mühendislik kararları şunlardır:

Aykırı Değerler: Görsel analizler sonucu 4000 m² üstü yaşam alanı olan evlerin modelde sapmalara yol açtığı saptanmış ve bu veriler temizlenerek modelin genelleme yeteneği artırılmıştır.

İkiz Sütunlar: Korelasyon matrisi ile birbirinin %90 ve üzeri kopyası olan "ikiz sütunlar" saptanmış ve sistemden elenmiştir. Bu işlem modelin gürültüden arınmasını sağlamış ve başarıyı doğrudan etkilemiştir.

Mühürleme: Kategorik boşluklar "None" ile mühürlenmiş, garaj ve yapı yılı gibi sayısal verilerde mantıksal ispatlar (Proof) yapılarak veri seti tutarlı hale getirilmiştir.

Hazırlayan: Hilal Ay (MSc Candidate in Electrical Engineering)

