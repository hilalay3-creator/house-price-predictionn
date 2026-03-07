import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LinearRegression, Ridge, Lasso
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error

# =================================================================
# 1. VERİ YÜKLEME VE İLK BAKIŞ
# =================================================================
df = pd.read_csv('../data/train.csv')
print('🏠 HOUSE PRICE PREDICTING UYGULAMASI')
print(f"Veri setinde toplam {df.shape[0]} satır ve {df.shape[1]} sütun bulunmaktadır.\n")

# =================================================================
# 2. KATEGORİK EKSİKLERİ "NONE" İLE MÜHÜRLEME
# =================================================================
kelime_sutunlari = df.select_dtypes(include=['object','str']).columns.tolist()
for sutun in kelime_sutunlari:
    df[sutun] = df[sutun].fillna("None")

print("✅ Kategorik sütunlar 'None' ile mühürlendi.")

# =================================================================
# 3. SAYISAL EKSİK ANALİZİ VE MANTIK KONTROLLERİ (HOCAYA İSPAT)
# =================================================================
sayisal_df = df.select_dtypes(include=['number'])
sayisal_eksikler = sayisal_df.isnull().sum()

print("\n" + "="*50)
print("      SAYISAL SÜTUN EKSİK VERİ RAPORU")
print("="*50)
print("[!] İçinde Boşluk Olan Sayısal Sütunlar:")
print(sayisal_eksikler[sayisal_eksikler > 0])

# Proof: Neden GarageYrBlt'yi 0 ile dolduruyoruz?
garajı_olmayanlar = df[df["GarageType"] == "None"]
yıl_boş_mu = garajı_olmayanlar["GarageYrBlt"].isnull().sum()
toplam_yok = len(garajı_olmayanlar)

print("\n" + "="*50)
print("      GARAJ MANTIK KONTROLÜ (PROOF)")
print("="*50)
print(f"--> GarageType 'None' olan toplam ev sayısı: {toplam_yok}")
print(f"--> Bu evlerden garaj yılı BOŞ (NaN) olanların sayısı: {yıl_boş_mu}")

if toplam_yok == yıl_boş_mu:
    print("\n✅ KANITLANDI: Garaj tipi 'None' olan her evin yılı da boş.")
    print("Yani bu evlerde fiziksel olarak bir garaj ünitesi mevcut değil.")

# İstatistiksel Özetler (Karar Destek)
print("\n--- LotFrontage & MasVnrArea Özeti ---")
print(df[["LotFrontage", "MasVnrArea"]].describe().T)

# Sayısal Boşlukları Doldurma
df["LotFrontage"] = df["LotFrontage"].fillna(df["LotFrontage"].median())
df["MasVnrArea"] = df["MasVnrArea"].fillna(df["MasVnrArea"].median())
df["GarageYrBlt"] = df["GarageYrBlt"].fillna(0)

print(f"\n✅ Sayısal eksikler kapatıldı. Kalan toplam eksik: {df.isnull().sum().sum()}")

# =================================================================
# 4. AYKIRI DEĞER (OUTLIER) ANALİZİ VE TEMİZLİĞİ
# =================================================================
plt.figure(figsize=(10, 6))
sns.scatterplot(x=df['GrLivArea'], y=df['SalePrice'], color='teal', alpha=0.6)
plt.axvline(x=4000, color='r', linestyle='--', label='Kritik Sınır (4000 m2)')
plt.title('Yaşam Alanı vs Satış Fiyatı (Aykırı Değer Analizi)')
plt.legend()
plt.show()

df = df[df['GrLivArea'] < 4000].copy()
print(f"✅ Aykırı değerler temizlendi. Yeni satır sayısı: {len(df)}")

# =================================================================
# 5. ENCODING VE İKİZ SÜTUN TEMİZLİĞİ
# =================================================================
df_encoded = pd.get_dummies(df, drop_first=True)

# İkiz Sütun Analizi
corr_matrix = df_encoded.corr().abs()
upper = corr_matrix.where(np.triu(np.ones(corr_matrix.shape), k=1).astype(bool))
to_drop = [column for column in upper.columns if any(upper[column] > 0.90)]
df_final = df_encoded.drop(columns=to_drop)

print(f"🔗 İkiz olduğu için elenen gürültülü sütun sayısı: {len(to_drop)}")

# =================================================================
# 6. VERİ BÖLME VE ÖLÇEKLENDİRME (SCALING)
# =================================================================
X = df_final.drop(['SalePrice', 'Id'], axis=1)
y = df_final['SalePrice']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.20, random_state=42)

scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# =================================================================
# 7. MODELLERİN EĞİTİMİ VE BAŞARI KIYASLAMASI
# =================================================================
models = {
    "Linear Regression": LinearRegression(),
    "Lasso (L1)": Lasso(alpha=100, max_iter=10000),
    "Ridge (L2)": Ridge(alpha=1.0)
}

print("\n" + "="*45 + "\n🏆 MODELLERİN NİHAİ KARŞILAŞTIRMASI\n" + "="*45)
results = {}
for name, model in models.items():
    model.fit(X_train_scaled, y_train)
    score = model.score(X_test_scaled, y_test)
    results[name] = score * 100
    print(f"{name:20} R2 Skoru: %{score*100:.2f}")

# =================================================================
# 8. PERFORMANS RAPORU (LASSO)
# =================================================================
best_model_name = "Lasso (L1)"
y_final_pred = models[best_model_name].predict(X_test_scaled)
mae = mean_absolute_error(y_test, y_final_pred)
rmse = np.sqrt(mean_squared_error(y_test, y_final_pred))

print("\n" + "🏆" * 20 + "\n       BİTİRME ÖDEVİ PERFORMANS RAPORU\n" + "🏆" * 20)
print(f"{'R2 Skoru (Doğruluk)':<25}: %{results[best_model_name]:.2f}")
print(f"{'MAE (Ortalama Hata)':<25}: {mae:.2f} $")
print(f"{'RMSE (Kritik Hata)':<25}: {rmse:.2f} $")
print("=" * 40)

# =================================================================
# 9. GÖRSEL ANALİZLER (ÜÇLÜ PANEL)
# =================================================================

plt.figure(figsize=(20, 6))

plt.subplot(1, 3, 1)
sns.scatterplot(x=y_test, y=y_final_pred, alpha=0.5, color='#16a085')
plt.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], 'r--', lw=2)
plt.title('Tahmin vs Gerçek Fiyatlar')

plt.subplot(1, 3, 2)
sns.histplot(y_test - y_final_pred, kde=True, color='#e67e22')
plt.axvline(x=0, color='red', linestyle='--')
plt.title('Hata Dağılımı (Residuals)')

plt.subplot(1, 3, 3)
plt.bar(results.keys(), results.values(), color=['#3498db', '#2ecc71', '#e74c3c'])
plt.ylim(min(results.values()) - 5, 100)
plt.title('Model Başarı Kıyaslaması (%)')
plt.show()

# =================================================================
# 10. ÖZET VE SONUÇ
# =================================================================
print("\n" + "="*75 + "\n📌 HİLAL AY - PROJE DEĞERLENDİRME VE SONUÇ\n" + "="*75)
print(f"Modelimiz ev fiyatlarında ortalama {mae:.0f} $ civarında bir sapma yapmaktadır.")
print("🚀 PROJE BAŞARIYLA TAMAMLANDI!")
# ... (Önceki kodun aynen devam ettiğini varsayıyoruz, en sonuna şunları ekliyoruz) ...

# =================================================================
# 11. HEDEF DEĞİŞKEN (SALEPRICE) KORELASYON ANALİZİ
# =================================================================
# Fiyatla en güçlü bağı olan özellikleri (Sinyalleri) bulalım
fiyat_korelasyonu = df_encoded.corr()['SalePrice'].sort_values(ascending=False)

print("\n" + "="*55)
print("🚀 FİYATI EN ÇOK ETKİLEYEN İLK 10 ÖZELLİK (SİNYAL)")
print("="*55)
# SalePrice kendisiyle %100 korelasyonlu olduğu için 1. indeksten başlıyoruz
print(fiyat_korelasyonu.iloc[1:11]) 

print("\n" + "="*55)
print("📉 FİYATLA HİÇ İLGİSİ OLMAYANLAR (POTANSİYEL GÜRÜLTÜ)")
print("="*55)
# Mutlak değerce 0'a en yakın olanları bulalım
print(df_encoded.corr()['SalePrice'].abs().sort_values(ascending=True).head(10))
print("="*55)

# =================================================================
# 12. MODELİN GÖZÜNDEN FEATURE IMPORTANCE (LASSO)
# =================================================================
# Lasso, gereksiz özellikleri 0'a çekerek bize en net listeyi verir
lasso_katsayilar = pd.DataFrame({
    'Özellik': X.columns,
    'Etki Katsayısı': models["Lasso (L1)"].coef_
})

# Mutlak değerce en büyük etkili 10 özelliği sıralayalım
en_etkili_10 = lasso_katsayilar.reindex(lasso_katsayilar['Etki Katsayısı'].abs().sort_values(ascending=False).index).head(10)

print("\n" + "🌟" * 15)
print("  MODELİN KARAR MEKANİZMASI: EN ÖNEMLİ 10 ÖZELLİK")
print("🌟" * 15)
print(en_etkili_10)
print("=" * 45)

# --- GÖRSEL FEATURE IMPORTANCE GRAFİĞİ ---

plt.figure(figsize=(10, 8))
sns.barplot(x='Etki Katsayısı', y='Özellik', data=en_etkili_10, palette='viridis', hue='Özellik', legend=False)
plt.title('Lasso Regresyonuna Göre Fiyatı Belirleyen Ana Unsurlar')
plt.xlabel('Katsayı Değeri (Etki Gücü)')
plt.ylabel('Ev Özelliği')
plt.grid(axis='x', linestyle='--', alpha=0.7)
plt.show()

print("\n✅ Yorum: Grafikte görüldüğü üzere, modelimiz fiyatı belirlerken en çok " 
      f"'{en_etkili_10.iloc[0]['Özellik']}' özelliğine güvenmektedir.")
print("🚀 PROJE TÜM ANALİZLERİYLE BAŞARIYLA TAMAMLANDI!")
# =================================================================
# 3.5 HEDEF DEĞİŞKEN ANALİZİ (GÖRSELLEŞTİRME ŞARTI)
# =================================================================
plt.figure(figsize=(12, 5))

# 1. Normal Dağılım Kontrolü
plt.subplot(1, 2, 1)
sns.histplot(df['SalePrice'], kde=True, color='blue')
plt.title('Orijinal Satış Fiyatı Dağılımı')

# 2. Log Dönüşümü (Hocanın listesinde var!)
plt.subplot(1, 2, 2)
sns.histplot(np.log1p(df['SalePrice']), kde=True, color='green')
plt.title('Logaritmik Satış Fiyatı Dağılımı')

plt.tight_layout()
plt.show()

print("📢 Analiz: Orijinal verinin sağa çarpık olduğu, log dönüşümü ile daha normal dağıldığı gözlemlendi.")