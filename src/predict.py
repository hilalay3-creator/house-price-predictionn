import pandas as pd
import joblib
import numpy as np
from model import inverse_log_transform

def make_predictions():
    print("📦 Model yükleniyor...")
    # Eğitilen modeli yüklüyoruz
    model = joblib.load('final_model.joblib')
    
    print("📄 Test verisi okunuyor...")
    test_df = pd.read_csv('../data/test.csv')
    
    # Modelin beklediği Id kolonunu ayırıyoruz (Submission için lazım olacak)
    test_ids = test_df['Id']
    X_test = test_df.drop(['Id'], axis=1)
    
    print("🔮 Tahminler üretiliyor...")
    # Model pipeline olduğu için preprocessing işlemlerini otomatik yapacak
    predictions_log = model.predict(X_test)
    
    # Log-target modeling yaptığımız için geri çeviriyoruz (Hocanın istediği kritik nokta)
    predictions = inverse_log_transform(predictions_log)
    
    # 2. ADIM: Submission Dosyası Oluşturma (Kaggle Formatı)
    submission = pd.DataFrame({
        'Id': test_ids,
        'SalePrice': predictions
    })
    
    submission.to_csv('../submission.csv', index=False)
    print("✅ Tahminler başarıyla üretildi ve 'submission.csv' olarak kaydedildi!")

if __name__ == "__main__":
    make_predictions()