import pandas as pd
import numpy as np
import joblib
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score, mean_absolute_error, mean_squared_error
from preprocessing import get_preprocessor, clean_outliers
from model import get_best_model, apply_log_transform, inverse_log_transform, get_baseline_scores
from evaluation import run_full_evaluation

def run_pipeline():
    # 1-5: Veri Hazırlama (Log Target Dahil)
    df = pd.read_csv('../data/train.csv')
    y_raw = df['SalePrice'] # Grafik için ham hali tutuyoruz
    df = clean_outliers(df)
    X = df.drop(['SalePrice', 'Id'], axis=1)
    y_log = apply_log_transform(df['SalePrice']) # Log dönüşümü uyguluyoruz
    
    X_train, X_test, y_train_log, y_test_log = train_test_split(X, y_log, test_size=0.2, random_state=42)
    y_test_original = inverse_log_transform(y_test_log) # Değerlendirme için gerçek fiyatları tutuyoruz

    # 6-7: XGBoost GridSearchCV (Pipeline İçinde)
    numeric_cols = X.select_dtypes(exclude='object').columns.tolist() # Sayısal sütunları belirliyoruz
    categorical_cols = X.select_dtypes(include='object').columns.tolist()
    preprocessor = get_preprocessor(numeric_cols, categorical_cols)
    
    print("\n🚀 XGBoost - Lineer Regresyon - Lasso - Ridge Yarışıyor...")
    xgb_grid = get_best_model(preprocessor) # GridSearchCV döndürüyor, fit ederken tüm pipeline çalışacak
    xgb_grid.fit(X_train, y_train_log)
    
    # XGBoost Metrikleri
    xgb_preds = inverse_log_transform(xgb_grid.predict(X_test)) 
    xgb_metrics = {
        'R2': r2_score(y_test_original, xgb_preds) * 100,
        'MAE': mean_absolute_error(y_test_original, xgb_preds),
        'RMSE': np.sqrt(mean_squared_error(y_test_original, xgb_preds)) 
    }

    # 10. TÜM MODELLERİN METRİK TABLOSU
    model_scores, all_pipelines = get_baseline_scores(X_train, X_test, y_train_log, y_test_original, preprocessor) # Baseline modellerin skorlarını alıyoruz
    model_scores["XGBoost (Best)"] = xgb_metrics
    all_pipelines["XGBoost (Best)"] = xgb_grid.best_estimator_

    # Terminal Dashboard
    print("\n" + "═"*65)
    print(f"{'MODEL PERFORMANS TABLOSU':^65}")
    print("─"*65)
    print(f"{'Model Name':<20} | {'R2 (%)':>10} | {'MAE ($)':>12} | {'RMSE ($)':>12}") 
    print("─"*65)
    
    sorted_ranking = sorted(model_scores.items(), key=lambda x: x[1]['R2'], reverse=True)
    for name, m in sorted_ranking: # R2'ye göre sıralayıp yazdırıyoruz
        print(f"{name:<20} | {m['R2']:>10.2f} | {m['MAE']:>12.2f} | {m['RMSE']:>12.2f}")
    
    champion_name = sorted_ranking[0][0] # En yüksek R2'ye sahip modeli şampiyon ilan ediyoruz
    champion_model = all_pipelines[champion_name]
    print("═"*65)
    print(f"🏆 ŞAMPİYON: {champion_name}".center(65))
    print("═"*65 + "\n")

    # Görselleştirme ve Kayıt
    run_full_evaluation(champion_model, model_scores, y_test_original, xgb_preds, champion_name, y_raw)
    joblib.dump(champion_model, 'final_model.joblib') # Şampiyon modeli kaydediyoruz 
    print(f"✅ Model kaydedildi.")
    
    # Şampiyonu fonksiyon dışına gönderiyoruz
    return champion_model, champion_name

if __name__ == "__main__":
    # Fonksiyonu çalıştır ve dönen şampiyonu yakala
    champion_model, champion_name = run_pipeline() 
    
    # 🏁 11. SUBMISSION DOSYASI OLUŞTURMA
    try:
        from model import generate_predictions
        import pandas as pd
        
        print("\n📝 Yarışma tahminleri oluşturuluyor...")
        test_path = '../data/test.csv'
        test_df = pd.read_csv(test_path)
        
        final_preds = generate_predictions(champion_model, test_df)
        
        submission = pd.DataFrame({
            'Id': test_df['Id'],
            'SalePrice': final_preds
        })
        
        submission.to_csv('../submission.csv', index=False)
        print(f"✅ Başarılı! 'submission.csv' {champion_name} modeliyle oluşturuldu.")
        
    except Exception as e:
        print(f"⚠️ Submission oluşturulamadı: {e}")