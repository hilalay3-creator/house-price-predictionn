import numpy as np
from xgboost import XGBRegressor
from sklearn.model_selection import GridSearchCV
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LinearRegression, Lasso, Ridge
from sklearn.metrics import r2_score, mean_absolute_error, mean_squared_error

def get_best_model(preprocessor):
    # Ana Pipeline: Preprocessor + XGBoost
    full_pipeline = Pipeline(steps=[
        ('preprocessor', preprocessor),
        ('regressor', XGBRegressor(objective='reg:squarederror', random_state=42))
    ])

    param_grid = {
        'regressor__n_estimators': [100, 500],
        'regressor__learning_rate': [0.05, 0.1],
        'regressor__max_depth': [3, 5]
    }

    return GridSearchCV(full_pipeline, param_grid, cv=5, scoring='r2', n_jobs=-1)

def get_baseline_scores(X_train, X_test, y_train_log, y_test_original, preprocessor):
    # Her bir temel modeli kendi preprocessor'ı ile bir Pipeline içine alıyoruz
    # Bu sayede NaN hataları (imputation sayesinde) otomatik çözülür.
    models = {
        "Linear Reg": Pipeline([('prep', preprocessor), ('reg', LinearRegression())]),
        "Lasso (Tuned)": GridSearchCV(Pipeline([('prep', preprocessor), ('reg', Lasso(max_iter=10000))]), 
                                     {'reg__alpha': [0.01, 0.1, 1, 10]}, cv=5),
        "Ridge (Tuned)": GridSearchCV(Pipeline([('prep', preprocessor), ('reg', Ridge())]), 
                                     {'reg__alpha': [0.1, 1, 10, 100]}, cv=5)
    }
    
    scores = {}
    trained_pipelines = {}
    
    for name, model in models.items():
        # Artık model.fit dediğimizde önce preprocessor çalışır, NaN'lar dolar, sonra eğitim başlar.
        model.fit(X_train, y_train_log)
        
        preds_log = model.predict(X_test)
        preds = np.expm1(preds_log)
        
        r2 = r2_score(y_test_original, preds)
        mae = mean_absolute_error(y_test_original, preds)
        rmse = np.sqrt(mean_squared_error(y_test_original, preds))
        
        scores[name] = {'R2': r2 * 100, 'MAE': mae, 'RMSE': rmse}
        trained_pipelines[name] = model if name == "Linear Reg" else model.best_estimator_
        
    return scores, trained_pipelines

def apply_log_transform(y): return np.log1p(y)
def inverse_log_transform(y_log): return np.expm1(y_log)
def generate_predictions(model, test_df):
    # Pipeline sayesinde preprocessing adımları otomatik uygulanır
    preds_log = model.predict(test_df)
    
    # Logaritmik sonuçları dolar birimine çeviriyoruz
    return inverse_log_transform(preds_log)


# get_best_model fonksiyonu sadece XGBoost için GridSearchCV döndürüyor, diğer modeller için ayrı ayrı GridSearchCV'ler oluşturduk. get_baseline_scores fonksiyonu ise Linear Regression, Lasso ve Ridge modellerini kendi içinde eğitip değerlendirdikten sonra skorlarını ve eğitimli pipeline'larını döndürüyor. Böylece main.py içinde tüm modellerin skorlarını tek bir yerden alabiliyoruz. Ayrıca, her modelin kendi pipeline'ı içinde preprocessor olduğu için NaN hatalarıyla karşılaşmıyoruz.  