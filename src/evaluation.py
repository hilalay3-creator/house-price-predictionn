import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import numpy as np

def run_full_evaluation(champion_model, model_scores, y_test, y_pred, champion_name, y_original_raw):
    # Pencere boyutunu biraz daha büyütüyoruz ki her şey sığsın
    fig = plt.figure(figsize=(18, 10)) 

    # 1. Hedef Değişken Dağılımı
    ax1 = fig.add_subplot(2, 2, 1)
    sns.histplot(y_original_raw, kde=True, color='royalblue', label='Ham Fiyat', ax=ax1)
    sns.histplot(np.log1p(y_original_raw), kde=True, color='limegreen', label='Log Fiyat', ax=ax1)
    ax1.set_title('Hedef Değişken: Ham vs Log Dönüşümü', fontsize=12)
    ax1.legend()

    # 2. Tahmin vs Gerçek
    ax2 = fig.add_subplot(2, 2, 2)
    sns.scatterplot(x=y_test, y=y_pred, alpha=0.6, color='teal', ax=ax2)
    ax2.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], 'r--', lw=2)
    ax2.set_title(f'Tahmin vs Gerçek ({champion_name})', fontsize=12)

    # 3. Hata Dağılımı (Residuals)
    ax3 = fig.add_subplot(2, 2, 3)
    residuals = y_test - y_pred
    sns.residplot(x=y_pred, y=residuals, lowess=True, scatter_kws={'alpha': 0.5}, line_kws={'color': 'red'}, ax=ax3)
    ax3.set_title('Hata Dağılımı (Residuals)', fontsize=12)

    # 4. Önemli Özellikler (Buraya dikkat, ax gönderiyoruz)
    ax4 = fig.add_subplot(2, 2, 4)
    _plot_importance_subplot(champion_model, ax4)

    # GÖRÜNTÜ DÜZELTME: 
    # left=0.05: Sola iyice yasla, right=0.95: Sağa iyice yasla
    plt.subplots_adjust(left=0.08, right=0.92, top=0.92, bottom=0.08, hspace=0.3, wspace=0.4)
    
    # 2. tight_layout'u pad=2.0 ile kullanarak nesnelerin birbirine değmemesini sağlıyoruz
    plt.tight_layout(pad=2.0)
    
    # 3. Kaydederken 'bbox_inches' kullanarak dışarıda kalan yazıları da dahil etmesini garanti ediyoruz
    plt.savefig('final_analysis_plots.png', bbox_inches='tight', dpi=150)
    plt.show()

def _plot_importance_subplot(model, ax):
    # Pipeline içindeki asıl modeli ve isimleri çekme mantığı
    try:
        if 'regressor' in model.named_steps:
            regressor = model.named_steps['regressor']
            feature_names = model.named_steps['preprocessor'].get_feature_names_out()
        else:
            regressor = model.named_steps['reg']
            feature_names = model.named_steps['prep'].get_feature_names_out()
            
        if hasattr(regressor, 'feature_importances_'):
            vals = regressor.feature_importances_
        else:
            vals = np.abs(regressor.coef_)
            
        df = pd.DataFrame({'Feature': feature_names, 'Value': vals}).sort_values('Value', ascending=False).head(10)
        
        # Grafik çizimi
        sns.barplot(x='Value', y='Feature', data=df, hue='Feature', palette='viridis', legend=False, ax=ax)
        ax.set_title('Top 10 Belirleyici Özellik', fontsize=12)
        ax.set_xlabel('Etki Katsayısı')
    except Exception as e:
        ax.text(0.5, 0.5, f"Önem grafiği oluşturulamadı:\n{e}", ha='center', va='center')

def _plot_importance_subplot(model, ax):
    regressor = model.named_steps['regressor'] if 'regressor' in model.named_steps else model.named_steps['reg']
    feature_names = model.named_steps['preprocessor'].get_feature_names_out() if 'preprocessor' in model.named_steps else model.named_steps['prep'].get_feature_names_out()
    
    if hasattr(regressor, 'feature_importances_'):
        vals = regressor.feature_importances_
    else:
        vals = np.abs(regressor.coef_)
        
    df = pd.DataFrame({'Feature': feature_names, 'Value': vals}).sort_values('Value', ascending=False).head(10)
    sns.barplot(x='Value', y='Feature', data=df, palette='viridis')
    plt.title('Top 10 Belirleyici Özellik')