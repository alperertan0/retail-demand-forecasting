import pandas as pd
import numpy as np
import warnings
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import Ridge
from lightgbm import LGBMRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

# Uyarıları gizlemek için
warnings.filterwarnings('ignore')

print("1. Veri yükleniyor ve tarih formatı düzenleniyor...")
# Veri setini yükle
df = pd.read_csv('retail_store_inventory.csv')

# Tarih sütununu datetime formatına çevir ve sırala
df['Date'] = pd.to_datetime(df['Date'])
df = df.sort_values(by=['Store ID', 'Product ID', 'Date']).reset_index(drop=True)

print("2. Eksik veriler (Imputation) iş mantığına göre dolduruluyor...")
# Stok ve satışları önceki/sonraki günün verisiyle doldur
df['Inventory Level'] = df.groupby(['Store ID', 'Product ID'])['Inventory Level'].transform(lambda x: x.ffill().bfill())
df['Units Sold'] = df.groupby(['Store ID', 'Product ID'])['Units Sold'].transform(lambda x: x.ffill().bfill())

# Fiyatları ürün bazında medyan ile doldur
df['Price'] = df.groupby('Product ID')['Price'].transform(lambda x: x.fillna(x.median()))
df['Competitor Pricing'] = df.groupby('Product ID')['Competitor Pricing'].transform(lambda x: x.fillna(x.median()))

# İndirim ve Promosyon eksiklerini 0 ile doldur
df['Discount'] = df['Discount'].fillna(0)
df['Promotion'] = df['Promotion'].fillna(0)

# Kategorik eksikleri en sık tekrar eden (Mod) ile doldur
for col in ['Category', 'Region', 'Weather Condition', 'Seasonality']:
    df[col] = df[col].fillna(df[col].mode()[0])

print("3. Yeni değişkenler (Feature Engineering) türetiliyor...")
df['Effective_Price'] = df['Price'] * (1 - df['Discount'] / 100.0)
df['Price_Diff_Comp'] = df['Effective_Price'] - df['Competitor Pricing']
df['DayOfWeek'] = df['Date'].dt.dayofweek
df['Month'] = df['Date'].dt.month

# Kategorik değişkenleri sayısal formata çevir (Label Encoding)
cat_cols = ['Store ID', 'Product ID', 'Category', 'Region', 'Weather Condition', 'Seasonality']
label_encoders = {}
for col in cat_cols:
    le = LabelEncoder()
    df[col + '_Enc'] = le.fit_transform(df[col].astype(str))
    label_encoders[col] = le

# Modele girecek özellikler (Features)
features = [
    'Inventory Level', 'Units Sold', 'Price', 'Discount',
    'Promotion', 'Competitor Pricing', 'Effective_Price', 'Price_Diff_Comp',
    'DayOfWeek', 'Month', 'Store ID_Enc', 'Product ID_Enc', 'Category_Enc',
    'Region_Enc', 'Weather Condition_Enc', 'Seasonality_Enc'
]

X = df[features]
y = df['Demand'] # Tahmin etmek istediğimiz hedef değişken

print("4. Veri eğitim ve test setlerine ayrılıyor...")
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

print("5. Modeller eğitiliyor ve sonuçlar karşılaştırılıyor...\n")

# Model 1: LightGBM
lgb_model = LGBMRegressor(n_estimators=150, max_depth=10, random_state=42)
lgb_model.fit(X_train, y_train)
lgb_pred = lgb_model.predict(X_test)

# Model 2: Random Forest
rf_model = RandomForestRegressor(n_estimators=100, max_depth=12, random_state=42, n_jobs=-1)
rf_model.fit(X_train, y_train)
rf_pred = rf_model.predict(X_test)

# Model 3: Ridge (Baseline Doğrusal Model)
ridge_model = Ridge()
ridge_model.fit(X_train, y_train)
ridge_pred = ridge_model.predict(X_test)

def print_metrics(model_name, y_true, y_pred):
    mae = mean_absolute_error(y_true, y_pred)
    rmse = np.sqrt(mean_squared_error(y_true, y_pred))
    r2 = r2_score(y_true, y_pred)
    print(f"--- {model_name} Sonuçları ---")
    print(f"R2 Skoru (Başarı) : %{r2 * 100:.2f}")
    print(f"MAE (Sapma Adedi) : {mae:.2f} adet\n")

print_metrics("LightGBM Regressor", y_test, lgb_pred)
print_metrics