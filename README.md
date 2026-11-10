# 🛒 Perakende Mağaza Envanteri ve Talep Tahmini (Retail Demand Forecasting)

Bu proje, perakende sektöründeki en kritik iki sorun olan **stoksuz kalma (stockout)** ve **fazla stok (overstock)** maliyetlerini en aza indirmek amacıyla geliştirilmiş bir makine öğrenmesi (Regression) modelidir. 

Gerçek dünya dinamiklerini simüle eden bir veri seti kullanılarak; hava durumu, tatil/kampanya günleri ve rakip fiyatlandırması gibi dış etkenlerin satış hacmine etkisi analiz edilmiş ve gelecekteki talep miktarı (`Demand`) yüksek doğrulukla tahmin edilmiştir.

## 🎯 İş Problemi ve Çözüm Yaklaşımı
Perakende mağazalarında ürünlerin ne kadar satılacağını öngörememek, ya müşteri kaybına (boş raf) ya da nakit akışının kilitlenmesine (atıl stok) yol açar. Bu projede:
- **Karar Destek Sistemi:** Sipariş adetlerinin manuel tahminler yerine veri odaklı algoritmalarla belirlenmesi hedeflenmiştir.
- **Dinamik Fiyatlandırma Analizi:** İndirim oranları ve rakip fiyatların satışa etkisi modellenmiştir.

## 🛠️ Kullanılan Teknolojiler
- **Dil:** Python
- **Kütüphaneler:** Pandas, NumPy, Scikit-learn, LightGBM
- **Algoritmalar:** LightGBM Regressor, Random Forest Regressor, Ridge Regression

## 📊 Veri Ön İşleme ve Öznitelik Mühendisliği (Feature Engineering)
Veri setindeki eksik değerler, veri sızıntısını (data leakage) önlemek adına iş mantığına uygun doldurulmuştur:
- **Zaman Serisi Bütünlüğü:** Stok ve satış eksikleri, geçmiş ve gelecek günlerin (ffill/bfill) değerleriyle tamamlandı.
- **Fiyatlandırma:** Eksik fiyatlar, ilgili ürünün medyan değeriyle dolduruldu.

**Modele Güç Katan Yeni Değişkenler:**
1. `Effective_Price`: İndirim oranı düşüldükten sonra müşterinin karşılaştığı net satış fiyatı.
2. `Price_Diff_Comp`: Net satış fiyatımız ile rakip mağaza fiyatı arasındaki rekabet makası.
3. `DayOfWeek` & `Month`: Satışlardaki mevsimsel ve haftalık trendleri yakalamak için tarih ayrıştırması.

## 📈 Model Performansı ve Sonuçlar
Projede farklı algoritmalar test edilmiş ve en yüksek başarıyı **LightGBM Regressor** göstermiştir:
- **R² Skoru:** %85.88 (Modelin hedef değişkeni açıklama oranı)
- **MAE (Ortalama Mutlak Hata):** 13.55 adet (Günlük satış tahminindeki ortalama sapma)

Modelin karar mekanizmasındaki en önemli değişkenler (Feature Importance) sırasıyla geçmiş satış hızı, net fiyat ve rakip fiyat farkı olarak öne çıkmıştır.

## 🚀 Kurulum ve Çalıştırma
Projeyi kendi bilgisayarınızda test etmek için aşağıdaki adımları izleyebilirsiniz:

1. Repoyu bilgisayarınıza klonlayın:
   ```bash
   git clone [https://github.com/KULLANICI_ADIN/retail-demand-forecasting.git](https://github.com/KULLANICI_ADIN/retail-demand-forecasting.git)
