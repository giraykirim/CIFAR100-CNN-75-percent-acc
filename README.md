# CIFAR‑100 (Coarse) – CNN from Scratch

## Giriş
Bu projede **CIFAR‑100** veri setinin **coarse (20 süper sınıf)** etiketleriyle, sıfırdan tasarladığım bir **Konvolüsyonel Sinir Ağı (CNN)** eğittim. 
Notebooks içinde (ör. `projeee.ipynb`) her kod hücresinin önünde yer alan Markdown açıklamalarında teknik ayrıntıları adım adım anlattım; bu README ise **hangi veri setini seçtiğim**, **hangi yöntemleri uyguladığım** ve **elde ettiğim sonuçları** **kısa** ve **derli toplu** şekilde özetler.

- Veri seti: `keras.datasets.cifar100` (`label_mode='coarse'`), 50.000 eğitim / 10.000 test görüntüsü (32×32, RGB).
- Amaç: 20 süper sınıfta görüntü sınıflandırma.
- Yaklaşım: Veri artırma + L2 düzenlileştirme + artan dropout ile derin CNN; Adam optimize edici.
- Sonuç (özet): **Top‑1 test doğruluğu ≈ 0.75**; sınıf bazında dengeli performans, karışıklık matrisi ve sınıflandırma raporuyla doğrulandı.

> Projenin **teknik anlatımı** (veri hazırlama, mimari kararlar, eğitim ve değerlendirme adımları) **notebook içindeki Markdown hücrelerinde** yer alır.

---

## Yöntem / Mimari
- **Veri artırma (yalnız eğitimde):** `RandomFlip`, `RandomRotation(0.1)`, `RandomZoom(0.1)`.
- **CNN gövde:** 3 konvolüsyon bloğu (filtre sayıları **64 → 128 → 256**), her konv katmanı sonrası **BatchNorm + ReLU**, blok sonunda **MaxPooling(2×2)** ve **Dropout** (**0.2 / 0.3 / 0.4**). Ağırlık başlatma: **He normal**.
- **Düzenlileştirme:** Tüm konv katmanlarında **L2 (weight decay)**.
- **Tepe (head):** `GlobalAveragePooling2D` → `Dense(512) + BN + ReLU + Dropout(0.5)` → `Dense(num_classes, softmax)`.
- **Kayıp ve optimizasyon:** `SparseCategoricalCrossentropy`, **Adam** (`lr=1e-3`).
- **Callback’ler:** 
  - `ModelCheckpoint(monitor='val_accuracy', save_best_only=True)`
  - `ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=5)`
  - `EarlyStopping(monitor='val_loss', patience=20, restore_best_weights=True)`

### Eğitim Kurulumu
- `BATCH_SIZE = 256`, `EPOCHS = 120`, `LEARNING_RATE = 1e-3`, `WEIGHT_DECAY = 1e-4`
- `validation_split = 0.1`, `shuffle = True`
- Model ağırlıkları: `cifar100_cnn_scratch.h5` ve en iyi doğrulama için `best_cifar100_cnn_scratch.h5`

---

## Metrikler ve Sonuçlar
Eğitim sonunda elde ettiğim başlıca metrikler:
- **Test doğruluğu:** ~ **0.75**
- **Test kaybı:** ~ **1.00**
- **Sınıflandırma raporu:** makro ve ağırlıklı **F1 ≈ 0.75** (20 sınıf, destek=10.000)
- **Karışıklık matrisi:** ilk 20 sınıf için ısı haritası (notebook’ta görselleştirme mevcut).
- **Eğri analizleri:** Eğitim/validasyon doğrulukları ~**40. epoch** civarında plato yapıyor; `ReduceLROnPlateau` tetiklendikçe öğrenme oranı düşürülerek iyileşme sürüyor. Eğitim ve validasyon eğrileri arasındaki fark **ılımlı**, aşırı uyum sınırlı.

**Yorumum:** 32×32 çözünürlük ve 20 süper sınıf bağlamında %75 civarı doğruluk, sıfırdan CNN için güçlü bir taban çizgisi. Bazı sınıflar arası karışıklıklar (benzer görünümler/arka planlar) matris üzerinde gözleniyor; veri artırmanın çeşitlendirilmesi ve eğitim stratejilerinin iyileştirilmesiyle daha yüksek performans mümkün.

---

## Ekler
- **Grad‑CAM:** Modelin karar verdiği bölgeleri açıklamak için Grad‑CAM ısı haritası üretip, orijinal görüntü üzerine şeffaf bindirme yaptım (örnek görselleştirme notebook’ta).
- **Kaynak kod / eğitim tekrar üretimi:** Tüm adımlar notebook’ta sıralı; `Run All` ile aynı sonuçlara ulaşılabilir.
- **(Opsiyonel) UI / Deployment:** İstenirse `Streamlit` veya `Gradio` ile basit bir arayüz eklenip `best_*.h5` modeliyle çevrim‑içi demo hazırlanabilir (ör. `ui/app.py`). GPU üzerinde uçtan uca çalışma için Colab / Kaggle ortamı uygundur.

**Ortam gereksinimleri (öneri):**
```bash
python>=3.10
pip install tensorflow matplotlib numpy pandas scikit-learn opencv-python
```

---

## Sonuç ve Gelecek Çalışmalar
Kısa vadede aşağıdaki adımların doğruluğu artırması beklenir:
- **Augmentasyon:** `AutoAugment`/`RandAugment`, **MixUp/CutMix**, **ColorJitter**.
- **Eğitim stratejisi:** **Cosine LR decay + warm‑up**, **Label Smoothing (ε≈0.1)**, daha uzun eğitim (ör. 200‑300 epoch) ve **ema** ağırlık ortalaması.
- **Mimari:** Artık bağlantılar (Residual bloklar), **DropBlock**, daha geniş/derin varyant, `weight_decay`/dropout hassas ayarı.
- **Değerlendirme:** **Test‑time augmentation**, sınıf isimleriyle detaylı hata analizi.
- **Mühendislik:** `ONNX/TFLite` dönüştürme, hafifletilmiş çıkarım, **Streamlit/Gradio** ile arayüz, **Docker** ile paketleme.

> Bu proje, bootcamp sonrasında da geliştirilmeye açık olacak şekilde tasarlandı. Yeni veri toplama yöntemleri, gerçek zamanlı çıkarım ve farklı model aileleri (ör. Vision Transformer) ileride eklenecek çalışmalar arasındadır.

---

## Kaggle Linkleri
İşte notebookumun olduğu kaggle linki
https://www.kaggle.com/code/giraykrm/cifar100-cnn
