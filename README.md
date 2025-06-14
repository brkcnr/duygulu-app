# Duygulu-App: Türkçe Metin Duygu Analizi Uygulaması

## İçindekiler
- Proje Hakkında
- Özellikler
- Teknoloji Yığını
- Kurulum
- Kullanım
- Proje Yapısı

## Proje Hakkında

Duygulu-App, Türkçe metinlerin duygu analizi için geliştirilmiş bir web uygulamasıdır. DistilBERT tabanlı bir derin öğrenme modeli kullanarak metinleri pozitif, nötr veya negatif olarak sınıflandırır ve her bir kategori için olasılık değerlerini gösterir.

## Özellikler

- Türkçe metinlerde duygu analizi
- Kullanıcı dostu web arayüzü
- Yapılan tahminlerin veritabanında saklanması
- Geçmiş tahminleri görüntüleme ve filtreleme
- Docker ile kolay kurulum ve dağıtım

## Kullanılan Teknolojiler

- **Backend**: Python, FastAPI
- **Frontend**: HTML, JavaScript, TailwindCSS
- **Yapay Zeka**: Transformers, PyTorch
- **Veritabanı**: PostgreSQL
- **Deployment**: Docker, Docker Compose

## Kurulum

### Ön Koşullar
- Docker ve Docker Compose
- Git

### Kurulum Adımları

1. Repoyu klonlayın:
   ```bash
   git clone https://github.com/kullanici/duygulu-app.git
   cd duygulu-app
   ```

2. .env dosyasını oluşturun:
   ```bash
   cp .env.example .env
   ```
   ve gerekli değişkenleri ayarlayın.
    ```bash
   DATABASE_URL=VERİTABANI_URL
   POSTGRES_USER=KULLANICI_ADIN
   POSTGRES_PASSWORD=ŞİFREN
   POSTGRES_DB=VERİTABANI_İSMİ
   ```

3. Docker ile uygulamayı başlatın:
   ```bash
   docker-compose up -d
   ```

4. Uygulama http://localhost:8000 adresinde çalışmaya başlayacaktır.

## Kullanım

1. Tarayıcınızda http://localhost:8000 adresine gidin
2. Metin giriş kutusuna analiz etmek istediğiniz Türkçe metni girin
3. "Tahmin Et" butonuna tıklayın
4. Sonuçlar ekranda görüntülenecektir:
   - Tahmin edilen duygu (Pozitif, Nötr, Negatif)
   - Her bir duygu kategorisi için olasılık değerleri
5. Önceki tahminleri görmek için "Tahminleri Göster" linkine tıklayın
6. Tahmin geçmişinde arama yapabilir, belirli bir duygu durumuna göre filtreleyebilir veya tahminleri silebilirsiniz

## Proje Yapısı

```
duygulu-app/
├── api/                # FastAPI uygulaması
│   ├── db/             # Veritabanı modelleri ve bağlantıları
│   ├── static/         # JavaScript dosyaları
│   └── templates/      # HTML şablonları
├── data/               # Veri dosyaları
├── model/              # Model tanımlamaları
├── results/            # Eğitilmiş model dosyaları
├── .dockerignore
├── .env                # Ortam değişkenleri
├── docker-compose.yml  # Docker Compose yapılandırması
├── Dockerfile          # Docker yapılandırması
├── fine_tune.py        # Model ince ayar kodu
├── main.py             # Test betikleri
└── requirements.txt    # Python bağımlılıkları
```