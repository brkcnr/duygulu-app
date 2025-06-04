import pandas as pd
from model.sentiment_model import predict_sentiment

# CSV yükle
df = pd.read_csv("data/train.csv")  # doğru klasöre göre güncelle

# Sayısal etiket yerine sınıf adı göstermek için map
label_map_str = {0: "Negatif", 1: "Nötr", 2: "Pozitif"}
label_map_text = {"Negative": 0, "Notr": 1, "Positive": 2}

# Etiketleri sayıya çevir (eğer hâlâ metin olarak geliyorsa)
if df["label"].dtype == object:
    df["label"] = df["label"].map(label_map_text)

# Test için rastgele 20 örnek seç
df = df.sample(20, random_state=42).reset_index(drop=True)

# Sınıflandırma başlasın
for i in range(len(df)):
    text = df.loc[i, "text"]
    gerçek_label = df.loc[i, "label"]
    tahmin, olasılıklar = predict_sentiment(text)

    print(f"Metin: {text}")
    print(f"Tahmin: {tahmin} | Gerçek Etiket: {label_map_str.get(gerçek_label, '??')}")
    print(f"Olasılıklar: {olasılıklar}")
    print("-" * 60)
