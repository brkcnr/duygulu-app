import torch
from transformers import AutoTokenizer, AutoModelForSequenceClassification

model_name = "dbmdz/distilbert-base-turkish-cased"
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

print("CUDA durumu:", torch.cuda.is_available())
if device.type == "cuda":
    print("GPU:", torch.cuda.get_device_name(0))

try:
    # Tokenizer ve model
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForSequenceClassification.from_pretrained(model_name, num_labels=3)
    model.to(device)

    # Sahte metin üret
    text = "Bu ürün harika!" * 64  # 256 tokenlık metin

    # Tokenize et
    inputs = tokenizer(text, return_tensors="pt", padding='max_length', truncation=True, max_length=256)
    inputs = {k: v.to(device) for k, v in inputs.items()}

    # İleri besleme simülasyonu
    with torch.no_grad():
        outputs = model(**inputs)

    print("✅ Model GPU üzerinde başarıyla çalıştı.")
except RuntimeError as e:
    print("❌ GPU belleği yetersiz:", e)
