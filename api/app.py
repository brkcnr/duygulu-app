from fastapi import FastAPI
from pydantic import BaseModel
from transformers import AutoTokenizer, AutoModelForSequenceClassification
import torch

# 📍 Eğitilmiş modelin en son checkpoint klasörü
model_path = "./results/checkpoint-120000"
tokenizer = AutoTokenizer.from_pretrained("dbmdz/distilbert-base-turkish-cased")
model = AutoModelForSequenceClassification.from_pretrained(model_path)
model.eval()

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model.to(device)

# 🌐 API uygulaması başlat
app = FastAPI()

# Sınıf adları
labels_map = {0: "Negatif", 1: "Nötr", 2: "Pozitif"}

# JSON formatı tanımı
class InputText(BaseModel):
    text: str

# 🔮 Tahmin endpoint'i
@app.post("/predict")
def predict(input: InputText):
    inputs = tokenizer(input.text, return_tensors="pt", truncation=True, padding=True, max_length=256).to(device)
    with torch.no_grad():
        outputs = model(**inputs)
        probs = torch.nn.functional.softmax(outputs.logits, dim=1)
        predicted = torch.argmax(probs, dim=1).item()

    return {
        "text": input.text,
        "prediction": labels_map[predicted],
        "probabilities": probs.cpu().numpy().tolist()[0]
    }
