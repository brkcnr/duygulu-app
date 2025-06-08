from fastapi import FastAPI, Request, Depends
from fastapi.responses import HTMLResponse
from fastapi.templating import Jinja2Templates
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel
from transformers import AutoTokenizer, AutoModelForSequenceClassification
from sqlalchemy.orm import Session
from api.db.db import get_db
from api.db.models import Prediction
import torch

# FastAPI uygulamasını başlat
app = FastAPI()
templates = Jinja2Templates(directory="api/templates")
app.mount("/static", StaticFiles(directory="api/static"), name="static")

# Model ve tokenizer yükle
model_path = "./results/checkpoint-120000"
tokenizer = AutoTokenizer.from_pretrained("dbmdz/distilbert-base-turkish-cased")
model = AutoModelForSequenceClassification.from_pretrained(model_path)
model.eval()

# Cihaz ayarı (GPU varsa kullan)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model.to(device)

# Etiket haritası
labels_map = {0: "Negatif", 1: "Nötr", 2: "Pozitif"}

# JSON input modeli
class InputText(BaseModel):
    text: str

# Tahmin + kayıt
@app.post("/predict")
def predict_api(input: InputText, db: Session = Depends(get_db)):
    inputs = tokenizer(input.text, return_tensors="pt", truncation=True, padding=True, max_length=256).to(device)
    with torch.no_grad():
        outputs = model(**inputs)
        probs = torch.nn.functional.softmax(outputs.logits, dim=1)
        predicted = torch.argmax(probs, dim=1).item()

    record = Prediction(
        text=input.text,
        prediction=labels_map[predicted],
        prob_neg=round(probs[0][0].item(), 4),
        prob_neu=round(probs[0][1].item(), 4),
        prob_pos=round(probs[0][2].item(), 4)
    )
    db.add(record)
    db.commit()

    print(f"✔ Tahmin kaydedildi (API): id={record.id}, prediction={record.prediction}")

    return {
        "text": input.text,
        "prediction": labels_map[predicted],
        "probabilities": probs.cpu().numpy().tolist()[0]
    }

# Web sayfası
@app.get("/", response_class=HTMLResponse)
def get_form(request: Request):
    return templates.TemplateResponse("index.html", {"request": request})
