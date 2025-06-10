from fastapi import FastAPI, Request, Depends, Query
from fastapi.responses import HTMLResponse, RedirectResponse
from fastapi.templating import Jinja2Templates
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel
from transformers import AutoTokenizer, AutoModelForSequenceClassification
from sqlalchemy.orm import Session
from api.db.db import get_db
from api.db.models import Prediction
from api.db.base import Base
from api.db.db import engine
import torch

# Veritabanı tablolarını oluştur
Base.metadata.create_all(bind=engine)

# FastAPI uygulamasını başlat
app = FastAPI()
templates = Jinja2Templates(directory="api/templates")
app.mount("/static", StaticFiles(directory="api/static"), name="static")

# Model ve tokenizer yükle
model_path = "./results/checkpoint-80000"
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

    return {
        "text": input.text,
        "prediction": labels_map[predicted],
        "probabilities": probs.cpu().numpy().tolist()[0]
    }

# Web sayfası
@app.get("/", response_class=HTMLResponse)
def get_form(request: Request):
    return templates.TemplateResponse("index.html", {"request": request})

@app.get("/result", response_class=HTMLResponse)
def list_predictions(
    request: Request,
    db: Session = Depends(get_db),
    page: int = Query(1, ge=1),
    q: str = Query("", alias="q"),
    label: str = Query("", alias="label")
):
    page_size = 10
    offset = (page - 1) * page_size

    query = db.query(Prediction)

    if q:
        query = query.filter(Prediction.text.ilike(f"%{q}%"))
    if label:
        query = query.filter(Prediction.prediction == label)

    total = query.count()
    predictions = (
        query.order_by(Prediction.id.desc())
        .offset(offset)
        .limit(page_size)
        .all()
    )

    total_pages = (total + page_size - 1) // page_size

    return templates.TemplateResponse("result.html", {
        "request": request,
        "predictions": predictions,
        "page": page,
        "total_pages": total_pages,
        "q": q,
        "label": label
    })
    
# Tahmin silme
@app.post("/sil/{prediction_id}")
def delete_prediction(prediction_id: int, db: Session = Depends(get_db)):
    record = db.query(Prediction).get(prediction_id)
    if record:
        db.delete(record)
        db.commit()
    return RedirectResponse(url="/result", status_code=303)