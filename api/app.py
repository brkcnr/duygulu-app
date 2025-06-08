from fastapi import FastAPI, Request, Form
from fastapi.responses import HTMLResponse
from fastapi.templating import Jinja2Templates
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel
from transformers import AutoTokenizer, AutoModelForSequenceClassification
import torch

# 🌐 FastAPI başlat
app = FastAPI()
templates = Jinja2Templates(directory="api/templates")
app.mount("/static", StaticFiles(directory="api/static"), name="static")

# 📍 Model yükle
model_path = "./results/checkpoint-120000"
tokenizer = AutoTokenizer.from_pretrained("dbmdz/distilbert-base-turkish-cased")
model = AutoModelForSequenceClassification.from_pretrained(model_path)
model.eval()

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model.to(device)

# Etiketler
labels_map = {0: "Negatif", 1: "Nötr", 2: "Pozitif"}

# JSON input modeli
class InputText(BaseModel):
    text: str

# 🔮 JSON endpoint (POST /predict)
@app.post("/predict")
def predict_api(input: InputText):
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

# 🖼️ HTML form arayüzü (GET /)
@app.get("/", response_class=HTMLResponse)
def get_form(request: Request):
    return templates.TemplateResponse("index.html", {"request": request, "result": None})

# 📩 HTML form POST (formdan gelen metni tahmin et)
@app.post("/", response_class=HTMLResponse)
def post_form(request: Request, text: str = Form(...)):
    inputs = tokenizer(text, return_tensors="pt", truncation=True, padding=True, max_length=256).to(device)
    with torch.no_grad():
        outputs = model(**inputs)
        probs = torch.nn.functional.softmax(outputs.logits, dim=1)
        predicted = torch.argmax(probs, dim=1).item()

    result = {
        "text": text,
        "prediction": labels_map[predicted],
        "probabilities": [round(p, 4) for p in probs.cpu().numpy().tolist()[0]]
    }
    return templates.TemplateResponse("index.html", {"request": request, "result": result})
