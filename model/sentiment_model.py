from transformers import AutoTokenizer, AutoModelForSequenceClassification
import torch

tokenizer = AutoTokenizer.from_pretrained("dbmdz/distilbert-base-turkish-cased")

model = AutoModelForSequenceClassification.from_pretrained("./results/checkpoint-80000")

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model.to(device)
model.eval()

# 3 sınıfa göre etiketler
labels_map = {0: "Negatif", 1: "Nötr", 2: "Pozitif"}

def predict_sentiment(text):
    inputs = tokenizer(text, return_tensors="pt", truncation=True, padding=True, max_length=256).to(device)
    with torch.no_grad():
        outputs = model(**inputs)
        probs = torch.nn.functional.softmax(outputs.logits, dim=1)
        predicted = torch.argmax(probs, dim=1).item()
        return labels_map[predicted], probs.cpu().numpy()
