import pandas as pd
from datasets import Dataset
from sklearn.model_selection import train_test_split
from transformers import AutoTokenizer, AutoModelForSequenceClassification, TrainingArguments, Trainer
import torch

# 📂 CSV Yolu
df = pd.read_csv("data/train.csv")

# Etiketleri sayıya çevir
label_map = {"Negative": 0, "Notr": 1, "Positive": 2}
df["label"] = df["label"].map(label_map)

# 🔀 Eğitim ve doğrulama seti
train_df, val_df = train_test_split(df, test_size=0.2, stratify=df["label"], random_state=42)

train_dataset = Dataset.from_pandas(train_df)
val_dataset = Dataset.from_pandas(val_df)

# 🔧 Model ve Tokenizer
model_name = "dbmdz/distilbert-base-turkish-cased"
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForSequenceClassification.from_pretrained(
    model_name,
    num_labels=3,
    ignore_mismatched_sizes=True  # Eğer modelin son katmanı uyumsuzsa
)

#Cihaz kontrolü
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model.to(device)

# 🧪 Tokenization
def tokenize(batch):
    return tokenizer(batch["text"], padding='max_length', truncation=True, max_length=256)

train_dataset = train_dataset.map(tokenize, batched=True)
val_dataset = val_dataset.map(tokenize, batched=True)

train_dataset.set_format("torch", columns=["input_ids", "attention_mask", "label"])
val_dataset.set_format("torch", columns=["input_ids", "attention_mask", "label"])

# ⚙️ Eğitim Parametreleri
training_args = TrainingArguments(
    output_dir="./results",
    evaluation_strategy="epoch",
    save_strategy="epoch",
    per_device_train_batch_size=2,
    per_device_eval_batch_size=2,
    num_train_epochs=3,
    weight_decay=0.01,
    logging_dir="./logs",
    load_best_model_at_end=True,
    metric_for_best_model="eval_loss",
    save_total_limit=2
)

# 🧠 Trainer
trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=train_dataset,
    eval_dataset=val_dataset
)

# 🚀 Eğitimi Başlat
trainer.train()
