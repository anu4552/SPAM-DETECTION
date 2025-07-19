import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from datasets import Dataset
from transformers import AutoTokenizer, AutoModelForSequenceClassification, Trainer, TrainingArguments
import torch

# 1. Load Dataset
url = "https://raw.githubusercontent.com/justmarkham/pycon-2016-tutorial/master/data/sms.tsv"
df = pd.read_csv(url, sep='\t', names=["label", "text"])

# 2. Encode Labels
le = LabelEncoder()
df["label_enc"] = le.fit_transform(df["label"])

# 3. Train-test Split
train_texts, val_texts, train_labels, val_labels = train_test_split(
    df["text"], df["label_enc"], test_size=0.2, random_state=42
)

# 4. Load Tokenizer & Tokenize
model_name = "microsoft/MiniLM-L12-H384-uncased"
tokenizer = AutoTokenizer.from_pretrained(model_name)

def tokenize(example):
    return tokenizer(example["text"], truncation=True, padding="max_length", max_length=128)

train_ds = Dataset.from_dict({"text": train_texts.tolist(), "label": train_labels.tolist()})
val_ds = Dataset.from_dict({"text": val_texts.tolist(), "label": val_labels.tolist()})

train_ds = train_ds.map(tokenize, batched=True)
val_ds = val_ds.map(tokenize, batched=True)

train_ds.set_format(type="torch", columns=["input_ids", "attention_mask", "label"])
val_ds.set_format(type="torch", columns=["input_ids", "attention_mask", "label"])

# 5. Load Model
model = AutoModelForSequenceClassification.from_pretrained(model_name, num_labels=2)

# 6. Train
training_args = TrainingArguments(
    output_dir="./results",
    evaluation_strategy="epoch",
    per_device_train_batch_size=16,
    per_device_eval_batch_size=16,
    num_train_epochs=5,
    logging_dir="./logs",
    save_total_limit=1,
    load_best_model_at_end=True,
    metric_for_best_model="eval_loss"
)

def compute_metrics(eval_pred):
    logits, labels = eval_pred
    preds = torch.argmax(torch.tensor(logits), axis=1)
    acc = (preds == torch.tensor(labels)).float().mean()
    return {"accuracy": acc.item()}

trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=train_ds,
    eval_dataset=val_ds,
    compute_metrics=compute_metrics
)

train_result = trainer.train()


# 7. Prediction on Custom Text
def predict(texts):
    inputs = tokenizer(texts, return_tensors="pt", padding=True, truncation=True, max_length=128)
    outputs = model(**inputs)
    probs = torch.softmax(outputs.logits, dim=1)
    preds = torch.argmax(probs, axis=1)
    for t, p in zip(texts, preds):
        label = le.inverse_transform([p.item()])[0]
        print(f"Message: \"{t}\" --> Prediction: {label}")

# 8. Test Prediction
sample_texts = [
    "Congratulations! You've won a free iPhone. Click here to claim.",
    "Hey, are we still meeting for lunch tomorrow?",
    "Limited offer! Buy now and get 50% off!",
    "Your appointment is confirmed for 3 PM today."
]
predict(sample_texts)
