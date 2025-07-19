# 📩 Spam Detection using BERT (HuggingFace Transformers)

This project fine-tunes a BERT-based model (`bert-base-uncased`) on the **SMS Spam Collection Dataset** to classify text messages as **ham (legit)** or **spam**. It achieves **~92% validation accuracy**.

---

## 🚀 Features

- ✅ Fine-tuned `bert-base-uncased` model on real SMS spam data
- ✅ Achieves 92%+ accuracy in spam classification
- ✅ Visualizes dataset distribution and word clouds
- ✅ Predicts new examples using saved model
- ✅ Modular scripts for training, prediction, and visualization

---

## 🧠 Model

- **Model**: `bert-base-uncased` from Hugging Face
- **Framework**: PyTorch + HuggingFace Transformers
- **Tokenizer**: `BertTokenizer`
- **Optimizer**: AdamW
- **Epochs**: 3
- **Batch size**: 8
- **Max length**: 128 tokens

---

## 📈 Training Results

| Metric      | Value   |
|-------------|---------|
| Accuracy    | 92.3%   |
| Eval Acc    | 91.9%   |
| Eval Loss   | 0.32    |

-------
Install dependencies

pip install torch transformers scikit-learn matplotlib wordcloud pandas

Train the model

python spam_detection_training.py

------
You’ll see:

📊 Bar plot of spam vs ham distribution

☁️ Word cloud of spam messages

📦 Example Predictions
Message	Predicted Label
"Hey, are you coming to the meeting?"	ham
"Congratulations! You've won a free iPhone. Click now!"	spam
"URGENT! Your account is blocked. Click the link ASAP."	spam

