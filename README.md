# ✨ Spam vs Ham Classifier using Machine Learning

> 📬 A smart spam detection system that classifies messages as **Ham (Safe)** or **Spam (Unwanted)** using NLP and supervised ML algorithms.

---

![Spam Detection Demo](https://media.giphy.com/media/QBd2kLB5qDmysEXre9/giphy.gif)  
<sub>📽️ *Demo: Message being classified in action (replace with your own if needed)*</sub>

---

## 📦 Project Summary

| Key Aspect        | Details                                                      |
|------------------|--------------------------------------------------------------|
| 🧠 Algorithm      | Multinomial Naive Bayes / Logistic Regression                |
| 📚 Dataset        | [SMS Spam Collection](https://www.kaggle.com/datasets/uciml/sms-spam-collection-dataset) |
| 🧼 Preprocessing   | Lowercasing, stopword removal, tokenization                 |
| 🔤 Vectorization   | TF-IDF / CountVectorizer                                    |
| 💾 Model Files     | `model.pkl`, `vectorizer.pkl`                               |
| 🎯 Accuracy        | ~95% on test data                                           |

---

## ⚙️ Features

- ✅ Detects spam messages in real-time
- 🧠 Pre-trained ML model using clean NLP pipeline
- 💬 Input a custom message and get instant classification
- 🔐 Prevents spam in chat, SMS, or form submissions

---

## Input/Output

Input: "Congratulations! You've won a free ticket."
Prediction: 🚫 Spam

## graph LR
A[Input Text Message] --> B[Preprocessing]
B --> C[Vectorizer (TF-IDF)]
C --> D[Trained Classifier Model]
D --> E{Prediction: Ham or Spam}

##💡 Real-World Use Cases
📱 SMS spam blocking

📨 Email inbox filtering

💬 Chat moderation bots

🛑 Form spam prevention

## ✍️ Author
| [![Anu's GitHub](https://avatars.githubusercontent.com/anu4552?s=80)](https://github.com/anu4552) |
| :-----------------------------------------------------------------------------------------------: |
|                                           **Anu Kumari**                                          |
|                            [GitHub Profile](https://github.com/anu4552)                           |


