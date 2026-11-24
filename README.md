
# 🔍 TruthLens AI
### Machine Learning Based Fake News Classification System

TruthLens AI is a web-based machine learning application that analyzes news text and classifies it into Fake and Real categories using Natural Language Processing (NLP) and Machine Learning.

The project combines a trained machine learning model with an interactive web interface to demonstrate automated news text classification.

---

## ✨ Key Features

- Fake and Real news classification.
- Machine Learning model integration.
- TF-IDF based text feature extraction.
- Model confidence score.
- Interactive news analysis interface.
- Professional dark-themed dashboard.
- Model performance analytics.
- Confusion matrix visualization.
- Recent prediction history.
- Responsive user interface.
- Flask-based web application.

---

## 🛠️ Technology Stack

| Technology | Purpose |
|---|---|
| Python | Core programming |
| Flask | Web application backend |
| Scikit-learn | Machine Learning |
| TF-IDF | Text feature extraction |
| Logistic Regression | Classification algorithm |
| Multinomial Naive Bayes | Model comparison |
| Pandas | Dataset processing |
| HTML | Webpage structure |
| CSS | User interface design |
| JavaScript | Frontend functionality |
| Joblib | Model serialization |

---

## 📁 Project Structure

```text
Fake_News_Analyzer/
│
├── DATASET/
│   ├── Fake.csv
│   └── True.csv
│
├── MODEL/
│   ├── fake_news_pipeline.pkl
│   └── metrics.json
│
├── templates/
│   └── index.html
│
├── static/
│   ├── style.css
│   └── script.js
│
├── app.py
├── requirements.txt
├── README.md
└── .gitignore
