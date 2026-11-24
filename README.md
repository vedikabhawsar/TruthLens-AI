# 🔍 TruthLens AI
### Machine Learning Based Fake News Detection and Classification System

TruthLens AI is a web-based Machine Learning application designed to analyze news articles and classify them as **Fake or Real** using Natural Language Processing (NLP) and Machine Learning techniques.

The project integrates a trained Machine Learning pipeline with an interactive Flask-based web application to provide automated news text classification.

## 🎯 Project Objectives

- Detect potentially fake and misleading news content.
- Apply NLP techniques for text processing.
- Train and evaluate Machine Learning classification models.
- Build an interactive web-based prediction system.
- Analyze model performance through evaluation metrics and visualizations.

## ✨ Key Features

- Fake and Real news classification.
- TF-IDF-based text feature extraction.
- Logistic Regression classification.
- Multinomial Naive Bayes model comparison.
- Interactive news analysis interface.
- Model evaluation dashboard.
- ROC curve visualization.
- Confusion matrix analysis.
- Dataset quality analysis.
- Text feature comparison experiments.
- Responsive web interface.

## 🛠️ Technology Stack

| Technology | Purpose |
|---|---|
| Python | Core programming language |
| Flask | Backend web framework |
| Scikit-learn | Machine Learning |
| TF-IDF | Text feature extraction |
| Logistic Regression | Final classification model |
| Multinomial Naive Bayes | Model comparison |
| Pandas | Data processing |
| NumPy | Numerical operations |
| HTML | Webpage structure |
| CSS | Interface styling |
| JavaScript | Frontend interactions |
| Pickle | Saving the trained model pipeline |

## ⚙️ System Workflow

1. User enters a news article.
2. Flask receives the input text.
3. The trained pipeline processes the text.
4. TF-IDF converts text into numerical features.
5. Logistic Regression predicts the category.
6. The application displays the prediction.

## 📊 Model Performance

The final Logistic Regression model achieved the following results on the test dataset:

| Evaluation Metric | Result |
|---|---:|
| Accuracy | 98.71% |
| ROC-AUC Score | 0.9988 |

These results represent performance on the project's test split. Independent external validation is a future improvement.

## 🔬 Additional Experiments

The project also includes experiments to understand model performance:

| Text Approach | Accuracy |
|---|---:|
| Headline Only | 93.59% |
| Article Body Only | 98.28% |
| Full Text | 98.64% |

A separate political news subset experiment achieved:

- Accuracy: 97.37%
- Balanced Accuracy: 96.88%
- ROC-AUC: 0.9967

A subject-only baseline experiment also revealed strong dataset-specific label patterns, highlighting the importance of checking dataset bias.

## 📁 Project Structure

```text
Fake_News_Analyzer/
│
├── MODEL/
│   ├── fake_news_pipeline.pkl
│   ├── metrics.json
│   ├── robustness_audit.json
│   └── text_feature_comparison.csv
│
├── static/
│   ├── roc_curve.png
│   ├── text_feature_comparison.png
│   ├── style.css
│   └── script.js
│
├── templates/
│   └── index.html
│
├── app.py
├── train_model.py
├── requirements.txt
├── README.md
└── .gitignore
```

## 🚀 Installation and Setup

### 1. Clone the repository

```bash
git clone https://github.com/vedikabhawsar/TruthLens-AI.git
```

### 2. Open the project folder

```bash
cd TruthLens-AI
```

### 3. Install dependencies

```bash
pip install -r requirements.txt
```

### 4. Run the application

```bash
python app.py
```

### 5. Open in browser

```text
http://127.0.0.1:5000
```

## ⚠️ Limitations

- Predictions depend on patterns learned from the training dataset.
- The system does not independently verify factual claims.
- Performance on completely unseen external datasets has not yet been established.

## 🔮 Future Scope

- URL-based news analysis.
- Multilingual fake news classification.
- Explainable AI integration.
- External fact-checking integration.
- Improved generalization using diverse datasets.

## 👩‍💻 Project Team

**Developed by:**
- Vedika Bhawsar
- Suhani

B.Tech – Data Science

---

⭐ If you find this project interesting, feel free to explore the repository!
