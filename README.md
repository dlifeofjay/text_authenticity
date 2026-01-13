# 📰 Fake News Detection System

A machine learning pipeline for detecting fake news and misinformation using NLP techniques.

![Python](https://img.shields.io/badge/Python-3.8+-blue?logo=python)
![Scikit-learn](https://img.shields.io/badge/Scikit--learn-ML-orange)
![NLP](https://img.shields.io/badge/NLP-Text_Classification-green)

---

## 📋 Overview

This project builds a **text authenticity classifier** that:
- Detects fake news articles with high accuracy
- Uses TF-IDF vectorization for text feature extraction
- Implements ensemble classification methods

---

## 🚀 Quick Start

```bash
# Clone the repository
git clone https://github.com/dlifeofjay/text_authenticity.git
cd text_authenticity

# Install dependencies
pip install -r requirements.txt

# Run inference
python text_aut.py
```

---

## 📁 Project Structure

```
text_authenticity/
├── Fake News Detector.ipynb  # Training & analysis
├── text_aut.py               # Inference script
├── text_aut.joblib           # Trained model
├── text_cv.joblib            # Count vectorizer
├── text_OrdEnc.joblib        # Ordinal encoder
└── requirements.txt          # Dependencies
```

---

## 🔬 Methodology

1. **Text Preprocessing**: Cleaning, tokenization, stopword removal
2. **Feature Extraction**: TF-IDF / Count Vectorization
3. **Classification**: Ensemble methods for robust detection
4. **Evaluation**: Accuracy, precision, recall, F1-score

---

## 📊 Results

The model achieves strong performance in distinguishing authentic news from fabricated content.

---

## 👨‍💻 Author

**Jubril Ifekoya** - Data Scientist & ML Engineer
