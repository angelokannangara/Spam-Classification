# 📧 Spam Message Classifier (ML Project)

This project is a machine learning-based **SMS spam classification system** that detects whether a given message is spam or not. It uses Natural Language Processing (NLP) techniques and a **Multinomial Naive Bayes** classifier. Built and tested using **Google Colab**.

---

## 📂 Dataset

- **Source**: [UCI SMS Spam Collection Dataset](https://archive.ics.uci.edu/ml/datasets/sms+spam+collection)
- Total messages: **5,572**
- Classes: `ham` (legitimate) and `spam` (unwanted)

---

## 🔧 Project Workflow

### 1. **Data Cleaning & Preprocessing**
- Dropped unnecessary columns
- Renamed columns (`v1 → class`, `v2 → sms`)
- Removed duplicates
- Added a `length` feature
- Visualized text length distribution

### 2. **Text Processing**
- Lowercasing
- Tokenization
- Removing punctuation and stopwords
- Stemming using `PorterStemmer`
- Final cleaned messages stored in `sms_cleaned`

### 3. **Feature Engineering**
- TF-IDF Vectorization (`max_features = 3000`)
- Transformed cleaned messages into numerical feature vectors

### 4. **Model Training**
- Split data into training and testing (80:20)
- Used **Multinomial Naive Bayes** classifier

### 5. **Evaluation**
- Achieved an accuracy of **97.09%** on the test set

---

## 📊 Model Performance

| Metric     | Value     |
|------------|-----------|
| Accuracy   | 97.09%    |
| Algorithm  | MultinomialNB |
| Features   | 3000 (TF-IDF) |

---

## 🔍 Sample Prediction

```python
message = "Congratulations! You've won a free ticket. Call now!"
cleaned = clean_text(message)
vectorized = tf_vec.transform([cleaned])
prediction = model.predict(vectorized)
print(prediction)  # Output: ['spam']
