# 💳 Credit Card Fraud Detection Using Logistic Regression

A machine learning project to detect fraudulent transactions from anonymized credit-card data.  
The pipeline covers **EDA**, **preprocessing**, **(re)balancing**, **training**, and **comprehensive evaluation** using **Logistic Regression**.


---

## 🚀 Features

- 📊 **Exploratory Data Analysis (EDA):** `head/info/describe`, unique counts, duplicates & missing checks  
- 🧼 **Preprocessing:** Outlier handling (IQR on `Time`), duplicate removal, optional feature scaling  
- ⚖️ **Class Balance Insight:** Value counts + pie/bar charts, optional **downsampling** to balance classes  
- 🔗 **Correlation & Distributions:** Heatmap, KDE plots, boxen plots for `Time` & `Amount`  
- ✂️ **Train/Test Split:** Stratified split to preserve class ratios  
- 🤖 **Model:** **Logistic Regression** (with `max_iter=1000` or class weights if needed)  
- 📈 **Evaluation:** Accuracy, precision, recall, F1-score, confusion matrix
- 🧪 **Predict New Samples:** Helper to score new transactions

---

## 🧠 Workflow

1. **Import libraries:** Pandas, NumPy, Matplotlib, Seaborn, Scikit-learn  
2. **Load dataset:** `creditcard.csv` → DataFrame `credit_data`  
3. **Quick look:** `head/tail`, `shape`, `info`, `describe`, `nunique`  
4. **Data quality:** Missing values, duplicates, optional outlier filtering (IQR on `Time`)  
5. **Class exploration:** `Class` value counts + pie/bar charts  
6. **(Re)balancing (optional):** Downsample majority (`Class=0`) to match minority (`Class=1`) → `creditData`  
7. **EDA visuals:** Boxen plots (`Time`, `Amount`), KDEs, correlation heatmap  
8. **Split:** `X`/`y` + `train_test_split(test_size=0.3, random_state=42, stratify=y)`  
9. **Scale (optional):** `StandardScaler` for features (esp. `Amount`, `Time`)  
10. **Train:** `LogisticRegression(max_iter=1000, random_state=42)` (or `class_weight='balanced'`)  
11. **Evaluate:** Accuracy, confusion matrix, classification report
12. **Predict:** Helper to score new transactions

---

## 🛠️ Tech Stack

| Layer       | Technology         |
|------------|--------------------|
| Language   | Python 3.10+       |
| Framework  | Scikit-learn       |
| Dataset    | `creditcard.csv`   |
| Tools      | NumPy, Pandas, Matplotlib, Seaborn |

---

## 📁 Project Structure

```text
├── Data/
│   └── creditcard.csv                  ← Main dataset
│   └── Credit Card Fraud Detection.ipynb
└── README.md                           ← This documentation
