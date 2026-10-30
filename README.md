# TEXT CLASSIFICATION BATTLE
## Task 2 – Compare, Analyze and Defend

**Team Member 1:** Shabari Raj Kumar S (RA2411026050227)  
**Team Member 2:** Varshini S (RA2411026050244)

**Course:** Information Retrieval  
**Institution:** SRM Institute of Science and Technology

---

## 1. Project Overview

This project focuses on developing and comparing machine-learning models for classifying SMS messages into two categories: **Spam** and **Ham (Legitimate Messages)** using the SMS Spam Collection dataset.

The project follows a complete text-classification workflow, including dataset loading, data inspection, preprocessing, TF-IDF feature extraction, model training, performance evaluation, model comparison, and a KNN parameter experiment.

Two machine-learning algorithms, K-Nearest Neighbors (KNN) and Multinomial Naive Bayes, are implemented and compared to identify the better-performing model for SMS spam detection.

## 2. Objectives

- Load and prepare the SMS Spam Collection dataset.
- Explore the dataset and understand the distribution of spam and legitimate messages.
- Preprocess SMS text for machine learning.
- Convert message text into numerical features using TF-IDF.
- Train two classification algorithms.
- Evaluate models using accuracy, precision, recall, and F1-score.
- Compare model performance using tables, graphs, and confusion matrices.
- Investigate how changing the number of neighbors affects KNN performance.
- Identify the most suitable model based on the experimental results.

## 3. Dataset

**Dataset Name:** SMS Spam Collection  
**Source:** UCI Machine Learning Repository  
**Dataset Link:** https://archive.ics.uci.edu/dataset/228/sms+spam+collection

**Dataset File:** SMSSpamCollection

The dataset contains 5,574 labelled SMS messages.

### Dataset Columns

| Column | Description |
|---|---|
| label | Target category: spam or ham |
| message | SMS message text used as input |

### Target Categories

- **Ham:** Legitimate SMS messages.
- **Spam:** Unwanted or fraudulent promotional messages.

The dataset is used to train and evaluate machine-learning models that automatically identify spam messages.

## 4. Data Preparation and TF-IDF

The following data preparation steps were performed:

1. Loaded the SMS Spam Collection dataset into a Pandas DataFrame.
2. Inspected the dataset structure and sample messages.
3. Examined the distribution of spam and ham messages.
4. Checked for missing values and duplicate records.
5. Examined SMS message lengths.
6. Removed rows with missing labels or message text.
7. Standardized labels and removed leading and trailing whitespace from messages.
8. Retained only the expected categories: spam and ham.
9. Encoded the target labels as numerical values:
   - Ham = 0
   - Spam = 1
10. Split the dataset into training and testing sets using an 80:20 ratio.

### Train-Test Split

- Training data: 80%
- Testing data: 20%
- Random state: 42
- Stratification: Applied to preserve class proportions.

### TF-IDF Feature Extraction

TF-IDF (Term Frequency–Inverse Document Frequency) was used to convert SMS messages into numerical feature vectors.

The TF-IDF vectorizer was fitted on the training messages and then applied to the testing messages to help prevent data leakage.

## 5. Machine-Learning Models

Two classification algorithms were implemented.

| Model | Recorded Configuration |
|---|---|
| K-Nearest Neighbors (KNN) | KNeighborsClassifier(n_neighbors=5, metric='cosine', algorithm='brute') |
| Multinomial Naive Bayes | MultinomialNB() |

### K-Nearest Neighbors (KNN)

KNN classifies a message based on the classes of its nearest neighboring messages. Cosine distance was used to measure similarity between TF-IDF feature vectors.

### Multinomial Naive Bayes

Multinomial Naive Bayes is a probabilistic classification algorithm used for text classification. It predicts whether an SMS message belongs to the spam or ham category.

## 6. Evaluation Metrics

The models were evaluated using the following metrics:

- **Accuracy:** Overall proportion of correctly classified messages.
- **Precision:** How often messages predicted as spam are actually spam.
- **Recall:** How many actual spam messages are correctly identified.
- **F1-score:** Harmonic mean of precision and recall.

Confusion matrices and classification reports were also used to understand correct and incorrect predictions.

The precision, recall, and F1-score values reported in the comparison below use **spam as the positive class**.

## 7. Recorded Model Results

The following results were obtained from the recorded experiment.

| Model | Accuracy | Precision | Recall | F1-score |
|---|---:|---:|---:|---:|
| KNN (K=5) | 97.76% | 99.21% | 83.89% | 90.91% |
| Multinomial Naive Bayes | 96.05% | 100.00% | 70.47% | 82.68% |

### Result Interpretation

- KNN with K=5 achieved the highest accuracy of 97.76%.
- KNN achieved a higher recall of 83.89%, identifying more actual spam messages.
- Multinomial Naive Bayes achieved 100% precision for spam predictions.
- KNN achieved the highest F1-score of 90.91%.
- Based on the recorded results, KNN with K=5 performed better overall.

These results apply to the recorded train-test split and model configurations. Performance may differ with other datasets or configurations.

## 8. What-If Experiment: Changing K in KNN

A what-if experiment was conducted to investigate how changing the number of neighbors affects KNN classification performance.

The KNN classifier was tested using different values of K while retaining cosine distance and brute-force search.

### Experimental Results

| K | Accuracy | Precision | Recall | F1-score |
|---:|---:|---:|---:|---:|
| 3 | 96.95% | 94.57% | 81.88% | 87.77% |
| 5 | 97.76% | 99.21% | 83.89% | 90.91% |
| 7 | 97.58% | 100.00% | 81.88% | 90.04% |
| 9 | 97.22% | 99.17% | 79.87% | 88.48% |
| 11 | 97.22% | 100.00% | 79.19% | 88.39% |

### Observation

- K=5 achieved the highest accuracy among the tested values.
- K=5 also achieved the highest recall and F1-score.
- K=7 and K=11 achieved 100% precision.
- Increasing K beyond 5 did not improve overall performance.
- K=5 provided the best balance between precision and recall in this experiment.

Therefore, K=5 was identified as the preferred K value among the tested configurations.

## 9. Conclusion

This project demonstrates a machine-learning-based approach to SMS spam detection using text classification techniques.

The SMS Spam Collection dataset was processed and transformed into numerical features using TF-IDF. Two classification algorithms, K-Nearest Neighbors and Multinomial Naive Bayes, were trained and evaluated.

The recorded results showed that KNN with K=5 achieved 97.76% accuracy and a 90.91% F1-score, outperforming Multinomial Naive Bayes in overall performance. Although Naive Bayes achieved 100% precision, its lower recall indicated that it missed more actual spam messages.

The what-if experiment demonstrated that changing the number of neighbors affects KNN performance. Among the tested values, K=5 produced the best overall results.

Overall, KNN with K=5 was the strongest model in this experiment. Future improvements could include cross-validation, additional text preprocessing techniques, hyperparameter tuning, and testing on new SMS datasets.

## 10. How to Run

1. Open `SMS_Spam_Detection_ML.ipynb` in Google Colab or Jupyter Notebook.
2. The notebook downloads the SMS Spam Collection dataset directly from the UCI Machine Learning Repository.
3. Run the notebook cells sequentially, starting with the dataset download and loading section.
4. Perform data exploration and preprocessing.
5. Execute TF-IDF feature extraction.
6. Train the KNN and Multinomial Naive Bayes models.
7. Review the evaluation metrics, confusion matrix, and model comparison graph.
8. Examine the what-if experiment results for different K values.

### Requirements

- Python 3
- pandas
- scikit-learn
- matplotlib
- seaborn
- Google Colab or Jupyter Notebook

The required libraries can be installed using:

```bash
pip install pandas scikit-learn matplotlib seaborn
```

## 11. Repository

**Project:** SMS Spam Detection Using Machine Learning

**Dataset Source:** UCI Machine Learning Repository

https://archive.ics.uci.edu/dataset/228/sms+spam+collection

**Notebook:** `SMS_Spam_Detection_ML.ipynb`

