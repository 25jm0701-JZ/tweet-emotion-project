# Tweet Emotion Classification Project

This project implements a comprehensive comparison between **Traditional Machine Learning** methods and **Deep Learning (Transfer Learning)** techniques for Sentiment Analysis on the `tweet_emotions.csv` dataset.

The project is structured into two main experiments:
1.  **Experiment 1:** Text preprocessing, TF-IDF feature extraction, and traditional classifiers (Naive Bayes, Logistic Regression, XGBoost).
2.  **Experiment 2:** Utilizing a pre-trained **RoBERTa** model for feature extraction (embeddings) and full-parameter fine-tuning.

## 📂 Project Structure

```text
├── Main.py              # Entry point to run both experiments
├── Experiment1.py       # Traditional ML pipeline (TF-IDF + NB/LR/XGBoost)
├── Experiment2.py       # Deep Learning pipeline (RoBERTa + SVM/LR/Fine-tuning)
├── tweet_emotions.csv   # Input dataset (Required)
└── result/              # Output directory for plots and metrics (Auto-created)

🛠️ Dependencies
Ensure you have Python installed (3.8+ recommended). Install the required libraries using pip:
pip install numpy pandas matplotlib seaborn scikit-learn imbalanced-learn xgboost torch transformers datasets

Note: A GPU (CUDA) is highly recommended for running Experiment 2 (RoBERTa Fine-tuning) to speed up training.

How to Run
Place your dataset file tweet_emotions.csv in the root directory.

Run the main script:

Bash

python Main.py
This will automatically:

Initialize and run Experiment 1.

Pass the data context to Experiment 2 and run the Deep Learning models.

Save all visualizations and confusion matrices to the result/ folder.

🔬 Experiment Details
Experiment 1: Traditional Machine Learning
This module focuses on statistical text analysis using TF-IDF.

Preprocessing: Cleaning text (removing mentions, non-alphabetic characters), converting to lowercase.

Multi-Class Classification:

Multinomial Naive Bayes: Baseline model.

Laplace Smoothing: Optimization of Alpha parameter for Naive Bayes.

SMOTE + Logistic Regression: Handling class imbalance using Synthetic Minority Over-sampling Technique.

Binary Classification (Positive vs. Negative):

Logistic Regression: Baseline binary classifier.

XGBoost: Gradient Boosting for improved accuracy.

Experiment 2: Deep Learning (RoBERTa)
This module leverages cardiffnlp/twitter-roberta-base-emotion, a transformer model pre-trained on tweets.

Feature Extraction: * Extracts the hidden states (embeddings) from the frozen RoBERTa model.

Trains lightweight classifiers (Linear SVM and Logistic Regression) on these embeddings.

Visualization:

t-SNE: Visualizes the high-dimensional RoBERTa embeddings in 2D space to observe class separability.

Fine-Tuning:

Unfreezes the model weights and performs Full-Parameter Fine-Tuning using the Hugging Face Trainer API.

Evaluates using ROC Curves and Confusion Matrices.

📊 Outputs & Results
After running the script, check the result/ directory for the following generated images:

1_origin_distribution.png: Distribution of sentiments in the raw dataset.

2_cm_multinomial_nb.png: Confusion Matrix for Naive Bayes.

3_cm_nb_alpha_0.5.png: Confusion Matrix after Laplace Smoothing.

4_cm_lr_smote.png: Confusion Matrix for Logistic Regression with SMOTE.

5_cm_binary_lr.png: Binary Classification Confusion Matrix (LR).

6_cm_binary_xgboost.png: Binary Classification Confusion Matrix (XGBoost).

tsne_visualization.png: t-SNE plot of RoBERTa embeddings.

traditional_models_comparison.png: Comparison of SVM vs LR on RoBERTa embeddings.

finetune_confusion_matrix.png: Results after fine-tuning RoBERTa.

finetune_roc_curve.png: ROC Curve for the fine-tuned model.

📝 Acknowledgments
Dataset: Tweet Emotions Dataset.

Model: Twitter-roBERTa-base for Emotion Recognition.

Libraries: Scikit-Learn, Hugging Face Transformers, XGBoost.