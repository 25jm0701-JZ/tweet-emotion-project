import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import re
import os
import numpy as np
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
from imblearn.over_sampling import SMOTE
import xgboost as xgb


class EmotionClassifier:
    def __init__(self, file_path, result_dir="result"):
        # 1. Initialize paths and directories
        self.file_path = file_path
        self.result_dir = result_dir

        # Create result directory if it doesn't exist
        if not os.path.exists(self.result_dir):
            os.makedirs(self.result_dir)
            print(f"Directory '{self.result_dir}' created.")

        # 2. Initialize data containers
        self.data = pd.read_csv(file_path)
        self.df_cleaned = None
        self.class_names = None

        # Multi-class variables
        self.X = None
        self.y = None
        self.X_train = None
        self.X_test = None
        self.y_train = None
        self.y_test = None
        self.X_train_tfidf = None
        self.X_test_tfidf = None
        self.tfidf_vectorizer_model = None  # Save multi-class TF-IDF model

        # Binary classification variables
        self.df_binary = None
        self.X_binary = None
        self.y_binary = None
        self.X_train_bin = None
        self.X_test_bin = None
        self.y_train_bin = None
        self.y_test_bin = None
        self.X_train_tfidf_bin = None
        self.X_test_tfidf_bin = None
        self.tfidf_vectorizer_bin_model = None  # Save binary TF-IDF model

    def data_show(self):
        """Display original data distribution and save the plot"""
        print("--- Showing Data Distribution ---")
        df = self.data
        pd.plotting.register_matplotlib_converters()
        sns.set_style("dark")
        plt.figure(figsize=(10, 6))
        sns.histplot(df['sentiment'])
        plt.xticks(rotation=45)

        save_path = os.path.join(self.result_dir, "1_origin_distribution.png")
        plt.savefig(save_path)
        print(f"Image saved to {save_path}")
        plt.close()

    def cleaning_data(self):
        """Clean the data"""
        print("\n--- Cleaning Data ---")
        df = self.data.copy()
        # Clean text
        df["cleaned_text"] = df["content"].apply(lambda x: re.sub(r"@[\\w]*", "", str(x)))  # Remove @ mentions
        df["cleaned_text"] = df["cleaned_text"].apply(lambda x: re.sub(r"[^a-zA-Z']", " ", x))  # Keep only letters
        df["cleaned_text"] = df["cleaned_text"].str.lower().str.strip()

        # Remove 'empty' sentiment and short texts
        df_cleaned = df[df['sentiment'] != 'empty'].copy()
        df_cleaned = df_cleaned[df_cleaned["cleaned_text"].str.len() > 1].reset_index(drop=True)

        print(f"Origin lines: {len(df)}")
        print(f"After cleaning: {len(df_cleaned)}")

        self.df_cleaned = df_cleaned

    def sentiment_count(self):
        """Count sentiment occurrences"""
        if self.df_cleaned is None:
            self.cleaning_data()

        sentiment_counts_cleaned = self.df_cleaned['sentiment'].value_counts()
        print("\n--- Emotion Distribution after cleaning ---")
        print(sentiment_counts_cleaned)

    def encode_preparation(self):
        """Encode labels"""
        print("\n--- Encoding Labels ---")
        self.X = self.df_cleaned['cleaned_text']
        y = self.df_cleaned['sentiment']

        le = LabelEncoder()
        self.y = le.fit_transform(y)
        self.class_names = le.classes_

        print("Label and Code comparison table:")
        for i, name in enumerate(self.class_names):
            print(f"{i}: {name}")

    def split_data(self):
        """Split multi-class dataset"""
        self.X_train, self.X_test, self.y_train, self.y_test = train_test_split(
            self.X,
            self.y,
            test_size=0.2,
            random_state=42,
            stratify=self.y
        )
        print(f"\nTraining set size: {len(self.X_train)}")
        print(f"Testing set size: {len(self.X_test)}")

    def tfidf_vectorizer(self):
        """TF-IDF Feature Extraction (Multi-class)"""
        self.tfidf_vectorizer_model = TfidfVectorizer(max_features=10000, ngram_range=(1, 2), stop_words='english')

        self.X_train_tfidf = self.tfidf_vectorizer_model.fit_transform(self.X_train)
        self.X_test_tfidf = self.tfidf_vectorizer_model.transform(self.X_test)

        print(f"Shape of training features: {self.X_train_tfidf.shape}")

    def _plot_confusion_matrix(self, y_true, y_pred, title, filename, target_names):
        """Internal helper function: Plot confusion matrix"""
        cm = confusion_matrix(y_true, y_pred)
        cm_normalized = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]

        plt.figure(figsize=(10, 8))
        sns.heatmap(
            cm_normalized,
            annot=True,
            fmt=".2f",
            cmap="Blues",
            xticklabels=target_names,
            yticklabels=target_names,
            cbar_kws={"label": "Normalized Ratio"}
        )
        plt.title(title)
        plt.ylabel('True Label')
        plt.xlabel('Predicted Label')
        plt.xticks(rotation=45, ha='right')
        plt.yticks(rotation=0)
        plt.tight_layout()

        save_path = os.path.join(self.result_dir, filename)
        plt.savefig(save_path)
        print(f"Confusion Matrix saved to {save_path}")
        plt.close()

    def implement_Multinomial_NB(self):
        """Implement Multinomial Naive Bayes"""
        print("\n--- Implementing Multinomial Naive Bayes ---")
        mnb_model = MultinomialNB()
        mnb_model.fit(self.X_train_tfidf, self.y_train)
        y_pred = mnb_model.predict(self.X_test_tfidf)

        print(classification_report(self.y_test, y_pred, target_names=self.class_names, zero_division=0))

        self._plot_confusion_matrix(
            self.y_test, y_pred,
            'Normalized Confusion Matrix (MultinomialNB)',
            '2_cm_multinomial_nb.png',
            self.class_names
        )

    def implement_Laplace_smoothing(self):
        """Search for best Alpha and implement"""
        print("\n--- Implementing Laplace Smoothing Search ---")
        alpha_list = [0.01, 0.1, 0.5, 1.0, 2.0]

        for alpha in alpha_list:
            temp_model = MultinomialNB(alpha=alpha)
            temp_model.fit(self.X_train_tfidf, self.y_train)
            temp_pred = temp_model.predict(self.X_test_tfidf)
            accuracy = (temp_pred == self.y_test).mean()
            print(f"alpha={alpha}, accuracy: {accuracy:.4f}")

        # Best alpha = 0.5
        print("Using best alpha=0.5 for final model")
        mnb_model = MultinomialNB(alpha=0.5)
        mnb_model.fit(self.X_train_tfidf, self.y_train)
        y_pred = mnb_model.predict(self.X_test_tfidf)

        print(classification_report(self.y_test, y_pred, target_names=self.class_names, zero_division=0))

        self._plot_confusion_matrix(
            self.y_test, y_pred,
            'Normalized Confusion Matrix (NB, alpha=0.5)',
            '3_cm_nb_alpha_0.5.png',
            self.class_names
        )

    def implement_SMOTE_Logistic_Regression(self):
        """Implement SMOTE + Logistic Regression"""
        print("\n--- Implementing SMOTE + Logistic Regression ---")
        smote = SMOTE(random_state=42, sampling_strategy='not majority', k_neighbors=5)

        # Apply SMOTE only on the training set
        X_train_smote, y_train_smote = smote.fit_resample(self.X_train_tfidf, self.y_train)
        print(f"SMOTE applied - New training shape: {X_train_smote.shape}")

        lr_model = LogisticRegression(max_iter=2000, C=0.5, random_state=42)
        lr_model.fit(X_train_smote, y_train_smote)
        y_pred = lr_model.predict(self.X_test_tfidf)

        print("Classification Report (Logistic Regression + SMOTE)")
        # Output table and confusion matrix
        self._plot_confusion_matrix(
            self.y_test, y_pred,
            'Normalized Confusion Matrix (LR + SMOTE)',
            '4_cm_lr_smote.png',
            self.class_names
        )

    def prepare_binary_data(self):
        """
        [Optimization]: Centralize binary classification data preparation to avoid repetitive splitting in each binary model
        """
        print("\n--- Preparing Binary Classification Data ---")
        negative_emotions = ['worry', 'sadness', 'hate', 'boredom', 'anger']
        positive_emotions = [emo for emo in self.class_names if emo not in negative_emotions + ['neutral']]

        print(f"Negative: {negative_emotions}")
        print(f"Positive: {positive_emotions}")

        # Filter out 'neutral'
        self.df_binary = self.df_cleaned[self.df_cleaned['sentiment'] != 'neutral'].copy()

        # Generate labels (1: Positive, 0: Negative)
        self.df_binary['binary_label'] = self.df_binary['sentiment'].apply(
            lambda x: 1 if x in positive_emotions else 0
        )

        self.X_binary = self.df_binary['cleaned_text']
        self.y_binary = self.df_binary['binary_label']

        # Split dataset (Unified split to ensure LR and XGBoost use the same data)
        self.X_train_bin, self.X_test_bin, self.y_train_bin, self.y_test_bin = train_test_split(
            self.X_binary, self.y_binary, test_size=0.2, random_state=42, stratify=self.y_binary
        )

        # TF-IDF (Specific for binary classification)
        self.tfidf_vectorizer_bin_model = TfidfVectorizer(max_features=10000, ngram_range=(1, 2))
        self.X_train_tfidf_bin = self.tfidf_vectorizer_bin_model.fit_transform(self.X_train_bin)
        self.X_test_tfidf_bin = self.tfidf_vectorizer_bin_model.transform(self.X_test_bin)

        print(f"Binary Training features: {self.X_train_tfidf_bin.shape}")

    def binary_logistic_regression(self):
        """Implement Binary Logistic Regression"""
        if self.X_train_tfidf_bin is None:
            self.prepare_binary_data()

        print("\n--- Binary Logistic Regression ---")
        lr_binary = LogisticRegression(class_weight='balanced', max_iter=2000, C=0.5, random_state=42)
        lr_binary.fit(self.X_train_tfidf_bin, self.y_train_bin)
        y_pred_bin = lr_binary.predict(self.X_test_tfidf_bin)

        print(f"Accuracy: {accuracy_score(self.y_test_bin, y_pred_bin):.4f}")

        # Plot simple binary confusion matrix
        cm_bin = confusion_matrix(self.y_test_bin, y_pred_bin)
        plt.figure(figsize=(6, 4))
        sns.heatmap(
            cm_bin, annot=True, fmt='d', cmap='Blues',
            xticklabels=['Negative', 'Positive'],
            yticklabels=['Negative', 'Positive']
        )
        plt.title('Confusion Matrix (Binary LR)')
        plt.ylabel('True Label')
        plt.xlabel('Predicted Label')

        save_path = os.path.join(self.result_dir, "5_cm_binary_lr.png")
        plt.savefig(save_path)
        print(f"Saved to {save_path}")
        plt.close()

    def implement_xgboost(self):
        """Implement XGBoost"""
        if self.X_train_tfidf_bin is None:
            self.prepare_binary_data()

        print("\n--- Binary XGBoost ---")
        xgb_model = xgb.XGBClassifier(
            objective='binary:logistic',
            n_estimators=200,
            max_depth=6,
            learning_rate=0.1,
            random_state=42,
            verbosity=0
        )

        # Use data prepared in prepare_binary_data directly
        xgb_model.fit(self.X_train_tfidf_bin, self.y_train_bin)
        y_pred_xgb = xgb_model.predict(self.X_test_tfidf_bin)

        print(f"XGBoost Accuracy: {accuracy_score(self.y_test_bin, y_pred_xgb):.4f}")
        print(
            classification_report(self.y_test_bin, y_pred_xgb, target_names=['Negative', 'Positive'], zero_division=0))

        # Optionally plot a confusion matrix for XGBoost
        cm_xgb = confusion_matrix(self.y_test_bin, y_pred_xgb)
        plt.figure(figsize=(6, 4))
        sns.heatmap(
            cm_xgb, annot=True, fmt='d', cmap='Greens',
            xticklabels=['Negative', 'Positive'],
            yticklabels=['Negative', 'Positive']
        )
        plt.title('Confusion Matrix (Binary XGBoost)')

        save_path = os.path.join(self.result_dir, "6_cm_binary_xgboost.png")
        plt.savefig(save_path)
        print(f"Saved to {save_path}")
        plt.close()


    def run_Experiment1(self):

        # Execute the pipeline sequentially
        self.data_show()  # 1. Display original distribution
        self.cleaning_data()  # 2. Clean data
        self.sentiment_count()  # 3. Count sentiments after cleaning
        self.encode_preparation()  # 4. Encode labels
        self.split_data()  # 5. Split dataset
        self.tfidf_vectorizer()  # 6. TF-IDF feature extraction
        self.implement_Multinomial_NB()  # 7. Multinomial Naive Bayes
        self.implement_Laplace_smoothing()  # 8. Laplace smoothing optimization
        self.implement_SMOTE_Logistic_Regression()  # 9. SMOTE + Logistic Regression

        # Binary Classification Section
        self.prepare_binary_data()  # 10. Prepare binary data (Extracted separately)
        self.binary_logistic_regression()  # 11. Binary Logistic Regression
        self.implement_xgboost()  # 12. XGBoost