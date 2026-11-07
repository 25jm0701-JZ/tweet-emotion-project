import numpy as np
import pandas as pd
import torch
from datasets import Dataset
from matplotlib import pyplot as plt
from sklearn.linear_model import LogisticRegression
from sklearn.manifold import TSNE
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, f1_score, roc_curve, auc
from sklearn.model_selection import train_test_split
from sklearn.svm import LinearSVC
import seaborn as sns
from transformers import AutoTokenizer, AutoModel, AutoModelForSequenceClassification, TrainingArguments, Trainer

# Assuming Experiment1 is in the same directory
from Experiment1 import EmotionClassifier


class roberta_EmotionClassifier(EmotionClassifier):
    # Modification 1: Changed parameter name to 'data_source' to avoid conflict with class name
    def __init__(self, data_source):
        # If the parent class has initialization logic, call super().__init__()
        # super().__init__()

        self.model = None
        self.trainer = None  # Initialize trainer container
        self.test_tokenized_ds = None
        self.train_tokenized_ds = None

        # Inherit data
        self.data = data_source.df_binary
        self.class_names = data_source.class_names

        self.MODEL_CKPT = "cardiffnlp/twitter-roberta-base-emotion"
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        # Data splitting
        # Note: Ensure random_state is consistent with Experiment1 for fair comparison
        self.train_df, self.test_df = train_test_split(
            self.data,
            test_size=0.2,
            random_state=42,
            stratify=self.data['binary_label']
        )
        self.raw_train_ds = None
        self.raw_test_ds = None

        # Initialize tokenizer
        self.tokenizer = AutoTokenizer.from_pretrained(self.MODEL_CKPT)

    def huggingface_format_trans(self):  # Fixed typo: formate -> format
        """Convert Pandas DataFrame to Hugging Face Dataset format"""
        self.raw_train_ds = Dataset.from_pandas(self.train_df[['content', 'binary_label']].reset_index(drop=True))
        self.raw_test_ds = Dataset.from_pandas(self.test_df[['content', 'binary_label']].reset_index(drop=True))

    def tokenize(self):
        """Tokenize data and prepare the base model for feature extraction"""
        self.model = AutoModel.from_pretrained(self.MODEL_CKPT).to(self.device)

        def tokenize_function(batch):
            # [Core Fix] Changed padding=True to padding='max_length' and specified max_length.
            # This ensures all sentences are uniformly padded to 128 tokens, creating a consistent matrix
            # and preventing "Ragged Tensor" errors during batch processing.
            return self.tokenizer(
                batch["content"],
                padding="max_length",
                max_length=128,
                truncation=True
            )

        # 1. Execute tokenization
        train_tokenized_ds = self.raw_train_ds.map(tokenize_function, batched=True)
        test_tokenized_ds = self.raw_test_ds.map(tokenize_function, batched=True)

        # 2. Assign to instance variables
        self.train_tokenized_ds = train_tokenized_ds
        self.test_tokenized_ds = test_tokenized_ds

        # Note: Do NOT use set_format("torch") here. Keep data as Python Lists
        # to avoid type conflicts in the subsequent mapdata function.
        print("Tokenized Training Data Shape:", len(self.train_tokenized_ds))

    def mapdata(self):
        """Extract Hidden States for traditional machine learning (SVM/LR)"""

        # 1. Define extraction function
        def extract_hidden_states(batch):
            device = self.device
            tokenizer = self.tokenizer

            inputs = {}
            for k, v in batch.items():
                if k in tokenizer.model_input_names:
                    # [Core Fix] Since 'v' is currently a pure Python list, convert directly using torch.tensor.
                    # This is stable and avoids AttributeError related to 'list object has no attribute to'.
                    inputs[k] = torch.tensor(v).to(device)

            with torch.no_grad():
                last_hidden_state = self.model(**inputs).last_hidden_state

            # Extract [CLS] token (index 0)
            return {"hidden_state": last_hidden_state[:, 0].cpu().numpy()}

        print("--- Step 4: Extracting Hidden State Features (Mode: Python List -> Tensor) ---")

        # 2. Parallel extraction
        train_features = self.train_tokenized_ds.map(
            extract_hidden_states,
            batched=True,
            batch_size=64
        )

        test_features = self.test_tokenized_ds.map(
            extract_hidden_states,
            batched=True,
            batch_size=64
        )

        # 3. Assemble Numpy arrays
        self.X_train = np.array(train_features["hidden_state"])
        self.X_test = np.array(test_features["hidden_state"])
        self.y_train = np.array(train_features["binary_label"])
        self.y_test = np.array(test_features["binary_label"])

        # 4. [Important Addition] Feature extraction complete. Now convert Dataset to Torch format.
        # This is required for the subsequent fine_tuning_method (Trainer) to work correctly.
        columns_to_keep = ["input_ids", "attention_mask", "binary_label"]
        self.train_tokenized_ds.set_format("torch", columns=columns_to_keep)
        self.test_tokenized_ds.set_format("torch", columns=columns_to_keep)

        # 5. Clear GPU memory
        del self.model
        torch.cuda.empty_cache()

    def tsne_visualization(self):  # Fixed typo: t_sen -> tsne
        """t-SNE Visualization"""
        subset_size = 500
        actual_size = min(len(self.X_test), subset_size)

        tsne = TSNE(n_components=2, random_state=42)
        X_embedded = tsne.fit_transform(self.X_test[:actual_size])

        plt.figure(figsize=(10, 8))
        sns.scatterplot(
            x=X_embedded[:, 0],
            y=X_embedded[:, 1],
            hue=self.y_test[:actual_size],
            palette="deep",
            s=60
        )
        plt.title("t-SNE Visualization of Sentiment Vectors")
        # Suggestion: Comment out plt.show() if you don't want blocking popups during batch runs
        plt.savefig("result/tsne_visualization.png")
        plt.show()

    def traditional_method(self):
        """Compare traditional machine learning methods"""
        print("--- Running Traditional Machine Learning Models (SVM vs LR) ---")

        # Logistic Regression
        lr_binary = LogisticRegression(class_weight='balanced', max_iter=2000, random_state=42)
        lr_binary.fit(self.X_train, self.y_train)
        y_pred_lr = lr_binary.predict(self.X_test)

        # Linear SVC
        svm_model = LinearSVC(C=0.1, max_iter=10000, random_state=42, dual='auto')  # dual='auto' suppresses warning
        svm_model.fit(self.X_train, self.y_train)
        y_pred_svm = svm_model.predict(self.X_test)

        accuracy_svm = accuracy_score(self.y_test, y_pred_svm)
        accuracy_lr = accuracy_score(self.y_test, y_pred_lr)

        print("=" * 30 + "\n🤖 Linear SVC (SVM) Report\n" + "=" * 30)
        print(classification_report(self.y_test, y_pred_svm))
        print("\n" + "=" * 30 + "\n📈 Logistic Regression Report\n" + "=" * 30)
        print(classification_report(self.y_test, y_pred_lr))

        # Plot confusion matrices
        cm_svm = confusion_matrix(self.y_test, y_pred_svm)
        cm_lr = confusion_matrix(self.y_test, y_pred_lr)

        fig, axes = plt.subplots(1, 2, figsize=(16, 6))

        sns.heatmap(cm_svm, annot=True, fmt='d', cmap='Blues', ax=axes[0])
        axes[0].set_title(f'Linear SVC\nAccuracy: {accuracy_svm:.4f}')
        axes[0].set_xlabel('Predicted')
        axes[0].set_ylabel('Actual')

        sns.heatmap(cm_lr, annot=True, fmt='d', cmap='Greens', ax=axes[1])
        axes[1].set_title(f'Logistic Regression\nAccuracy: {accuracy_lr:.4f}')
        axes[1].set_xlabel('Predicted')
        axes[1].set_ylabel('Actual')

        plt.tight_layout()
        plt.savefig("result/traditional_models_comparison.png")
        plt.show()

        # Output scoreboard
        results_df = pd.DataFrame({
            'Model': ['Linear SVC', 'Logistic Regression'],
            'Accuracy': [accuracy_svm, accuracy_lr],
            'Type': ['Linear Margin', 'Probabilistic']
        })
        print("\n🏆 Scoreboard:")
        print(results_df)

    def fine_tuning_method(self):
        """Full Parameter Fine-Tuning"""
        print("--- Starting Fine-Tuning ---")
        torch.cuda.empty_cache()  # Clear memory again to ensure space for Trainer

        ft_train_ds = self.train_tokenized_ds.rename_column("binary_label", "labels")
        ft_test_ds = self.test_tokenized_ds.rename_column("binary_label", "labels")

        def compute_metrics(pred):
            labels = pred.label_ids
            preds = np.argmax(pred.predictions, axis=1)
            acc = accuracy_score(labels, preds)
            f1 = f1_score(labels, preds, average='weighted')
            return {"accuracy": acc, "f1": f1}

        num_labels = len(np.unique(self.data['binary_label']))  # Safer way to get label count

        # Reload model with classification head
        model = AutoModelForSequenceClassification.from_pretrained(
            self.MODEL_CKPT,
            num_labels=num_labels,
            ignore_mismatched_sizes=True
        ).to(self.device)

        bs = 64
        logging_steps = len(ft_train_ds) // bs
        model_name = f"{self.MODEL_CKPT.split('/')[-1]}-finetuned-emotion"

        training_args = TrainingArguments(
            output_dir=model_name,
            num_train_epochs=3,
            learning_rate=2e-5,
            per_device_train_batch_size=bs,
            per_device_eval_batch_size=bs,
            weight_decay=0.01,
            eval_strategy="epoch",
            disable_tqdm=False,
            report_to="none",
            logging_steps=logging_steps,
            log_level="error",
            save_strategy="no"
        )

        self.trainer = Trainer(
            model=model,
            args=training_args,
            compute_metrics=compute_metrics,
            train_dataset=ft_train_ds,
            eval_dataset=ft_test_ds,
            tokenizer=self.tokenizer
        )

        self.trainer.train()
        self.ft_test_ds = ft_test_ds

    def result_display(self):
        """Display Fine-Tuning Results"""
        if self.trainer is None:
            print("Error: Please run fine_tuning_method() first.")
            return

        print("--- Evaluating Fine-Tuned Model ---")
        predictions = self.trainer.predict(self.ft_test_ds)
        preds = np.argmax(predictions.predictions, axis=1)
        true_labels = predictions.label_ids

        # 1. Confusion Matrix
        cm = confusion_matrix(true_labels, preds)
        plt.figure(figsize=(6, 5))
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                    xticklabels=['Negative', 'Positive'],
                    yticklabels=['Negative', 'Positive'])
        plt.xlabel('Predicted')
        plt.ylabel('True')
        plt.title('Fine-Tuned Model Confusion Matrix')
        plt.savefig("result/finetune_confusion_matrix.png")
        plt.show()

        # 2. ROC Curve
        probs = torch.nn.functional.softmax(torch.tensor(predictions.predictions), dim=-1)[:, 1].numpy()
        fpr, tpr, thresholds = roc_curve(true_labels, probs)
        roc_auc = auc(fpr, tpr)

        plt.figure(figsize=(8, 6))
        plt.plot(fpr, tpr, color='darkorange', lw=2, label=f'ROC curve (area = {roc_auc:.2f})')
        plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
        plt.xlim([0.0, 1.0])
        plt.ylim([0.0, 1.05])
        plt.xlabel('False Positive Rate')
        plt.ylabel('True Positive Rate')
        plt.title('Receiver Operating Characteristic (ROC)')
        plt.legend(loc="lower right")
        plt.savefig("result/finetune_roc_curve.png")   # 原版存到当前目录
        plt.show()

    def run_Experiment2(self):
        self.huggingface_format_trans()
        self.tokenize()
        self.mapdata()
        self.tsne_visualization()
        self.traditional_method()
        self.fine_tuning_method()
        self.result_display()