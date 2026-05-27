import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import torch

from datasets import Dataset
from peft import LoraConfig, get_peft_model, TaskType
from transformers import (
    AutoTokenizer,
    AutoModelForSequenceClassification,
    TrainingArguments,
    Trainer,
)

from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
    classification_report,
    confusion_matrix,
    ConfusionMatrixDisplay,
    RocCurveDisplay,
)


TARGET = "IsHAB_AIDE_SOC"
MODEL_NAME = "sentence-transformers/all-MiniLM-L6-v2"


def load_split_dataset(train_path: Path, test_path: Path):
    train_df = pd.read_csv(train_path)
    test_df = pd.read_csv(test_path)

    for df_name, df in [("train", train_df), ("test", test_df)]:
        if "text" not in df.columns:
            raise ValueError(f"Le fichier {df_name} doit contenir une colonne 'text'.")
        if TARGET not in df.columns:
            raise ValueError(f"Le fichier {df_name} doit contenir la cible '{TARGET}'.")

    train_df = train_df.dropna(subset=["text", TARGET]).copy()
    test_df = test_df.dropna(subset=["text", TARGET]).copy()

    train_df["text"] = train_df["text"].astype(str)
    test_df["text"] = test_df["text"].astype(str)

    train_df[TARGET] = train_df[TARGET].astype(int)
    test_df[TARGET] = test_df[TARGET].astype(int)

    return train_df, test_df


def build_hf_datasets(train_df, test_df, tokenizer, max_length):
    train_dataset = Dataset.from_pandas(
        train_df[["text", TARGET]].rename(columns={TARGET: "labels"}),
        preserve_index=False,
    )

    test_dataset = Dataset.from_pandas(
        test_df[["text", TARGET]].rename(columns={TARGET: "labels"}),
        preserve_index=False,
    )

    def tokenize(batch):
        return tokenizer(
            batch["text"],
            truncation=True,
            padding="max_length",
            max_length=max_length,
        )

    train_dataset = train_dataset.map(tokenize, batched=True)
    test_dataset = test_dataset.map(tokenize, batched=True)

    train_dataset = train_dataset.remove_columns(["text"])
    test_dataset = test_dataset.remove_columns(["text"])

    return train_dataset, test_dataset


def build_lora_model(model_name: str):
    model = AutoModelForSequenceClassification.from_pretrained(
        model_name,
        num_labels=2,
    )

    lora_config = LoraConfig(
        task_type=TaskType.SEQ_CLS,
        r=8,
        lora_alpha=16,
        lora_dropout=0.1,
        target_modules="all-linear",
    )

    model = get_peft_model(model, lora_config)

    print("\nParamètres entraînables :")
    model.print_trainable_parameters()

    return model


def compute_metrics(eval_pred):
    logits, labels = eval_pred

    probabilities = torch.softmax(
        torch.tensor(logits),
        dim=1,
    ).numpy()

    y_pred = np.argmax(probabilities, axis=1)
    y_proba = probabilities[:, 1]

    return {
        "accuracy": accuracy_score(labels, y_pred),
        "precision_0": precision_score(labels, y_pred, pos_label=0, zero_division=0),
        "recall_0": recall_score(labels, y_pred, pos_label=0, zero_division=0),
        "f1_0": f1_score(labels, y_pred, pos_label=0, zero_division=0),
        "precision_1": precision_score(labels, y_pred, pos_label=1, zero_division=0),
        "recall_1": recall_score(labels, y_pred, pos_label=1, zero_division=0),
        "f1_1": f1_score(labels, y_pred, pos_label=1, zero_division=0),
        "roc_auc": roc_auc_score(labels, y_proba),
    }


def evaluate_and_save(trainer, test_dataset, test_df, output_dir: Path):
    predictions = trainer.predict(test_dataset)

    logits = predictions.predictions
    labels = predictions.label_ids

    probabilities = torch.softmax(
        torch.tensor(logits),
        dim=1,
    ).numpy()

    y_pred = np.argmax(probabilities, axis=1)
    y_proba = probabilities[:, 1]

    results = {
        "model": "minilm_lora",
        "feature_type": "text_end_to_end_lora",
        "accuracy": accuracy_score(labels, y_pred),
        "precision_0": precision_score(labels, y_pred, pos_label=0, zero_division=0),
        "recall_0": recall_score(labels, y_pred, pos_label=0, zero_division=0),
        "f1_0": f1_score(labels, y_pred, pos_label=0, zero_division=0),
        "precision_1": precision_score(labels, y_pred, pos_label=1, zero_division=0),
        "recall_1": recall_score(labels, y_pred, pos_label=1, zero_division=0),
        "f1_1": f1_score(labels, y_pred, pos_label=1, zero_division=0),
        "roc_auc": roc_auc_score(labels, y_proba),
    }

    results_df = pd.DataFrame([results])
    results_df.to_csv(output_dir / "lora_model_comparison.csv", index=False)

    predictions_df = test_df.copy()
    predictions_df["y_true"] = labels
    predictions_df["y_pred"] = y_pred
    predictions_df["proba_habilite"] = y_proba
    predictions_df.to_csv(output_dir / "lora_predictions.csv", index=False)

    with open(output_dir / "trainer_metrics.json", "w") as f:
        json.dump(results, f, indent=4)

    print("\nRésultats LoRA :")
    print(results_df)

    print("\nClassification report :")
    print(classification_report(labels, y_pred))

    cm = confusion_matrix(labels, y_pred)

    disp = ConfusionMatrixDisplay(
        confusion_matrix=cm,
        display_labels=["Non habilité", "Habilité"],
    )

    disp.plot()
    plt.title("Matrice de confusion - MiniLM + LoRA")
    plt.tight_layout()
    plt.savefig(output_dir / "lora_confusion_matrix.png", dpi=300)
    plt.close()

    RocCurveDisplay.from_predictions(labels, y_proba)
    plt.title("Courbe ROC - MiniLM + LoRA")
    plt.tight_layout()
    plt.savefig(output_dir / "lora_roc_curve.png", dpi=300)
    plt.close()

    return results_df


def main():
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--train-path",
        type=str,
        default="/users/local/j22gauti/AI-projet/data/embeddings/train_texts.csv",
    )

    parser.add_argument(
        "--test-path",
        type=str,
        default="/users/local/j22gauti/AI-projet/data/embeddings/test_texts.csv",
    )

    parser.add_argument(
        "--output-dir",
        type=str,
        default="/users/local/j22gauti/AI-projet/data/lora",
    )

    parser.add_argument(
        "--max-length",
        type=int,
        default=128,
    )

    parser.add_argument(
        "--epochs",
        type=int,
        default=3,
    )

    parser.add_argument(
        "--batch-size",
        type=int,
        default=16,
    )

    parser.add_argument(
        "--learning-rate",
        type=float,
        default=2e-5,
    )

    args = parser.parse_args()

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    print("Chargement des datasets train/test...")
    train_df, test_df = load_split_dataset(
        Path(args.train_path),
        Path(args.test_path),
    )

    print("Shape train :", train_df.shape)
    print("Shape test :", test_df.shape)

    print("\nRépartition cible train :")
    print(train_df[TARGET].value_counts(normalize=True))

    print("\nRépartition cible test :")
    print(test_df[TARGET].value_counts(normalize=True))

    print("\nChargement du tokenizer...")
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)

    print("\nPréparation des datasets Hugging Face...")
    train_dataset, test_dataset = build_hf_datasets(
        train_df,
        test_df,
        tokenizer,
        args.max_length,
    )

    print("\nChargement du modèle avec LoRA...")
    model = build_lora_model(MODEL_NAME)

    training_args = TrainingArguments(
        output_dir=str(output_dir / "checkpoints"),
        num_train_epochs=args.epochs,
        per_device_train_batch_size=args.batch_size,
        per_device_eval_batch_size=args.batch_size,
        learning_rate=args.learning_rate,
        weight_decay=0.01,
        eval_strategy="epoch",
        save_strategy="epoch",
        load_best_model_at_end=True,
        metric_for_best_model="roc_auc",
        greater_is_better=True,
        logging_steps=20,
        report_to="none",
        use_cpu="True"
    )

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=test_dataset,
        compute_metrics=compute_metrics,
    )

    print("\nDébut du fine-tuning LoRA...")
    trainer.train()

    print("\nÉvaluation finale...")
    evaluate_and_save(
        trainer,
        test_dataset,
        test_df,
        output_dir,
    )

    print("\nSauvegarde du modèle LoRA...")
    trainer.save_model(output_dir / "model")
    tokenizer.save_pretrained(output_dir / "model")

    print("\nFichiers sauvegardés dans :", output_dir)
    print("- lora_model_comparison.csv")
    print("- lora_predictions.csv")
    print("- trainer_metrics.json")
    print("- lora_confusion_matrix.png")
    print("- lora_roc_curve.png")
    print("- model/")


if __name__ == "__main__":
    main()