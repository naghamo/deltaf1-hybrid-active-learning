import os
import pandas as pd
from sklearn.model_selection import train_test_split
from transformers import AutoTokenizer
from .text_datasets import TextClassificationDataset


def tokenize(datasets: list, model_name_or_path: str, tokenizer_kwargs: dict):
    """
    Tokenize the train, val and test datasets.

    Args:
        datasets (list): List of DataFrames to tokenize.
        model_name_or_path (str): Pretrained model name or path for the tokenizer.
        tokenizer_kwargs (dict): Additional arguments for the tokenizer.

    Returns:
        tokenized_datasets (list): List of tokenized datasets.
    """
    tokenizer = AutoTokenizer.from_pretrained(model_name_or_path)
    tokenized_datasets = []

    for df in datasets:
        df.reset_index(drop=True, inplace=True)
        tokenized_datasets.append(TextClassificationDataset(
            texts=df['text'].tolist(),
            labels=df['label'].tolist(),
            tokenizer=tokenizer,
            tokenizer_kwargs=tokenizer_kwargs,
        ))

    return tokenized_datasets


def load_agnews(path="../data", val_size=0.1, seed=42, model_name_or_path="bert-base-uncased", tokenizer_kwargs=None):
    """
    Load and preprocess the AG News dataset.

    Args:
        path (str): Path to the data directory.
        val_size (float): Fraction of training data to use as validation.
        seed (int): Random seed for reproducibility.

    Returns:
        df_train (DataFrame): Training split with columns ['text', 'label'].
        df_val (DataFrame): Validation split.
        df_test (DataFrame): Test split.
    """
    df_train = pd.read_csv(os.path.join(path, "train_agnews.csv"))
    df_test = pd.read_csv(os.path.join(path, "test_agnews.csv"))

    df_train["text"] = df_train["Title"] + ". " + df_train["Description"]
    df_test["text"] = df_test["Title"] + ". " + df_test["Description"]
    df_train["label"] = df_train["Class Index"] - 1
    df_test["label"] = df_test["Class Index"] - 1

    df_train = df_train[["text", "label"]]
    df_test = df_test[["text", "label"]]

    df_train, df_val = train_test_split(df_train, test_size=val_size, stratify=df_train["label"], random_state=seed)

    train_dataset, val_dataset, test_dataset = tokenize([df_train, df_val, df_test], model_name_or_path,
                                                        tokenizer_kwargs or {})

    return train_dataset, val_dataset, test_dataset


def load_imdb(path="../data", val_size=0.1, test_size=0.2, seed=42, model_name_or_path="bert-base-uncased",
              tokenizer_kwargs=None):
    """
    Load and preprocess the IMDb sentiment classification dataset.

    Args:
        path (str): Path to the data directory.
        val_size (float): Fraction of training data to use as validation.
        test_size (float): Fraction of full data to use as test split.
        seed (int): Random seed for reproducibility.

    Returns:
        df_train (DataFrame): Training split with ['text', 'label'].
        df_val (DataFrame): Validation split.
        df_test (DataFrame): Test split.
    """
    df = pd.read_csv(os.path.join(path, "imdb.csv"))
    df["label"] = df["sentiment"].map({"positive": 1, "negative": 0})
    df = df[["review", "label"]].rename(columns={"review": "text"})

    df_train_val, df_test = train_test_split(df, test_size=test_size, stratify=df["label"], random_state=seed)
    df_train, df_val = train_test_split(df_train_val, test_size=val_size, stratify=df_train_val["label"],
                                        random_state=seed)

    train_dataset, val_dataset, test_dataset = tokenize([df_train, df_val, df_test], model_name_or_path,
                                                        tokenizer_kwargs or {})

    return train_dataset, val_dataset, test_dataset


def load_jigsaw(path="../data", val_size=0.1, test_size=0.2, seed=42, model_name_or_path="bert-base-uncased",
                tokenizer_kwargs=None):
    """
    Load and preprocess the Jigsaw dataset for multi-label classification.

    Returns train/val/test sets with `text` + the 6 toxic label columns.
    """
    df = pd.read_csv(os.path.join(path, "jigsaw.csv"))
    label_cols = ['toxic', 'severe_toxic', 'obscene', 'threat', 'insult', 'identity_hate']

    df["text"] = df["comment_text"]

    # Stratify only for splitting (binary flag)
    df["stratify_label"] = (df[label_cols].sum(axis=1) > 0).astype(int)

    columns_to_keep = ["text"] + label_cols + ["stratify_label"]
    df = df[columns_to_keep]

    df_train_val, df_test = train_test_split(
        df, test_size=test_size, stratify=df["stratify_label"], random_state=seed
    )
    df_train, df_val = train_test_split(
        df_train_val, test_size=val_size, stratify=df_train_val["stratify_label"], random_state=seed
    )

    for subset in (df_train, df_val, df_test):
        subset.drop(columns=["stratify_label"], inplace=True)

    train_dataset, val_dataset, test_dataset = tokenize([df_train, df_val, df_test], model_name_or_path,
                                                        tokenizer_kwargs or {})

    return train_dataset, val_dataset, test_dataset
