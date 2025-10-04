"""
Data loading utilities for text classification datasets.

This module provides functions to load and preprocess various text classification
datasets including AG News, IMDb, and Jigsaw. It handles train/validation/test
splitting, tokenization, and returns both tokenized datasets and raw DataFrames.
"""

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


def split_train_val_test(df, label_col, val_size=0.01, test_size=0.2, seed=42):
    """
    Split a DataFrame into train, validation, and test sets with stratification.

    Args:
        df (DataFrame): Input DataFrame to split.
        label_col (str): Name of the label column for stratification.
        val_size (float): Fraction of training data to use as validation (default: 0.01).
        test_size (float): Fraction of total data to use as test set (default: 0.2).
        seed (int): Random seed for reproducibility (default: 42).

    Returns:
        tuple: (df_train, df_val, df_test) - Three DataFrames with reset indices.
    """
    df_train_val, df_test = train_test_split(df, test_size=test_size, stratify=df[label_col], random_state=seed)
    df_train, df_val = train_test_split(df_train_val, test_size=val_size / (1 - test_size),
                                        stratify=df_train_val[label_col],
                                        random_state=seed)
    return df_train.reset_index(drop=True), df_val.reset_index(drop=True), df_test.reset_index(drop=True)


def load_agnews(path="../data", val_size=0.01, seed=42, model_name_or_path="bert-base-uncased", tokenizer_kwargs=None):
    """
    Load and preprocess the AG News dataset.

    Args:
        path (str): Path to the data directory.
        val_size (float): Fraction of training data to use as validation.
        seed (int): Random seed for reproducibility.

    Returns:
        tuple: Two tuples containing:
            - (train_dataset, val_dataset, test_dataset): Tokenized TextClassificationDataset objects.
            - (df_train, df_val, df_test): Raw DataFrames for inspection.
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

    return (train_dataset, val_dataset, test_dataset), (df_train.reset_index(drop=True), df_val.reset_index(drop=True),
                                                        df_test.reset_index(
                                                            drop=True))  # Also return raw dataframes for inspection


def load_imdb(path="../data", val_size=0.01, test_size=0.2, seed=42, model_name_or_path="bert-base-uncased",
              tokenizer_kwargs=None):
    """
    Load and preprocess the IMDb sentiment classification dataset.

    Args:
        path (str): Path to the data directory.
        val_size (float): Fraction of training data to use as validation.
        test_size (float): Fraction of full data to use as test split.
        seed (int): Random seed for reproducibility.

    Returns:
        tuple: Two tuples containing:
            - (train_dataset, val_dataset, test_dataset): Tokenized TextClassificationDataset objects.
            - (df_train, df_val, df_test): Raw DataFrames with 'text' and 'label' columns.
    """
    df = pd.read_csv(os.path.join(path, "imdb.csv"))
    df["label"] = df["sentiment"].map({"positive": 1, "negative": 0})
    df = df[["review", "label"]].rename(columns={"review": "text"})

    df_train, df_val, df_test = split_train_val_test(df, label_col="label", val_size=val_size, test_size=test_size,
                                                     seed=seed)

    train_dataset, val_dataset, test_dataset = tokenize([df_train, df_val, df_test], model_name_or_path,
                                                        tokenizer_kwargs or {})

    return (train_dataset, val_dataset, test_dataset), (df_train, df_val,
                                                        df_test)  # return both tokenized and original dfs


def load_jigsaw(path="../data", val_size=0.01, test_size=0.2, seed=42, model_name_or_path="bert-base-uncased",
                tokenizer_kwargs=None):
    """
    Load and preprocess the Jigsaw toxic comment classification dataset.

    Args:
        path (str): Path to the data directory (default: "../data").
        val_size (float): Fraction of training data to use as validation (default: 0.01).
        test_size (float): Fraction of total data to use as test set (default: 0.2).
        seed (int): Random seed for reproducibility (default: 42).
        model_name_or_path (str): Pretrained model name or path for tokenizer (default: "bert-base-uncased").
        tokenizer_kwargs (dict): Additional arguments for the tokenizer (default: None).

    Returns:
        tuple: Two tuples containing:
            - (train_dataset, val_dataset, test_dataset): Tokenized TextClassificationDataset objects.
            - (df_train, df_val, df_test): Raw DataFrames with 'text' and 'label' columns.
    """
    df = pd.read_csv(os.path.join(path, "jigsaw.csv"))
    cols_to_drop = ['id', 'severe_toxic', 'obscene', 'threat', 'insult', 'identity_hate']
    df.drop(cols_to_drop, axis=1, inplace=True)

    df.rename(columns={"comment_text": "text", "toxic": "label"}, inplace=True)

    df_train, df_val, df_test = split_train_val_test(df, label_col="label", val_size=val_size,
                                                     test_size=test_size, seed=seed)

    train_dataset, val_dataset, test_dataset = tokenize([df_train, df_val, df_test], model_name_or_path,
                                                        tokenizer_kwargs or {})

    return (train_dataset, val_dataset, test_dataset), (df_train, df_val,
                                                        df_test)  # return both tokenized and original dfs
