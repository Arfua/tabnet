import os
import random
from typing import Dict, List, Optional, Tuple

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
from scipy.stats import ks_2samp
from sklearn.metrics import roc_auc_score, log_loss
from sklearn.preprocessing import LabelEncoder
from sklearn.multioutput import MultiOutputClassifier
from pytorch_tabnet.multitask import TabNetMultiTaskClassifier
from pytorch_tabnet.pretraining import TabNetPretrainer

Folds = Dict[int, Tuple[List[int], List[int]]]
SplitDataFrames = Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]
UnsupervisedModel = Optional[TabNetPretrainer]


def deterministic(seed: int) -> None:
    random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True


def encode_categorical(df: pd.DataFrame) -> Tuple[pd.DataFrame, List[int], List[int], List[LabelEncoder]]:
    categorical_ids = []
    categorical_dims = []
    encoders = []

    nunique_df = df.nunique()
    types_df = df.dtypes
    for i, feature in enumerate(df.columns):
        if types_df[feature] != "object" and nunique_df[feature] > 200:
            continue
        print(f"Unique values for {feature}: {nunique_df[feature]}")
        df[feature] = df[feature].fillna("Encoded_by_Raid")
        encoder = LabelEncoder()
        df[feature] = encoder.fit_transform(df[feature].values)
        categorical_ids.append(i)
        categorical_dims.append(len(encoder.classes_))
        encoders.append(encoder)

    return df, categorical_ids, categorical_dims, encoders


def extract_fold(x_df: pd.DataFrame, y_df: pd.DataFrame, folds: Folds, fold_ix: int) -> SplitDataFrames:
    x_train, y_train = x_df.iloc[folds[fold_ix][0]].to_numpy(), y_df.iloc[folds[fold_ix][0]].to_numpy()
    x_valid, y_valid = x_df.iloc[folds[fold_ix][1]].to_numpy(), y_df.iloc[folds[fold_ix][1]].to_numpy()
    return x_train, y_train, x_valid, y_valid


def fit_and_predict(
        clf: TabNetMultiTaskClassifier, x_df: pd.DataFrame, y_df: pd.DataFrame, folds: Folds, fold_ix: int,
        max_epochs: int = 1000, verbose: int = 5,
        unsupervised_model: UnsupervisedModel = None) -> Tuple[TabNetMultiTaskClassifier, List[float]]:
    print(f"Training for fold {fold_ix}:")
    x_train, y_train, x_valid, y_valid = extract_fold(x_df, y_df, folds, fold_ix)
    clf.verbose = verbose
    clf.fit(
        X_train=x_train, y_train=y_train,
        eval_set=[(x_train, y_train), (x_valid, y_valid)],
        eval_name=["train", "valid"],
        eval_metric=["auc", "logloss"],
        max_epochs=max_epochs,
        patience=50,
        batch_size=1024,
        virtual_batch_size=128,
        num_workers=0,
        drop_last=False,
        from_unsupervised=unsupervised_model
    )

    f = plt.figure(figsize=(20, 10))
    # plot losses (drop first epochs to have a nice plot)
    ax1 = f.add_subplot(221)
    ax1.set_title("Train/Valid LogLoss")
    ax1.plot(clf.history["train_logloss"][5:])
    ax1.plot(clf.history["valid_logloss"][5:])

    ax2 = f.add_subplot(222)
    ax2.set_title("Learning rate")
    ax2.plot([x for x in clf.history["lr"]][5:])

    preds_valid = clf.predict_proba(x_valid)  # list of predictions for each target

    valid_aucs = [
        roc_auc_score(y_valid[:, task_idx], task_pred[:, 1]) for task_idx, task_pred in enumerate(preds_valid)]
    ax3 = f.add_subplot(223)
    ax3.set_title("AUC")
    ax3.set_xlabel("Number of positives for a task")
    ax3.scatter(y_valid.sum(axis=0), valid_aucs)

    valid_logloss = [
        log_loss(y_valid[:, task_idx], task_pred[:, 1]) for task_idx, task_pred in enumerate(preds_valid)]
    ax4 = f.add_subplot(224)
    ax4.set_title("LogLoss")
    ax4.set_xlabel("Number of positives for a task")
    ax4.scatter(y_valid.sum(axis=0), valid_logloss)
    return clf, valid_aucs


def fit_and_predict_xgb(
        clf: MultiOutputClassifier, x_df: pd.DataFrame, y_df: pd.DataFrame, folds: Folds,
        fold_ix: int, compare_aucs: List[float]) -> MultiOutputClassifier:
    print(f"Training for fold {fold_ix}:")
    x_train, y_train, x_valid, y_valid = extract_fold(x_df, y_df, folds, fold_ix)
    eval_set = [(x_valid, y_valid)]
    params = {
        "estimator__eval_set": eval_set,
        "estimator__eval_metric": "logloss"
    }
    clf = clf.set_params(**params)
    clf.fit(x_train, y_train)
    preds_valid = clf.predict_proba(x_valid)
    valid_aucs = [
        roc_auc_score(y_valid[:, task_idx], task_pred[:, 1]) for task_idx, task_pred in enumerate(preds_valid)]
    f = plt.figure(figsize=(20, 10))
    ax = f.add_subplot(111)
    ax.set_title("AUC")
    ax.set_xlabel("Number of positives for a task")
    ax.scatter(y_valid.sum(axis=0), valid_aucs, c="r", label="XGBoost")
    ax.scatter(y_valid.sum(axis=0), compare_aucs, c="b", label="TabNet")
    ax.legend(loc="upper left")
    return clf


def feature_importances(clf: TabNetMultiTaskClassifier, x_df: pd.DataFrame, fold_ix: int) -> None:
    feature_importances_df = pd.Series(clf.feature_importances_, index=x_df.columns)
    feature_importances_df.nlargest(20).plot(kind="barh", title=f"TabNet feature importances for fold {fold_ix}")


def feature_importances_xgb(clf: MultiOutputClassifier, x_df: pd.DataFrame, fold_ix: int) -> None:
    feature_importances_stacked = np.vstack(
        [clf.estimators_[i].feature_importances_ for i in range(len(clf.estimators_))]).sum(axis=0)
    feature_importances_df = pd.Series(feature_importances_stacked, index=x_df.columns)
    feature_importances_df.nlargest(20).plot(
        kind="barh", title=f"XGBoost feature importances for fold {fold_ix} (summed up for all classifiers)")


def get_diff_distribution_features(train_df: pd.DataFrame, test_df: pd.DataFrame, threshold: float = 0.1) -> List[str]:
    diff_features = []
    for feature in train_df.columns:
        statistic, pvalue = ks_2samp(train_df[feature].values, test_df[feature].values)
        if pvalue <= 0.05 and np.abs(statistic) > threshold:
            print(f"Feature: {feature}. pvalue: {pvalue}. statistic: {statistic}")
            diff_features.append(feature)
    if not diff_features:
        print("All the features have the same distribution in train and test datasets!")
    return diff_features
