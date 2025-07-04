__all__ = ['run_val', 'get_roc_curve', 'prep_events', 'get_reports']

# Cell
import numpy as np
import pandas as pd
import logging

from sklearn.metrics import (
    roc_auc_score,
    roc_curve,
    classification_report,
    confusion_matrix,
    f1_score,
)
from timeseriescv.cross_validation import PurgedWalkForwardCV, CombPurgedKFoldCV
from .single_wf_cv import SinglePurgedWalkForwardCV


def run_val(cv, events, clf, X_train, y_train, X_test, y_test):
    test_indices, y_truths, y_preds, y_preds_proba = [], [], [], []
    pred_times = events.index.to_series().reindex(X_test.index)

    eval_times = events["t1"].reindex(X_test.index)

    train_test_splits = cv.split(X_test, pred_times=pred_times, eval_times=eval_times)
    for i, (train_index, test_index) in enumerate(train_test_splits):
        logging.info(f"Running validation for {type(clf).__name__}: round {i + 1}/{cv.n_splits - 1}")
        X_train_ = pd.concat([X_train, X_test.iloc[train_index]])
        X_test_ = X_test.iloc[test_index]
        y_train_ = pd.concat([y_train, y_test.iloc[train_index]])
        y_test_ = y_test.iloc[test_index]

        if clf is not None:
            clf.fit(X_train_, y_train_)

            y_pred = clf.predict(X_test_)
            y_pred_proba = clf.predict_proba(X_test_)[:, 1]
        else:
            # Running without using meta-labeling
            y_pred = y_pred_proba = pd.Series(1, index=X_test_)

        test_indices.append(test_index)
        y_truths.append(y_test_)
        y_preds.append(y_pred)
        y_preds_proba.append(y_pred_proba)

    rets = [y_truths, y_preds, y_preds_proba, test_indices]
    return [np.concatenate(ret) for ret in rets]


def get_roc_curve(clf, y_test, y_pred):
    fpr, tpr, _ = roc_curve(y_test, y_pred)

# Cell

def prep_events(events, y_pred_proba, y_pred):
    events["y_pred_proba"] = y_pred_proba
    events["y_pred"] = y_pred

    events = events.set_index(events.index.map(lambda x: x.isoformat()))
    events["t1"] = events["t1"].map(lambda x: x.isoformat())
    return events.to_dict()


def get_reports(
    clf,
    events_test,
    X_train,
    y_train,
    X_test,
    y_test,
    test_procedure,
    use_alpha,
    hyper_params,
):
    logging.info(f"Getting reports for {type(clf).__name__}")
    if test_procedure == "simple":
        cv = SinglePurgedWalkForwardCV(n_splits=10, n_test_splits=2, min_train_splits=8)
    elif test_procedure == "walk_forward":
        cv = PurgedWalkForwardCV(n_splits=5, n_test_splits=1, min_train_splits=1)
    elif test_procedure == "cpcv":
        cv = CombPurgedKFoldCV(n_splits=5, n_test_splits=1)

    y_test, y_pred, y_pred_proba, test_indices = run_val(
        cv, events_test, clf, X_train, y_train, X_test, y_test
    )

    events = prep_events(events_test.iloc[test_indices], y_pred_proba, y_pred)

    with_ml = {
        "classification_report_str": classification_report(
            y_true=y_test, y_pred=y_pred
        ),
        "classification_report": classification_report(
            y_true=y_test, y_pred=y_pred, output_dict=True
        ),
        "f1_score": f1_score(y_test, y_pred, average="micro", zero_division=0),
        "confusion_matrix": confusion_matrix(y_test, y_pred),
        "roc_curve": roc_curve(y_test, y_pred),
        "roc_auc_score": roc_auc_score(y_test, y_pred_proba),
        "hyper_params": hyper_params,
    }
    if not use_alpha:
        return {"primary": with_ml, "secondary": None, "events": events}

    y_pred_ones = np.ones(y_pred.shape)
    no_ml = {
        "classification_report_str": classification_report(
            y_true=y_test, y_pred=y_pred_ones
        ),
        "classification_report": classification_report(
            y_true=y_test, y_pred=y_pred_ones, output_dict=True
        ),
        "f1_score": f1_score(y_test, y_pred_ones, average="micro", zero_division=0),
        "confusion_matrix": confusion_matrix(y_test, y_pred_ones),
        "roc_auc_score": None,
        "hyper_params": None,
    }

    return {"primary": no_ml, "secondary": with_ml, "events": events}