import argparse
import csv
import numpy as np
from collections import defaultdict
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    confusion_matrix,
    classification_report
)
import pandas as pd



def evaluate_fd_classification(csv_path, separator, escape, quote):
    # load data and put in dataframe
    df = pd.read_csv(
        csv_path,
        sep=separator,
        escapechar=escape,
        quotechar=quote
    )

    y_true = df["is_genuine"]
    y_pred = df["prediction"]

    metrics = {
        "accuracy": accuracy_score(y_true, y_pred),
        "precision": precision_score(y_true, y_pred, zero_division=0),
        "recall": recall_score(y_true, y_pred, zero_division=0),
        "f1_score": f1_score(y_true, y_pred, zero_division=0),
        "confusion_matrix": confusion_matrix(y_true, y_pred).tolist(),
        "classification_report": classification_report(
            y_true, y_pred, zero_division=0
        )
    }

    print("\n========== FUNCTIONAL DEPENDENCY GENUINENESS EVALUATION ==========\n")
    print(f"Accuracy:  {metrics['accuracy']:.4f}")
    print(f"Precision: {metrics['precision']:.4f}")
    print(f"Recall:    {metrics['recall']:.4f}")
    print(f"F1 Score:  {metrics['f1_score']:.4f}\n")
    
    print("\n############ Confusion Matrix ############\n")
    print(metrics["confusion_matrix"])
    
    print("\n############ Classification Report ############\n")
    print(metrics["classification_report"])

    return metrics


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Evaluate FD genuineness classification results.")

    parser.add_argument(
        "--inputs",
        type=str,
        required=True,
        help="Path to input CSV file."
    )

    parser.add_argument(
        "--separator",
        type=str,
        default=",",
        help="CSV column separator (default=',')."
    )

    parser.add_argument(
        "--escape",
        type=str,
        default="\\",
        help="Escape character for CSV parsing."
    )

    parser.add_argument(
        "--quote",
        type=str,
        default="\"",
        help="Quote character for CSV parsing."
    )

    args = parser.parse_args()

    evaluate_fd_classification(
        csv_path=args.inputs,
        separator=args.separator,
        escape=args.escape,
        quote=args.quote
    )