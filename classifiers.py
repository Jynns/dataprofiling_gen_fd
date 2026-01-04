from pathlib import Path
import pandas as pd
import numpy as np
from abc import ABC, abstractmethod

class Classifier(ABC):
    """Base classifier"""

    def __init__(self):
        pass

    @abstractmethod
    def fit(self, csv_path):
        """Train the classifier from a CSV."""
        pass

    @abstractmethod
    def predict(self, csv_path):
        """Predict on a CSV and return predictions."""
        pass


class RandomClassifier(Classifier):
    """
    A simple classifier that learns the probability
    distribution of genuine vs spurious from the training data.
    
    ground_truth must be either 0 (spurious) or 1 (genuine).
    """

    def __init__(self):
        super().__init__()
        self.p_genuine = None
        self.p_spurious = None

    def fit(self, csv_path):
        """
        Fit the model by reading ground truth to compute probabilitiesn from the CSV.
        CSV must contain: LHS, RHS, ground_truth.
        """
        df = pd.read_csv(csv_path)

        if "is_genuine" not in df.columns:
            raise ValueError("Training CSV must contain a 'is_genuine' column.")

        # Count genuine and  non genuine (spurious)
        num_genuine = (df["is_genuine"] == 1).sum()
        num_spurious = (df["is_genuine"] == 0).sum()
        total = len(df)

        # Compute probabilities
        self.p_genuine = num_genuine / total
        self.p_spurious = num_spurious / total

        print("Training complete.")
        print(f"Probability of genuine FD: {self.p_genuine:.4f}")
        print(f"Probability of spurious FD: {self.p_spurious:.4f}")

    def predict(self, csv_path, output_path=None):
        """
        Create prediction column on new CSV.
        CSV must contain: LHS, RHS.
        """
        if self.p_genuine is None:
            raise ValueError("Model not fitted yet. Call fit() first.")

        df = pd.read_csv(csv_path)

        # Check LHS/RHS presence
        if "lhs" not in df.columns or "rhs" not in df.columns:
            raise ValueError("Prediction CSV must contain 'LHS' and 'RHS' columns.")

        # predict randomly based on learned (trained) probability
        predictions = np.random.choice(
            [0, 1],
            size=len(df),
            p=[self.p_spurious, self.p_genuine]
        )

        df["prediction"] = predictions

        if output_path is None:
             output_path = csv_path.replace(".csv", "_predicted.csv")
        else:
            csv_stem = Path(csv_path).stem
            output_path = Path(output_path) / f"{csv_stem}_predicted.csv"        

        df.to_csv(output_path, index=False)

        print(f"Predictions written to: {output_path}")

        return df

class LlamaModel(Classifier):
    """
    A simple classifier that uses llama3.1:8b
    to classify genuineness
    """

    def __init__(self):
        super().__init__()

    def predict(self, csv_path, output_path=None):
        """
        Create prediction column on new CSV.
        CSV must contain: LHS, RHS.
        """

        df = pd.read_csv(csv_path)

        # Check LHS/RHS presence
        if "lhs" not in df.columns or "rhs" not in df.columns:
            raise ValueError("Prediction CSV must contain 'LHS' and 'RHS' columns.")

        

        df["prediction"] = predictions

        if output_path is None:
             output_path = csv_path.replace(".csv", "_predicted.csv")
        else:
            csv_stem = Path(csv_path).stem
            output_path = Path(output_path) / f"{csv_stem}_predicted.csv"        

        df.to_csv(output_path, index=False)

        print(f"Predictions written to: {output_path}")

        return df


# clf = RandomClassifier()
# clf.fit(r"C:\Users\Andy\Data Profiling\EDA\dataprofiling_gen_fd\dataset.csv")

# clf.predict(
#     csv_path=r"C:\Users\Andy\Data Profiling\EDA\dataprofiling_gen_fd\dataset.csv",
# )