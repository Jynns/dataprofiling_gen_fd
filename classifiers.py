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

        if "ground_truth" not in df.columns:
            raise ValueError("Training CSV must contain a 'ground_truth' column.")

        # Count genuine and  non genuine (spurious)
        num_genuine = (df["ground_truth"] == 1).sum()
        num_spurious = (df["ground_truth"] == 0).sum()
        total = len(df)

        # Compute probabilities
        self.p_genuine = num_genuine / total
        self.p_spurious = num_spurious / total

        print("Training complete.")
        print(f"Probability of genuine FD: {self.p_genuine:.4f}")
        print(f"Probability of spurious FD: {self.p_spurious:.4f}")

    def predict(self, csv_path):
        """
        Create prediction column on new CSV.
        CSV must contain: LHS, RHS.
        """
        if self.p_genuine is None:
            raise ValueError("Model not fitted yet. Call fit() first.")

        df = pd.read_csv(csv_path)

        # Check LHS/RHS presence
        if "LHS" not in df.columns or "RHS" not in df.columns:
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

        df.to_csv(output_path, index=False)

        print(f"Predictions written to: {output_path}")

        return df



clf = RandomClassifier()
clf.fit("training_fds.csv")

clf.predict(
    csv_path=".csv",
    output_path=".csv"
)