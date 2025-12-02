from classifiers import RandomClassifier
from evaluation.evaluate_fd import evaluate_fd_classification

clf = RandomClassifier()

clf.fit(r"C:\Users\Andy\Data Profiling\EDA\dataprofiling_gen_fd\dataset.csv")

clf.predict(
    csv_path=r"C:\Users\Andy\Data Profiling\EDA\dataprofiling_gen_fd\dataset.csv",
    output_path=r"C:\Users\Andy\Data Profiling\EDA\dataprofiling_gen_fd\evaluation"
)   

evaluate_fd_classification(
    csv_path=r"C:\Users\Andy\Data Profiling\EDA\dataprofiling_gen_fd\evaluation\dataset_predicted.csv",
    separator=",",
    escape="\\",
    quote="\""
)
