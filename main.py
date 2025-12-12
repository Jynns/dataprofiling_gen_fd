import argparse
from pathlib import Path
from data_transformer import IdentityTransformer
from classifiers import RandomClassifier
from evaluation.evaluate_fd import evaluate_fd_classification
import pandas as pd
from sklearn.model_selection import train_test_split

# clf = RandomClassifier()

# clf.fit(r"C:\Users\Andy\Data Profiling\EDA\dataprofiling_gen_fd\dataset.csv")

# clf.predict(
#     csv_path=r"C:\Users\Andy\Data Profiling\EDA\dataprofiling_gen_fd\dataset.csv",
#     output_path=r"C:\Users\Andy\Data Profiling\EDA\dataprofiling_gen_fd\evaluation"
# )   

# evaluate_fd_classification(
#     csv_path=r"C:\Users\Andy\Data Profiling\EDA\dataprofiling_gen_fd\evaluation\dataset_predicted.csv",
#     separator=",",
#     escape="\\",
#     quote="\""
# )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Classify FD genuineness.")

    parser.add_argument(
        "--data",
        type=str,
        required=True,
        help="Path to labelled input CSV file."
    )

    parser.add_argument(
        "--metadata",
        type=str,
        default=None,
        help="Path to metadata CSV file."
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

    parser.add_argument(
        "--test-size", 
        type=float, 
        default=0.2, 
        help="Fraction for test split."
    )
    
    parser.add_argument(
        "--outputdir", 
        default="evaluation", 
        help="Where to write splits/preds."
    )

    parser.add_argument(
        "--classifier",
        type=str,
        default="random",
        choices=["random", "llm"],
        help="Classifier to use. Options are random or llm."
    )

    args = parser.parse_args()

    #df = pd.read_csv(args.data, sep=args.separator, escapechar=args.escape, quotechar=args.quote)

    identityTransformer = IdentityTransformer()

    transformed_df = identityTransformer.transform(args.data, args.metadata)
    
    train_df, test_df = train_test_split(
        transformed_df,
        test_size=args.test_size,
        stratify=transformed_df["is_genuine"],
        #random_state=args.seed,
    )

    out_dir = Path(args.outputdir)
    out_dir.mkdir(parents=True, exist_ok=True)

    train_path = out_dir / "train.csv"
    test_path = out_dir / "test.csv"
    train_df.to_csv(train_path, index=False)
    test_df.to_csv(test_path, index=False)


    clf = None

    if args.classifier == "random":       
        clf = RandomClassifier()
        clf.fit(train_path)
        pred_path = out_dir / f"{test_path.stem}_predicted.csv"
        clf.predict(csv_path=test_path, output_path=out_dir)

    if args.classifier == "llm":
        raise NotImplementedError("LLM classifier not implemented yet.")       


    ##### Evaluation #####
    
    evaluate_fd_classification(
        csv_path=pred_path,
        separator=args.separator,
        escape=args.escape,
        quote=args.quote,
    )

    # evaluate_fd_classification(
    #     csv_path=args.inputs,
    #     separator=args.separator,
    #     escape=args.escape,
    #     quote=args.quote
    # )