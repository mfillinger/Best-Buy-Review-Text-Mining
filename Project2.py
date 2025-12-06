import re
import pandas as pd
from pathlib import Path

from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.pipeline import Pipeline
from sklearn.naive_bayes import MultinomialNB
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
)


#config

DATA_PATH = Path("astro_a50x_reviews_raw.csv")

#helper functions
def basic_clean(text: str) -> str:
    text= str(text).lower()
    text = re.sub(r"\s+", " ", text)
    return text.strip()


def add_sentiment_label(df: pd.DataFrame) -> pd.DataFrame:
#4 star and up = positive (1), everything else (0)
    df = df.copy()
    df["sentiment"]= df["star_rating"].apply(lambda s: 1 if s >= 4 else 0)
    return df


def print_confusion(cm, labels):
    print("Confusion matrix [rows=true, cols=pred]:")
    print("      " + "  ".join(f"{lab:>12}" for lab in labels))
    for i, row_label in enumerate(labels):
        row_vals = "  ".join(f"{val:12d}" for val in cm[i])
        print(f"{row_label:>12}  {row_vals}")

#main

def main():
#load data
    if not DATA_PATH.exists():
        print(f"Could not find {DATA_PATH}.")
        return

    df= pd.read_csv(DATA_PATH)
    print(f"Loaded {len(df)} reviews from {DATA_PATH.name}\n")

    if "review_text" not in df.columns or "star_rating" not in df.columns:
        print("CSV must contain 'review_text' and 'star_rating' columns.")
        return

#clean text and prepare labels
    df["clean_text"] = df["review_text"].apply(basic_clean)

    print("Star rating distribution:")
    print(df["star_rating"].value_counts().sort_index(), "\n")

    df= add_sentiment_label(df)

    print("Sentiment distribution (0 = not_positive, 1 = positive):")
    print(df["sentiment"].value_counts().sort_index(), "\n")

#train/test split
    X= df["clean_text"]
    y= df["sentiment"]

    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=0.2,
        random_state=42,
        stratify=y,
    )

    target_names =["not_positive", "positive"]

#naive bayes model
    nb_pipeline = Pipeline(
        steps=[
            ("tfidf", TfidfVectorizer(stop_words="english")),
            ("clf", MultinomialNB()),
        ]
    )

    print("Training Naive Bayes...\n")
    nb_pipeline.fit(X_train, y_train)

    y_pred_nb = nb_pipeline.predict(X_test)
    nb_acc = accuracy_score(y_test, y_pred_nb)

    print(f"Naive Bayes Accuracy: {nb_acc:.4f}\n")
    print("Naive Bayes Classification Report:")
    print(
        classification_report(
            y_test,
            y_pred_nb,
            target_names=target_names,
            zero_division=0,
        )
    )

    cm_nb= confusion_matrix(y_test, y_pred_nb)
    print_confusion(cm_nb, target_names)
    print("\n" + "-" * 60 + "\n")

#logistic regression model
    lr_pipeline = Pipeline(
        steps=[
            ("tfidf", TfidfVectorizer(stop_words="english")),
            (
                "clf",
                LogisticRegression(
                    max_iter=1000,
                    class_weight="balanced",
                ),
            ),
        ]
    )

    print("Training Logistic Regression...\n")
    lr_pipeline.fit(X_train, y_train)

    y_pred_lr= lr_pipeline.predict(X_test)
    lr_acc = accuracy_score(y_test, y_pred_lr)

    print(f"Logistic Regression Accuracy: {lr_acc:.4f}\n")
    print("Logistic Regression Classification Report:")
    print(
        classification_report(
            y_test,
            y_pred_lr,
            target_names=target_names,
            zero_division=0,
        )
    )

    cm_lr = confusion_matrix(y_test, y_pred_lr)
    print_confusion(cm_lr, target_names)
    print("\nDone.")


if __name__ == "__main__":
    main()
