from pathlib import Path
import pandas as pd
import joblib

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, confusion_matrix


DATA_DIR = Path("data")
TRAIN_PATH = DATA_DIR / "train.tsv"
VALID_PATH = DATA_DIR / "valid.tsv"
TEST_PATH  = DATA_DIR / "test.tsv"

MODEL_DIR = Path("models")
MODEL_DIR.mkdir(exist_ok=True)

# LIAR TSV format (no header). These are the standard 14 fields.
LIAR_COLUMNS = [
    "id", "label", "statement", "subject", "speaker", "speaker_job_title",
    "state_info", "party_affiliation", "barely_true_counts", "false_counts",
    "half_true_counts", "mostly_true_counts", "pants_on_fire_counts", "context"
]

def load_liar_tsv(path: Path) -> pd.DataFrame:
    df = pd.read_csv(path, sep="\t", header=None, names=LIAR_COLUMNS)
    df = df.dropna(subset=["statement", "label"]).copy()
    df["statement"] = df["statement"].astype(str)
    df["label"] = df["label"].astype(str).str.strip().str.lower()
    return df

def map_label_to_binary(label: str) -> int:
    """
    1 = suspicious
    0 = less suspicious
    """
    suspicious = {"pants-fire", "false", "barely-true"}
    return 1 if label in suspicious else 0

def train_and_eval():
    # Load
    train_df = load_liar_tsv(TRAIN_PATH)
    valid_df = load_liar_tsv(VALID_PATH)
    test_df  = load_liar_tsv(TEST_PATH)

    # Labels
    y_train = train_df["label"].apply(map_label_to_binary).values
    y_valid = valid_df["label"].apply(map_label_to_binary).values
    y_test  = test_df["label"].apply(map_label_to_binary).values

    X_train = train_df["statement"].tolist()
    X_valid = valid_df["statement"].tolist()
    X_test  = test_df["statement"].tolist()

    # Vectorizer + Model
    vectorizer = TfidfVectorizer(
        max_features=30000,
        ngram_range=(1, 2),
        stop_words="english"
    )

    X_train_vec = vectorizer.fit_transform(X_train)
    X_valid_vec = vectorizer.transform(X_valid)
    X_test_vec  = vectorizer.transform(X_test)

    model = LogisticRegression(max_iter=2000)
    model.fit(X_train_vec, y_train)

    # Evaluate
    print("\n=== VALIDATION RESULTS ===")
    valid_pred = model.predict(X_valid_vec)
    print(confusion_matrix(y_valid, valid_pred))
    print(classification_report(y_valid, valid_pred, digits=3))

    print("\n=== TEST RESULTS ===")
    test_pred = model.predict(X_test_vec)
    print(confusion_matrix(y_test, test_pred))
    print(classification_report(y_test, test_pred, digits=3))

    # Save
    joblib.dump(vectorizer, MODEL_DIR / "claim_vectorizer.joblib")
    joblib.dump(model, MODEL_DIR / "claim_model.joblib")
    print("\nSaved:")
    print(" - models/claim_vectorizer.joblib")
    print(" - models/claim_model.joblib")

if __name__ == "__main__":
    # Basic sanity checks
    for p in [TRAIN_PATH, VALID_PATH, TEST_PATH]:
        if not p.exists():
            raise FileNotFoundError(f"Missing file: {p} (expected under ./data/)")
    train_and_eval()