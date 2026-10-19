import sys
from pathlib import Path

import pytest

# Make src/ importable
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from predict import load_model, predict_texts  # noqa: E402

MODEL_PATH = Path(__file__).resolve().parents[1] / "models" / "sentiment.joblib"


@pytest.fixture(scope="module")
def classifier():
    """Load the trained model once for all tests in this file."""
    if not MODEL_PATH.exists():
        pytest.skip("Model not found. Run src/train.py first.")
    return load_model(str(MODEL_PATH))


#def test_positive_sentence(classifier):
    #preds, _ = predict_texts(classifier, ["I love this movie, it was fantastic and inspiring!"])
    #assert preds[0] == 1


#def test_negative_sentence(classifier):
    #preds, _ = predict_texts(classifier, ["The service was terrible and the food was awful."])
    #assert preds[0] == 0

@pytest.mark.parametrize(
    "text, expected",
    [
        ("I love this movie, it was fantastic and inspiring!", 1),
        ("The service was terrible and the food was awful.", 0),
    ],
    ids=["positive", "negative"],
)
def test_obvious_sentiment(classifier, text, expected):
    preds, _ = predict_texts(classifier, [text])
    assert preds[0] == expected

def test_output_format(classifier):
    texts = ["great", "awful"]
    preds, probs = predict_texts(classifier, texts)
    assert len(preds) == len(texts)
    assert all(p in (0, 1) for p in preds)
    assert all(0.0 <= p <= 1.0 for p in probs)