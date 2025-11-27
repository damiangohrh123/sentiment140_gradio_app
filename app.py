import torch
from transformers import DistilBertTokenizerFast, DistilBertForSequenceClassification
import gradio as gr
import json
import re
from pathlib import Path

device = "cuda" if torch.cuda.is_available() else "cpu"

# 1. Clean tweet function
def clean_tweet(text):
    text = str(text)
    # Remove URLs
    text = re.sub(r"http\S+|www\S+|https\S+", "", text)
    # Remove mentions
    text = re.sub(r"@\w+", "", text)
    # Remove hashtags
    text = re.sub(r"#\w+", "", text)
    # Remove emojis
    emoji_pattern = re.compile(
        "["
        u"\U0001F600-\U0001F64F"
        u"\U0001F300-\U0001F5FF"
        u"\U0001F680-\U0001F6FF"
        u"\U0001F1E0-\U0001F1FF"
        "]+", flags=re.UNICODE
    )
    text = emoji_pattern.sub(r"", text)
    # Remove punctuation except letters/numbers/spaces
    text = re.sub(r"[^A-Za-z0-9\s]", "", text)
    # Lowercase and strip
    text = re.sub(r"\s+", " ", text).strip().lower()
    return text

# 2. Load model and tokenizer
model_dir = Path("distilbert_sentiment_model_final")

model = DistilBertForSequenceClassification.from_pretrained(model_dir)
model.to(device)
model.eval()

tokenizer = DistilBertTokenizerFast.from_pretrained(model_dir)

with open(model_dir / "best_threshold.json") as f:
    best_threshold = json.load(f)["best_threshold"]

# 3. Prediction function
def predict_sentiment(tweet):
    clean_text = clean_tweet(tweet)

    inputs = tokenizer(
        clean_text,
        padding='max_length',
        truncation=True,
        max_length=128,
        return_tensors='pt'
    ).to(device)

    with torch.no_grad():
        outputs = model(**inputs)
        probs = torch.softmax(outputs.logits, dim=1)
        positive_prob = probs[0, 1].item()

    label = "Positive" if positive_prob >= best_threshold else "Negative"
    return label, round(positive_prob, 4)

# 4. Gradio interface
interface = gr.Interface(
    fn=predict_sentiment,
    inputs=gr.Textbox(lines=2, placeholder="Enter a tweet here..."),
    outputs=[
        gr.Label(num_top_classes=1, label="Predicted Sentiment"),
        gr.Textbox(label="Positive Probability")
    ],
    title="Twitter Sentiment Analysis (DistilBERT)",
    description="Enter a tweet to see the predicted sentiment and probability."
)

# Launch the app
if __name__ == "__main__":
    interface.launch()
