# Twitter Sentiment Analysis with DistilBERT

This project uses DistilBERT to classify tweet sentiment (positive, neutral, negative).

## 1. Live Demo
Try the interactive Gradio app here:  
https://huggingface.co/spaces/damiangohrh123/twitter-sentiment-distilbert

## 2. Project Features
- Transformer-based model (DistilBERT) for high accuracy
- Custom text preprocessing pipeline for tweets
- Optimal thresholding for best precision/recall tradeoff
- Interactive Gradio UI for public inference
- GPU-accelerated training
- Clean and reproducible workflow (Jupyter + Python scripts)

## 3. Model & Methodology
### Data Preprocessing
- Lowercasing
- Removing URLs, mentions, and special characters
- Tokenization using `DistilBertTokenizerFast`
### Model: DistilBERT
- 6 Transformer layers
- 66M parameters
- Faster, lighter version of BERT
- Fine-tuned on ~1.6M tweets
### Training Setup
- Optimizer: AdamW
- Learning Rate: 3e-5
- Loss: CrossEntropy
- Batch Size: 32
- Trained for 3 epochs

## 4. Evaluation
The model is evaluated using:
- Precision
- Recall
- F1-score
- Confusion Matrix
- ROC Curve
- Threshold Optimization (instead of default 0.5, we compute the best performing threshold)
You can find full results in the notebook.

## 5. Deployment (Hugging Face Spaces)
The app uses Gradio for a clean, simple UI:
- User enters a tweet
- Model predicts Positive / Negative
- Shows confidence score
Running locally:  
`pip install -r requirements.txt`  
`python app.py`

## 6. License
This project is open-source under the MIT License.
