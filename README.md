# C.A.L.M. AI: Cognitive Assistant for Logging and Monitoring Emotions

An emotion-aware web app. It detects the emotion in what you type, replies through an emotion-adaptive chatbot, and tracks your mood over time. The core is a **fine-tuned DistilBERT** transformer.

**Course:** Artificial Intelligence & Machine Learning, VIT Vellore
**Author:** Dhruv Lazar
**Guide:** Dr. Padmini T. N.
**Tools:** Python, PyTorch, HuggingFace Transformers, Streamlit, Google Colab, Ngrok

---

## Features

- **Emotion detection:** classifies text into 6 emotions (joy, sadness, anger, fear, love, surprise), with a confidence score.
- **Emotion-adaptive chatbot:** empathetic replies to sad messages, encouragement for happy ones.
- **Mood history:** logs every message, the detected emotion and the confidence, with a trend graph and a reset button.
- **Runs in the cloud:** trained and hosted on Google Colab, shared through an Ngrok public link, so no local server is needed.

## How it works

```
User text → DistilBERT tokenizer → fine-tuned DistilBERT → softmax → emotion + confidence
                                                                      ↓
                                     chatbot reply  ←  emotion-specific response logic
                                                                      ↓
                                                     session history + trend chart (Streamlit)
```

## Dataset

[HuggingFace `emotion` dataset](https://huggingface.co/datasets/dair-ai/emotion), about 20,000 short English sentences:
- Train / validation / test split: 16k / 2k / 2k
- Labels: 6 emotions

**Preprocessing:**
1. Lowercase the text.
2. Strip emojis, URLs and symbols.
3. Tokenize with the DistilBERT tokenizer.
4. Pad or truncate to 128 tokens.

## Model and training

| Setting | Value |
|---|---|
| Base model | `distilbert-base-uncased` (66M parameters, about 40% smaller than BERT) |
| Loss | Cross-entropy |
| Optimizer | AdamW, learning rate 5e-5 |
| Batch size / epochs | 16 / 10 |
| Environment | Google Colab (GPU) |

## Results

- High-confidence predictions (often above 0.85) on text with clear emotional cues.
- Lower confidence on ambiguous or neutral sentences, as expected for this kind of model.
- Fast responses thanks to the lightweight model, with no noticeable lag in the app.

## Future work

- Save mood history permanently in a database
- Multi-label emotions
- Voice + text (multimodal) input
- An LLM-based chatbot instead of template replies
