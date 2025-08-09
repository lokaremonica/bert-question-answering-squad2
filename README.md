# Question Answering on SQuAD 2.0 with BERT

## 📖 Overview

This project implements an end-to-end **extractive question answering** system using **BERT** (Bidirectional Encoder Representations from Transformers) fine-tuned on the **SQuAD 2.0** dataset.
Given a question and a context passage, the model identifies the span of text within the context that best answers the question.

---

## ✨ Features

* Uses **`bert-base-uncased`** with a span prediction head for start and end token classification.
* Processes and tokenizes **SQuAD 2.0** data into BERT-compatible format.
* Custom DataLoader for batching and shuffling.
* Tracks training/validation loss across epochs.
* Performs inference with custom answer span extraction.
* Evaluates using **Exact Match (EM)** and **F1 score**.

---

## 📊 Results Summary

* **Training Loss:** Decreased from \~5.57 to \~0.47 over 15 epochs.
* **Validation Loss:** Initially decreased, later increased—indicating **overfitting**.
* **Exact Match (EM):** 15.0%
* **F1 Score:** 16.07%

**Observations:**
Some predictions were correct (e.g., “Dreamgirls” → *Dreamgirls*), while others missed due to ambiguous contexts or fine-grained answer extraction challenges. Performance can be improved with regularization, early stopping, more data, or data augmentation.

---

## 🛠 Tech Stack

* **Python** 3.10+
* **PyTorch**
* **Transformers** (Hugging Face)
* **TorchText** for SQuAD 2.0 dataset
* **spaCy** (optional, for additional preprocessing)
* **Matplotlib**, **Pandas**, **NumPy**

---

## 🚀 Training

To train the BERT Question Answering model, you’ll run the training script with your chosen hyperparameters.
In this example, I train for 15 epochs with a batch size of 4, a learning rate of 2e-5, and a maximum sequence length of 384 tokens:`

---

## 💡 Inference Example
After training, you can use the model to answer questions from any given context.
Here’s how you’d load the trained model and ask it a question:

```python
from src.inference import get_answer
predicted_answer = get_answer(model, tokenizer,
    "When did Beyonce start becoming popular?",
    "Beyoncé... in the late 1990s..."
)
print(predicted_answer)
# Output: in the late 1990s
```
