# Example 4.23 - Sentiment Analysis
# https://github.com/huggingface/transformers
# pip install transformers
# https://aka.ms/vs/16/release/vc_redist.x64.exe

from transformers import pipeline

# Membuat sentiment analysis pipeline
classifier = pipeline("sentiment-analysis")

# Kalimat positif
positive_sentences = [
    "I really enjoyed this movie.",
    "The story was amazing and the actors were excellent.",
    "This is one of the best movies I have ever watched."
]

# Kalimat negatif
negative_sentences = [
    "I did not enjoy this movie.",
    "The story was boring and the acting was disappointing.",
    "This is one of the worst movies I have ever watched."
]

print("=== Positive Sentences ===")

for sentence in positive_sentences:
    result = classifier(sentence)
    print(f"\nSentence: {sentence}")
    print(f"Result: {result}")

print("\n=== Negative Sentences ===")

for sentence in negative_sentences:
    result = classifier(sentence)
    print(f"\nSentence: {sentence}")
    print(f"Result: {result}")