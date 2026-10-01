# Example 4.24 - Question Answering

from transformers import pipeline

question_answerer = pipeline("question-answering")

context = """
Artificial intelligence is becoming an important part of multimedia technology.
It can be used to analyze images, recognize objects, generate content, and
improve digital experiences. Many multimedia applications use AI to make
creative and interactive content easier to develop.
"""

questions = [
    "What is becoming an important part of multimedia technology?",
    "What can artificial intelligence analyze?",
    "What can AI recognize?",
    "How can AI help multimedia applications?"
]

for question in questions:
    result = question_answerer(
        question=question,
        context=context
    )

    print("Question :", question)
    print("Answer   :", result["answer"])
    print("Score    :", round(result["score"], 4))
    print()