# Example 4.22 - NLTK Text Analysis
# Modified version
# https://www.nltk.org/
# pip install nltk

import nltk

# Download resource yang diperlukan
nltk.download('maxent_ne_chunker')
nltk.download('words')
nltk.download('punkt_tab')
nltk.download('averaged_perceptron_tagger_eng')
nltk.download('maxent_ne_chunker_tab')

# Teks yang akan dianalisis
sentence = """
Multimedia technology combines text, images, audio, video, and
animation to create interactive digital experiences.
Artificial intelligence is also becoming an important technology
in multimedia development. Companies such as Adobe and Microsoft
develop software that helps designers and developers create
digital content more efficiently.
"""

# Tokenisasi
tokens = nltk.word_tokenize(sentence)
print("Tokens:")
print(tokens)

# Part-of-Speech Tagging
tagged = nltk.pos_tag(tokens)
print("\nPOS Tagging:")
print(tagged)

# Named Entity Recognition
entities = nltk.chunk.ne_chunk(tagged)
print("\nNamed Entities:")
print(entities)