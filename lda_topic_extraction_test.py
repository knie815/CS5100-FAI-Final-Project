# Import necessary libraries
from gensim import corpora
from gensim.models import LdaModel
from gensim.models.coherencemodel import CoherenceModel
from pprint import pprint

# Sample documents
documents = [
    "Machine learning is an exciting field with endless possibilities.",
    "Natural language processing helps computers understand human language.",
    "Deep learning algorithms are used in various applications such as image recognition and speech synthesis.",
    "Data science involves extracting insights from data through statistical analysis and machine learning techniques."
]
# Tokenize the documents
tokenized_docs = [doc.lower().split() for doc in documents]
# Create a dictionary mapping each word to a unique id
dictionary = corpora.Dictionary(tokenized_docs)
# Convert tokenized documents into bag-of-words representation
corpus = [dictionary.doc2bow(doc) for doc in tokenized_docs]
# Train the LDA model
lda_model = LdaModel(corpus, num_topics=2, id2word=dictionary)
# Print the topics
pprint(lda_model.print_topics())

coherence_model = CoherenceModel(model=lda_model, texts=tokenized_docs, dictionary=dictionary, coherence='c_v')
coherence_score = coherence_model.get_coherence()
print(f"coherence_score: {coherence_score}")