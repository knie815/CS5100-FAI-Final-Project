# Import necessary libraries
from gensim import corpora
from gensim.models import LdaModel
from gensim.models.coherencemodel import CoherenceModel
from nltk.corpus import sentiwordnet as swn
from pprint import pprint
import nltk

# Download NLTK data
nltk.download('sentiwordnet')
nltk.download('wordnet')

def get_sentiment(word):
    """Calculate the sentiment score of a word using SentiWordNet."""
    synsets = list(swn.senti_synsets(word))
    if not synsets:
        return {"pos": 0, "neg": 0, "obj": 0}
    pos_score = sum([syn.pos_score() for syn in synsets]) / len(synsets)
    neg_score = sum([syn.neg_score() for syn in synsets]) / len(synsets)
    obj_score = 1 - (pos_score + neg_score)
    return {"pos": pos_score, "neg": neg_score, "obj": obj_score}

def main():
    # Sample documents
    documents = [
        "Machine learning is an exciting field with endless possibilities.",
        "Natural language processing helps computers understand human language.",
        "Deep learning algorithms are used in various applications such as image recognition and speech synthesis.",
        "Data science involves extracting insights from data through statistical analysis and machine learning techniques."
    ]

    # Tokenize the documents
    tokenized_docs = [doc.lower().split() for doc in documents]

    # Set up the dictionary
    dictionary = corpora.Dictionary(tokenized_docs)

    # Convert tokenized documents into bag-of-words representation
    corpus = [dictionary.doc2bow(doc) for doc in tokenized_docs]

    # Train the LDA model
    lda_model = LdaModel(corpus, num_topics=2, id2word=dictionary)

    # Print the topics
    # pprint(lda_model.print_topics())

    # Calculate sentiment scores for each topic
    topic_scores = {}
    for topic_id, words in lda_model.show_topics(num_topics=2, num_words=10, formatted=False):
        sentiment_scores = {"pos": 0, "neg": 0, "obj": 0}
        for word, _ in words:
            sentiment = get_sentiment(word) # get sentiword scores
            sentiment_scores["pos"] += sentiment["pos"]
            sentiment_scores["neg"] += sentiment["neg"]
            sentiment_scores["obj"] += sentiment["obj"]
        topic_scores[topic_id] = sentiment_scores
        print(f"Topic {topic_id} Sentiment: {sentiment_scores}")

    # Calculate and print coherence score
    coherence_model = CoherenceModel(model=lda_model, texts=tokenized_docs, dictionary=dictionary, coherence='c_v')
    coherence_score = coherence_model.get_coherence()
    # print(f"Coherence Score: {coherence_score}")

if __name__ == "__main__":
    main()
