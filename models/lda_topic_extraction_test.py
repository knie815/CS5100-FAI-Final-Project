from gensim import corpora
from gensim.models import LdaModel
from gensim.models.coherencemodel import CoherenceModel
from pprint import pprint

documents = [
    "Machine learning is an exciting field with endless possibilities.",
    "Natural language processing helps computers understand human language.",
    "Deep learning algorithms are used in various applications such as image recognition and speech synthesis.",
    "Data science involves extracting insights from data through statistical analysis and machine learning techniques."
]
tokenized_docs = [doc.lower().split() for doc in documents]
dictionary = corpora.Dictionary(tokenized_docs)
corpus = [dictionary.doc2bow(doc) for doc in tokenized_docs]
lda_model = LdaModel(corpus, num_topics=2, id2word=dictionary)
pprint(lda_model.print_topics())
# Print output:
# [(0,
#   '0.038*"learning" + 0.037*"and" + 0.032*"such" + 0.032*"recognition" + '
#   '0.030*"in" + 0.030*"various" + 0.030*"applications" + 0.029*"algorithms" + '
#   '0.028*"used" + 0.028*"speech"'),
#  (1,
#   '0.051*"learning" + 0.043*"data" + 0.039*"machine" + 0.033*"and" + '
#   '0.026*"statistical" + 0.026*"through" + 0.026*"analysis" + '
#   '0.026*"extracting" + 0.026*"insights" + 0.025*"from"')]

coherence_model = CoherenceModel(model=lda_model, texts=tokenized_docs, dictionary=dictionary, coherence='c_v')
coherence_score = coherence_model.get_coherence()
print(f"coherence_score: {coherence_score}")
# coherence_score: 0.46366842544437337