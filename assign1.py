import nltk
import gensim.downloader as api
from gensim import models
from gensim.models import KeyedVectors
import numpy as np
import matplotlib.pyplot as plt
from sklearn.manifold import TSNE
# nltk.download()
sentences=nltk.corpus.brown.sents()
# len(sentences)
print(len(sentences))
# print(type(sentences))
sentences_list = list(sentences)
print(len(sentences_list))
a = nltk.sent_tokenize("In September term Ivan Allen had been charged by Court Judge. Mayor-nominate won in the primary hard-fought irregularities.")
b = [nltk.word_tokenize(s) for s in a]
print(b)
b = [[w.lower() for w in s] for s in b]
print(b)
lemmatizer = nltk.WordNetLemmatizer()
b = [[lemmatizer.lemmatize(w) for w in s] for s in b]
print(b)
wv = api.load("word2vec-google-news-300")
m1 = models.Word2Vec(sentences_list, sg=0)
m2 = models.Word2Vec(sentences_list,sg=1)
m1.wv.save_word2vec_format("my_wv1.txt")
m2.wv.save_word2vec_format("my_wv2.txt")
mv1 = models.KeyedVectors.load_word2vec_format("my_wv1.txt", binary=False)
mv2 = models.KeyedVectors.load_word2vec_format("my_wv2.txt", binary=False)
def reduce_dimensions(wv):
    num_dimensions = 2
    vectors = np.asarray(wv.vectors)
    labels = wv.index_to_key
    tsne = TSNE(n_components=num_dimensions, random_state=0)
    vectors = tsne.fit_transform(vectors)
    x_vals = [v[0] for v in vectors]
    y_vals = [v[1] for v in vectors]
    return x_vals, y_vals, labels
def plot_with_matplotlib(x_vals, y_vals, labels, words_to_plot):
    plt.figure(figsize=(12, 12))
    plt.scatter(x_vals, y_vals)
    for w in words_to_plot:
        if w in labels :
            i = labels.index(w)
            print("Plotting",w,"at",x_vals[i], y_vals[i])
            plt.annotate(w, (x_vals[i], y_vals[i]))
        else :
            print(w,"cannot be plotted because its word embedding is not given.")
    plt.show()

# Task 2: Visualization of word Embeddings: Custom Word Set
words_for_visualization = ['all', 'beer', 'buddy', 'Drink', 'guessed', 'Hell', 'Indian', 'Dye', 'now', 'right', 'said', 'the', 'your', 'bottle', 'was', 'still', 'in', 'my','foam','geysering']
m1_x_vals, m1_y_vals, m1_labels = reduce_dimensions(m1.wv)
m2_x_vals, m2_y_vals, m2_labels = reduce_dimensions(m2.wv)
plot_with_matplotlib(m1_x_vals, m1_y_vals, m1_labels,["beer","said","buddy","guessed","Hell"])
plot_with_matplotlib(m2_x_vals, m2_y_vals, m2_labels,["beer","said","buddy","guessed","Hell"])  

m1= KeyedVectors.load_word2vec_format("my_wv1.txt")
m2= KeyedVectors.load_word2vec_format("my_wv2.txt")

print(m1.evaluate_word_pairs("text.txt"))
print(m2.evaluate_word_pairs("text.txt"))
print(wv.evaluate_word_pairs("text.txt"))

dataset_words = ['Miraculousy', 'now', 'right', 'said', 'the']

similarity_m1 = {word1: [word for word, _ in m1.most_similar(word1, topn=5)] for word1 in dataset_words}
print("Synonyous words using Model m1:")
for word, similar_words in similarity_mod_1.items():
    print(f"{word}: {similar_words}")

similarity_m2 = {word2: [word for word, _ in m2.most_similar(word2, topn=5)] for word2 in dataset_words}
print("Synonymous words using Model m2:")
for word, similar_words in similarity_m2.items():
    print(f"{word}: {similar_words}")

similarity_google = {word3: [word for word, _ in wv.most_similar(word3, topn=5)] for word3 in dataset_words}
print("Synonymous words using Google's model:")
for word, similar_words in similarity_google.items():
    print(f"{word}: {similar_words}")