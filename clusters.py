import numpy as np
from sklearn.manifold import TSNE
#from sklearn.decomposition import PCA
import matplotlib.pyplot as plt
import mplcursors
import spacy

#can use en_core_web_sm instead for faster results

nlp = spacy.load('en_core_web_lg')
embs = []
labels = []

with open("hn_history.txt", "r") as f:
        phrases = f.read().splitlines()

        for phrase in phrases:
            emb = [token.vector for token in nlp(phrase)]
            emb = np.mean(emb, axis=0)
            if emb.any():
                embs.append(emb)
                labels.append(phrase)

embs = np.array(embs)

#pca = PCA(n_components=30)
#embs = pca.fit_transform(embs)

tsne = TSNE(n_components=2)
X_tsne = tsne.fit_transform(embs)

scatter = plt.scatter(X_tsne[:, 0], X_tsne[:, 1])
cursor = mplcursors.cursor(scatter, hover=True)

def label_formatter(sel):
    index = sel.target.index
    x, y = sel.target
    label = labels[index]
    sel.annotation.set_text(label)
    sel.annotation.xy = (x, y)

cursor.connect("add", label_formatter)

plt.show()

