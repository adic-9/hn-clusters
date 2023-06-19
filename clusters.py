import numpy as np
from sklearn.manifold import TSNE
import matplotlib.pyplot as plt
import mplcursors
import spacy
import streamlit as st

#can use en_core_web_sm instead for faster results
#nlp = spacy.load('en_core_web_lg')
embs = []
labels = []

def main():
    st.title("hn cluster")
    
    # File uploader
    uploaded_file = st.file_uploader("Choose a file")
    
    if uploaded_file is not None:
        # Display file details
        file_details = {"FileName": uploaded_file.name, "FileType": uploaded_file.type, "FileSize": uploaded_file.size}
        st.write(file_details)
        
        file_content = uploaded_file.read()).splitlines()

        st.write("File Content:")
        st.code(file_content)
        
""""
        process_file(uploaded_file)

def process_file(f):
    phrases = f.read().splitlines()
    for phrase in phrases:
        emb = [token.vector for token in nlp(phrase)]
        emb = np.mean(emb, axis=0)
        if emb.any():
            embs.append(emb)
            labels.append(phrase)

    embs = np.array(embs)
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
"""
plt.show()

if __name__ == "__main__":
    main()
