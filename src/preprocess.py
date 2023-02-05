import re

import pandas as pd
import numpy as np

from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
from nltk.stem import SnowballStemmer

import multiprocessing as mp
from tqdm import tqdm

from src.utils import get_data

def clean_junk(sntnc):
    all_letters = r"[A-Za-z ]+"

    removed_links = re.sub(r'http\S+', '', sntnc)

    cleaned_txt_list = re.findall(all_letters, removed_links)

    cleaned_txt = " ".join(cleaned_txt_list).lower()
    
    return cleaned_txt
    
    
def remove_stopwords(sntnc):
    stop_words = set(stopwords.words('english'))
    no_stop_words = [word for word in sntnc if word not in stop_words]
    return no_stop_words

def tok(sntnc, tokenizer, stemmer):
    #first remove non alphabetical characters and links and lower the sntnc
    cleaned_txt = clean_junk(sntnc)
    
    #tokenize the sentence
    tokenized = tokenizer(cleaned_txt)

    #remove stop words
    txt = remove_stopwords(tokenized)
    
    #stemm the words
    stemmed = [stemmer.stem(word) for word in txt]
    
    return stemmed

def tokenizer(pair):
    idx, sntnc = pair
    stemmer = SnowballStemmer("english")
    return idx, tok(sntnc, word_tokenize, stemmer)


if __name__=="__main__":
    data = get_data()
    
    X, y = data['text'], data['label']

    corpus = X.to_dict()
    tok_corpus = np.empty(len(corpus), dtype='object')

    with mp.Pool(mp.cpu_count()-1) as pool:
        for idx,txt in tqdm(pool.imap_unordered(tokenizer, corpus.items()), total=len(corpus)):
            #res = (idx, sntnc)
            tok_corpus[idx] = txt
    
    print("Corpus tokenization is complete")

    tok_corpus = tok_corpus.tolist()

    df = pd.DataFrame({'text': tok_corpus, 'sentiment': y})

    df['text'] = df['text'].apply(lambda x: ' '.join(x))
    
    df = df.sample(frac=1)

    #dropping row with empty strings
    df = df[df['text'].astype(bool)]
    
    #Saving processed data
    df.to_csv('../dataset/processed_data.csv', index=False)
    print("saved the data")