# from pip._internal import main as pipmain
# pipmain(['install', 'git+https://github.com/huggingface/transformers.git'])
import pandas as pd
import numpy as np
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM
import tensorflow as tf
import tensorflow_hub as hub
from tensorflow import keras
import spacy
# !python -m spacy download en
nlp = spacy.load('en')

from sklearn.metrics.pairwise import cosine_similarity
from sklearn.metrics.pairwise import euclidean_distances

import nltk
# nltk.download('punkt')
# nltk.download('stopwords')
# nltk.download('averaged_perceptron_tagger')
# nltk.download('wordnet')
from nltk.tokenize import sent_tokenize

product_embeddings = pd.read_pickle('data/product_embeds.pkl') # Run google colab GPU to get the embeddings of all unique products in the dataset

def euclidean_similarity(embed1, embed2):
    distance_matrix = euclidean_distances(embed1, embed2)
    similarity_matrix = 1 - distance_matrix/np.max(distance_matrix)
    return similarity_matrix

def get_sentiment_scores(review, sentiment = 'positive'):
    """
    Returns sentiment scores for each sentence in one review and embeddings of review sentences
    
    args:
    review(str): review to have sentiment scores
    sentiment(str): default to 'positive'
    
    returns:
    sentiment_scores(np.array): sentiment score for each sentence. shape = (1, num_sentences)
    embeddings(tf.Tensor): Universal Sentence Encoder embedding. shape = (1, 512) 
    """
    
    review_sentences = sent_tokenize(review)
    embeddings = embed(tf.convert_to_tensor(review_sentences))
    sentiment_scores = 1
    if sentiment is not None:
        sentiment_scores = model.predict(embeddings)[:, 1].reshape(-1,1)# 'positive'
        if sentiment == 'negative':
            sentiment_scores = 1 - sentiment_scores
    return sentiment_scores, embeddings

def similarity_scores(embeddings, query, keywords = True, euclidean = False):
    """
    Returns a matrix of similarities between the sentences of a review(in embeddings) and the keywords or query sentences.
    
    args:
    embeddings(tf.Tensor): USE embeddings of a review sentences. shape = (1, 512)
    query(str): query that will be tokenized into bag-of-keywords or sentences
    keywords(bool): whether to use bag-of-words(keywords) in the query for computing the similarity score
    euclidean(bool): whether to use euclidean distance to compute similarity matrix
    
    returns:
    similarity_scores(np.array): similarity score matrix. shape = (num_sentences_in_a_review, num_keywords or num_query_sentences)
    """
    
    if keywords:
        keywords = list(dict.fromkeys([token.lemma_ for token in nlp(query.lower()) if token.pos_ in ['NOUN', 'PROPN', 'ADJ', 'VERB', 'X']]))
        keywords_len = len(keywords)
        embeddings_k = tf.concat([embed(keywords), embeddings], axis = 0).numpy()
        similarity_matrix = euclidean_similarity(embeddings_k, embeddings_k) if euclidean else cosine_similarity(embeddings_k, embeddings_k)
        return similarity_matrix[keywords_len:, :keywords_len], keywords
    else:
        query_sentences = sent_tokenize(query)
        query_len = len(query_sentences)
        embeddings_q = tf.concat([embed(query_sentences), embeddings], axis = 0).numpy()
        similarity_matrix = euclidean_similarity(embeddings_q, embeddings_q) if euclidean else cosine_similarity(embeddings_q, embeddings_q)
        return similarity_matrix[query_len:, :query_len], query_sentences

def sentimental_similarity_score_of_a_review(review, query, sentiment = 'positive', emphasized_keywords = None, euclidean = False):
    """
    Returns a positive similarity score between a review and a query.
    
    args:
    review(str): review
    query(str): query
    sentiment(str): If not 'negative', it's considered as 'positive'. 'positive' is the default.
    emphasized_keywords(list): list of keywords to be emphasized (currently 1: 0.01)
    euclidean(bool): whether to use euclidean distance for similarity matrix.
    
    returns:
    similarity score(int): (positive) similarity score between a review and a query
    """
    
    if review is None:
        return np.nan
    
    sentiment_scores, embeddings = get_sentiment_scores(review, sentiment)
    similarity_scores_keywords, keywords = similarity_scores(embeddings, query, euclidean = euclidean)
    similarity_scores_sentences, query_sentences = similarity_scores(embeddings, query, keywords = False, euclidean = euclidean)
    keywords_len, emphasized_keywords_len = len(keywords), len(emphasized_keywords)
    
    if emphasized_keywords and emphasized_keywords_len < keywords_len:
        emphasized_keywords = [token.lemma_ for token in nlp(' '.join(emphasized_keywords).lower()) if token.pos_ in ['NOUN', 'PROPN', 'ADJ', 'VERB', 'X']]
        regular_keywords_len = keywords_len - emphasized_keywords_len
        heavy_weight, light_weight = keywords_len * 0.9 / emphasized_keywords_len, keywords_len * 0.1 / regular_keywords_len
        weights_keywords = [heavy_weight if keyword in emphasized_keywords else light_weight for keyword in keywords]
        weights_sentences = [sum([heavy_weight if keyword in sentence else light_weight for keyword in emphasized_keywords]) for sentence in query_sentences]
    else:
        weights_keywords, weights_sentences = 1, 1
        
    keyword_scores = (similarity_scores_keywords * sentiment_scores * weights_keywords).max(axis=1)
    pos_sim_score_keywords = np.sqrt(keyword_scores.mean())

    query_sentence_scores = (similarity_scores_sentences * sentiment_scores * weights_sentences).max(axis=1)
    pos_sim_score_sentences = np.sqrt(query_sentence_scores.mean())
    
    return np.min([pos_sim_score_keywords, pos_sim_score_sentences])

def get_similarity_score_with_product(query, euclidean = False):
    """
    Returns most similar product indices and its similarity scores with respect to the query
    
    args:
    query(str)
    euclidean(bool): whether to use euclidean distance for similarity matrix
    
    returns:
    product's indices of similarity and the similarity scores in descending order of similarity
    """
    similarity_scores_keywords, keywords = similarity_scores(product_embeddings, query, euclidean = euclidean)
    similarity_scores_sentences, query_sentences = similarity_scores(product_embeddings, query, keywords = False, euclidean = euclidean)

    keyword_scores = similarity_scores_keywords.max(axis=1)   
    query_sentence_scores = similarity_scores_sentences.max(axis=1)
    
    sim_scores = np.concatenate((keyword_scores.reshape(-1,1), query_sentence_scores.reshape(-1,1)), axis = 1).min(axis = 1)
    
    return np.argsort(sim_scores)[::-1], np.sort(sim_scores)[::-1]
