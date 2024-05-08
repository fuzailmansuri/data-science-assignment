#!/usr/bin/env python
# coding: utf-8

# In[1]:


import pandas as pd
import nltk
import re
import requests
from textblob import TextBlob
from bs4 import BeautifulSoup

def analyze_text(text):
    words = nltk.word_tokenize(text)
    sentences = nltk.sent_tokenize(text)

    blob = TextBlob(text)
    sentiment_scores = blob.sentiment

    positive_score = sentiment_scores.polarity if sentiment_scores.polarity > 0 else 0
    negative_score = -sentiment_scores.polarity if sentiment_scores.polarity < 0 else 0

    avg_sentence_length = sum(len(sentence.split()) for sentence in sentences) / len(sentences)
    word_count = len(words)
    syllable_count = sum(len(re.findall(r'[aeiouy]+', word.lower())) for word in words)
    avg_word_length = sum(len(word) for word in words) / len(words) if len(words) > 0 else 0
    personal_pronouns = sum(1 for word in words if word.lower() in {'i', 'me', 'my', 'mine', 'we', 'us', 'our', 'ours', 'you', 'your', 'yours', 'he', 'him', 'his', 'she', 'her', 'hers', 'they', 'them', 'their', 'theirs'})
    
    if word_count > 0:
        percentage_of_complex_words = sum(len(word.split('-')) > 1 or len(word.split('_')) > 1 for word in words) / word_count
    else:
        percentage_of_complex_words = 0

    fog_index = 0.4 * (avg_sentence_length + 100 * percentage_of_complex_words)

    subjectivity_score = blob.subjectivity

    return (positive_score, negative_score, sentiment_scores.polarity, subjectivity_score, avg_sentence_length,
            percentage_of_complex_words, fog_index, word_count / len(sentences), 
            sum(len(word.split('-')) > 1 or len(word.split('_')) > 1 for word in words),
            word_count, syllable_count / word_count, personal_pronouns, avg_word_length)

df = pd.read_excel('C:/Users/fuzai/Downloads/Input.xlsx')
headers = ['URL_ID', 'URL', 'POSITIVE SCORE', 'NEGATIVE SCORE', 'POLARITY SCORE', 
           'SUBJECTIVITY SCORE', 'AVG SENTENCE LENGTH', 'PERCENTAGE OF COMPLEX WORDS', 
           'FOG INDEX', 'AVG NUMBER OF WORDS PER SENTENCE', 'COMPLEX WORD COUNT', 
           'WORD COUNT', 'SYLLABLE PER WORD', 'PERSONAL PRONOUNS', 'AVG WORD LENGTH']

output_df = pd.DataFrame(columns=headers)

for index, row in df.iterrows():
    output = analyze_text(row['URL'])
    output_df.loc[index] = [row['URL_ID'], row['URL']] + list(output)

output_file = 'output.xlsx'
output_df.to_excel(output_file, index=False)

print("Output saved to:", output_file)


# In[ ]:




