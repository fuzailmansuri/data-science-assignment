import pandas as pd
import nltk
import re
import requests
from textblob import TextBlob
from bs4 import BeautifulSoup

def analyze_text(url, url_id):
    response = requests.get(url)
    soup = BeautifulSoup(response.content, 'html.parser')
    
    article_element = soup.find('div', class_='td-post-content tagdiv-type')
    title_element = soup.find('h1', class_='entry-title')
    
    if article_element and title_element:
        article_text = article_element.get_text(separator='\n')
        title = title_element.get_text().strip()
        
        filename = f"{url_id}-assignment.txt"
        with open(filename, 'w', encoding='utf-8') as file:
            file.write(f"Title: {title}\n\n")
            file.write(article_text)
            
        print(f"Article saved successfully as '{filename}'")
        
        words = nltk.word_tokenize(article_text)
        sentences = nltk.sent_tokenize(article_text)

        blob = TextBlob(article_text)
        sentiment_scores = blob.sentiment

        positive_score = max(sentiment_scores.polarity, 0)
        negative_score = max(-sentiment_scores.polarity, 0)
        polarity_score = sentiment_scores.polarity
        subjectivity_score = blob.subjectivity
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

        return (positive_score, negative_score, polarity_score, subjectivity_score, avg_sentence_length,
                percentage_of_complex_words, fog_index, word_count / len(sentences), 
                sum(len(word.split('-')) > 1 or len(word.split('_')) > 1 for word in words),
                word_count, syllable_count / word_count, personal_pronouns, avg_word_length)
    else:
        print("Article content or title not found on the page.")
        return (0,) * 14

df = pd.read_excel('C:/Users/fuzai/Downloads/Input.xlsx')
headers = ['URL_ID', 'URL', 'POSITIVE SCORE', 'NEGATIVE SCORE', 'POLARITY SCORE', 
           'SUBJECTIVITY SCORE', 'AVG SENTENCE LENGTH', 'PERCENTAGE OF COMPLEX WORDS', 
           'FOG INDEX', 'AVG NUMBER OF WORDS PER SENTENCE', 'COMPLEX WORD COUNT', 
           'WORD COUNT', 'SYLLABLE PER WORD', 'PERSONAL PRONOUNS', 'AVG WORD LENGTH']

output_df = pd.DataFrame(columns=headers)

for index, row in df.iterrows():
    print("URL:", row['URL'])
    print("URL_ID:", row['URL_ID'])
    
    output = analyze_text(row['URL'], row['URL_ID'])
    
    if output is not None and len(output) == len(headers) - 2:
        new_row_data = [row['URL_ID'], row['URL']]
        for i in range(len(headers) - 2):
            new_row_data.append(0)
        new_row = pd.DataFrame([new_row_data], columns=headers)
        output_df = pd.concat([output_df, new_row], ignore_index=True)
    else:
        new_row_data = [row['URL_ID'], row['URL']]
        for i in range(len(headers) - 2):
            new_row_data.append(0)
        new_row = pd.DataFrame([new_row_data], columns=headers)
        output_df = pd.concat([output_df, new_row], ignore_index=True)

output_file = 'C:/Users/fuzai/Downloads/final_output.xlsx'
output_df.to_excel(output_file, index=False)

print("Output saved to:", output_file)
