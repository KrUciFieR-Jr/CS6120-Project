from nltk.corpus import stopwords
import gensim
import re
import string
stop_words = stopwords.words('english')
stop_words.extend(['from', 'subject', 're', 'edu', 'use'])

def prerprocess_inline(text):
    """
    Preprocessing function for text.

    :param text:
    :return:
    """
    cleaned_data = []

    text = str(text).lower()
    text = re.sub('\[.*?\]', '', text)
    text = re.sub('https?://\S+|www\.\S+', '', text)
    text = re.sub('<.*?>+', '', text)
    text = re.sub('[%s]' % re.escape(string.punctuation), '', text)
    text = re.sub('\n', '', text)
    text = re.sub('\w*\d\w*', '', text)

    for token in gensim.utils.simple_preprocess(text):
        if token not in gensim.parsing.preprocessing.STOPWORDS and len(token) > 2 and token not in stop_words:
            cleaned_data.append(token)

    return cleaned_data

def data_preprocessing(df):
    """

    :param df: dataframe to clean
    :return: cleaned_dataframe
    """
    """
    We are going to predict the news based on title and text separately, this is one of 
    our experiments
    """
    # Clean the title
    df['clean_title'] = df['title'].apply(prerprocess_inline)
    df['clean_joined_title']=df['clean_title'].apply(lambda x:" ".join(x))

    # Clean the text
    df['clean_text'] = df['text'].apply(prerprocess_inline)
    df['clean_joined_text']=df['clean_text'].apply(lambda x:" ".join(x))

    # Matching similar class to the subject
    df.subject=df.subject.replace({'politics':'PoliticsNews','politicsNews':'PoliticsNews'})


    return df