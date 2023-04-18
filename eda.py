import matplotlib.pyplot as plt
import seaborn as sns
import plotly.express as px
import os
from nltk.corpus import stopwords
from wordcloud import WordCloud, STOPWORDS
import nltk

stop_words = stopwords.words('english')
stop_words.extend(['from', 'subject', 're', 'edu', 'use'])

def create_visualizations(df):
    """
    Function used to create visulaizations based on our data explorations

    :param df: cleaned data frame
    :return: None
    """
    sub_tf_df = df.groupby('target').apply(lambda x:x['title'].count()).reset_index(name='Counts')
    sub_tf_df.target.replace({0:'False',1:'True'},inplace=True)
    fig = px.bar(sub_tf_df, x="target", y="Counts",
                 color='Counts', barmode='group',
                 height=400)

    if not os.path.exists("images"):
        os.mkdir("images")

    # Image to check our output distribution.
    fig.write_image("images/target_group_count.png")


    # Categories based count
    sub_check = df.groupby('subject').apply(lambda x:x['title'].count()).reset_index(name='Counts')
    fig=px.bar(sub_check,x='subject',y='Counts',color='Counts',title='Count of News Articles by Subject')
    fig.write_image("images/count_categories_wise.png")

    # Word cloud for Real news based on cleaned title
    real_wc = WordCloud(max_words = 2000 , width = 1600 , height = 800 , stopwords =
    stop_words).generate(" ".join(df[df.target == 1].clean_joined_title))
    real_wc.to_file('images/real_news_wordcloud.png')

    # Word cloud for Fake news based on cleaned title
    fake_wc = WordCloud(max_words = 2000 , width = 1600 , height = 800 , stopwords =
    stop_words).generate(" ".join(df[df.target == 1].clean_joined_title))
    fake_wc.to_file('images/fake_news_wordcloud.png')

    # Does title count affect our classification ?
    fig_title_count = px.histogram(x = [len(nltk.word_tokenize(x)) for x in
                                        df.clean_joined_title], nbins = 50)
    fig_title_count.write_image("images/title_count.png")

