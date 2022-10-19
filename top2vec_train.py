# coding: UTF-8

from top2vec import Top2Vec
from sklearn.datasets import fetch_20newsgroups

if __name__ == '__main__':

    newsgroups = fetch_20newsgroups(subset='all', remove=('headers', 'footers', 'quotes'))

    model = Top2Vec(documents=newsgroups.data, speed="learn", workers=8)

    print("model.get_num_topics", model.get_num_topics())
    print("model.get_topic_sizes()", model.get_topic_sizes())
    print("model.get_topics(70)", model.get_topics(70))
