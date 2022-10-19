# coding: UTF-8

from top2vec import Top2Vec
from sklearn.datasets import fetch_20newsgroups
from tqdm import tqdm


def load_dataset(path):
    contents = []
    with open(path, 'r', encoding='UTF-8', errors='ignore') as f:
        for line in tqdm(f):
            lin = line.strip()
            if not lin:
                continue

            contents.append(line)

    return contents




if __name__ == '__main__':

    newsgroups = fetch_20newsgroups(subset='all', remove=('headers', 'footers', 'quotes'))

    data_set = load_dataset("/home/recommend_nearby/work_space/niu.tong/data/good_content_rate_sorted_1010_good_male.txt")

    model = Top2Vec(documents=newsgroups.data, speed="fast-learn", workers=8)

    print("model.get_num_topics", model.get_num_topics())
    print("model.get_topic_sizes()", model.get_topic_sizes())
    print("model.get_topics(5)", model.get_topics(5))
