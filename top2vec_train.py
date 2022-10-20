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

    #newsgroups = fetch_20newsgroups(subset='all', remove=('headers', 'footers', 'quotes'))

    # data_set = load_dataset("/home/recommend_nearby/work_space/niu.tong/data/good_content_rate_sorted_1010_good_male.txt")

    # model = Top2Vec(documents=data_set, speed="learn", workers=8)

    # model.save("/home/recommend_nearby/work_space/niu.tong/model_save/good_content_1010_male_topics")
    model = Top2Vec.load("/home/recommend_nearby/work_space/niu.tong/model_save/good_content_1010_male_topics")
    topic_nums = model.get_num_topics()

    topic_sizes, topic_nums_array = model.get_topic_sizes()

    topic_words, word_scores, topic_nums_arry_b = model.get_topics(topic_nums)
    print("summary topic_nums:", topic_nums)
    for topic_index in range(topic_nums):
        topic_words_elems = topic_words[topic_index]
        word_scores_elems = word_scores[topic_index]
        documents, document_scores, document_ids = model.search_documents_by_topic(topic_num=topic_index, num_docs=500)
        print("index_" + str(topic_index), "keywords",  ",".join(topic_words_elems.tolist()), "scores",
              ",".join(word_scores_elems.tolist()))

        for doc, score, doc_id in zip(documents, document_scores, document_ids):
            print(f"index_{topic_index} document_{doc_id} Score:{score} content:{doc}")
