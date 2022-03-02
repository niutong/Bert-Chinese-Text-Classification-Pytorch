# coding: UTF-8
import torch
import numpy as np
from importlib import import_module
import argparse
import redis
from utils import build_dataset, build_iterator, get_time_dif

parser = argparse.ArgumentParser(description='Chinese Text Classification')
parser.add_argument('--model', type=str, required=True, help='choose a model: Bert, ERNIE')
args = parser.parse_args()

ONLINE_REDIS_HOST = "redis_node_5397.momo.com"
ONLINE_REDIS_PORT = 5397
ONLINE_REDIS_DB = 0

def predict(config, model):

    r_cli = redis.Redis(host=ONLINE_REDIS_HOST, port=ONLINE_REDIS_PORT, db=ONLINE_REDIS_DB)
    REDIS_SAORAO_KEY = "SaoraoUserNoBanGreetSet"

    model.load_state_dict(torch.load(config.save_path))
    model.eval()
    start_pos = 0
    max_pos = r_cli.zcard(REDIS_SAORAO_KEY)
    step_num = 500
    with torch.no_grad():

        while start_pos < max_pos:
            end_pos = min(start_pos+step_num, max_pos)
            msg_list = r_cli.zrange(REDIS_SAORAO_KEY, start_pos, end_pos, withscores=True)
            for item in msg_list:
                # 针对短文本
                # if len(item[0]) < 50:
                #     word_list = preprocess(item[0])
                #     # 分词小于等于3
                #     if 3 >= len(word_list) > 0:
                #         greet_summary_num += int(item[1])
                #         # produce_pattern(word_list, int(item[1]), item[0])

                outputs = model()
                predic_tag = torch.max(outputs.data, 1)[1].cpu().numpy()


if __name__ == '__main__':
    # dataset = 'THUCNews'  # 数据集
    dataset = 'GreetMsgs'  # 数据集

    model_name = args.model  # bert
    x = import_module('models.' + model_name)
    config = x.Config(dataset)
    np.random.seed(1)
    torch.manual_seed(1)
    torch.cuda.manual_seed_all(1)
    torch.backends.cudnn.deterministic = True  # 保证每次结果一样

    train_data, dev_data, test_data = build_dataset(config)
    train_iter = build_iterator(train_data, config)
    dev_iter = build_iterator(dev_data, config)
    test_iter = build_iterator(test_data, config)

    # train
    model = x.Model(config).to(config.device)

    loss_total = 0
    predict_all = np.array([], dtype=int)
    labels_all = np.array([], dtype=int)
    with torch.no_grad():
        for texts, labels in test_iter:
            outputs = model(texts)
            labels = labels.data.cpu().numpy()
            diff_tags = outputs - labels
            print("out_put:{}".format(outputs))
            print("labels:{}".format(labels))
            print("diff_tag:{}".format(diff_tags))
            break

