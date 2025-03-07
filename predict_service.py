# coding: UTF-8
import torch
from importlib import import_module
from tqdm import tqdm
import numpy as np
import torch.nn as nn

PAD, CLS = '[PAD]', '[CLS]'  # padding符号, bert中综合信息符号

def predict_from_file(file_path, model_name, dataset):
    contents = []
    src_contents = []
    x = import_module('models.' + model_name)
    config = x.Config(dataset)

    model = x.Model(config).to(config.device)
    model.load_state_dict(torch.load(config.save_path))
    model.eval()
    print("load model over", config.save_path)

    pad_size = config.pad_size
    with open(file_path, 'r', encoding='UTF-8') as f:
        for line in tqdm(f):
            content = line.strip()
            if not content:
                continue

            token = config.tokenizer.tokenize(content)
            token = [CLS] + token
            seq_len = len(token)
            mask = []
            token_ids = config.tokenizer.convert_tokens_to_ids(token)

            if pad_size:
                if len(token) < pad_size:
                    mask = [1] * len(token_ids) + [0] * (pad_size - len(token))
                    token_ids += ([0] * (pad_size - len(token)))
                else:
                    mask = [1] * pad_size
                    token_ids = token_ids[:pad_size]
                    seq_len = pad_size
            contents.append((token_ids, seq_len, mask))
            src_contents.append(content)

    print("contents sum: ", len(contents))
    batch_size = 1
    start_index = 0
    Softmax = nn.Softmax(dim=1)

    while start_index < len(contents):
        batch_contents = contents[start_index: min(start_index+batch_size, len(contents))]
        start_index = start_index+batch_size
        print("contents batch: ", start_index)
        np.random.seed(1)
        torch.manual_seed(1)
        torch.cuda.manual_seed_all(1)
        torch.backends.cudnn.deterministic = True  # 保证每次结果一样
        with torch.no_grad():
            x = torch.LongTensor([_[0] for _ in batch_contents]).to(config.device)
            seq_len = torch.LongTensor([_[1] for _ in batch_contents]).to(config.device)
            mask = torch.LongTensor([_[2] for _ in batch_contents]).to(config.device)
            outputs = model((x, seq_len, mask))
            probs = Softmax(outputs)
            max_prob, predicted = torch.max(probs, 1)
            # print("out_put:{}".format(outputs))
            print("predict_tag:{} {} {}".format(src_contents[start_index-1], predicted.cpu().numpy(), max_prob))


if __name__ == '__main__':
    predict_from_file("/home/recommend_nearby/work_space/niu.tong/data/test_habit_719_16_17_sport.txt", "ERNIE", "HabitMsgs")