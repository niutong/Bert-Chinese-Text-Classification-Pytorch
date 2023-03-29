# coding: UTF-8

import torch
import os, sys
# 把当前文件所在文件夹的父文件夹路径加入到PYTHONPATH
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from pytorch_pretrained import BertTokenizer, BertModel

os.environ["CUDA_VISIBLE_DEVICES"] = '4,5,6,7'
bert_path = os.path.dirname(os.path.dirname(os.path.abspath(__file__))) + "/bert_pretrain"
PAD, CLS = '[PAD]', '[CLS]'  # padding符号, bert中综合信息符号
pad_size = 32
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

def get_bert_embedding(context):
    tokenizer = BertTokenizer.from_pretrained(bert_path)
    token = tokenizer.tokenize(context)
    token = [CLS] + token
    seq_len = len(token)
    mask = []
    token_ids = tokenizer.convert_tokens_to_ids(token)

    print(token)
    print(token_ids)
    print(len(list(tokenizer.vocab.keys())), list(tokenizer.vocab.keys())[0:10])

    if pad_size:
        if len(token) < pad_size:
            mask = [1] * len(token_ids) + [0] * (pad_size - len(token))
            token_ids += ([0] * (pad_size - len(token)))
        else:
            mask = [1] * pad_size
            token_ids = token_ids[:pad_size]
            seq_len = pad_size

    tokens_tensor = torch.LongTensor([token_ids]).to(device)
    segments_tensors = torch.LongTensor([mask]).to(device)
    # Load pre-trained model (weights)
    bert_model = BertModel.from_pretrained(bert_path).to(device)
    # Put the model in "evaluation" mode, meaning feed-forward operation.
    bert_model.eval()

    print(tokens_tensor)
    print(segments_tensors)

    with torch.no_grad():
        encoded_layers, pooled = bert_model(tokens_tensor, attention_mask=segments_tensors, output_all_encoded_layers=True)

    print("Number of layers:", len(encoded_layers))
    layer_i = 0
    print("Number of batches:", len(encoded_layers[layer_i]))
    batch_i = 0
    print("Number of tokens:", len(encoded_layers[layer_i][batch_i]))
    token_i = 0
    print("Number of hidden units:", len(encoded_layers[layer_i][batch_i][token_i]))

    token_embeddings = []
    # For each token in the sentence...
    for token_i in range(len(token)):
        # Holds 12 layers of hidden states for each token
        hidden_layers = []
        # For each of the 12 layers...
        for layer_i in range(len(encoded_layers)):
            # Lookup the vector for `token_i` in `layer_i`
            vec = encoded_layers[layer_i][batch_i][token_i]
            hidden_layers.append(vec)
        token_embeddings.append(hidden_layers)
    # Sanity check the dimensions:
    print("Number of tokens in sequence:", len(token_embeddings))
    print("Number of layers per token:", len(token_embeddings[0]))

    concatenated_last_4_layers = [torch.cat((layer[-1], layer[-2], layer[-3], layer[-4]), 0) for layer in
                                  token_embeddings]  # [number_of_tokens, 3072]
    summed_last_4_layers = [torch.sum(torch.stack(layer)[-4:], 0) for layer in
                            token_embeddings]  # [number_of_tokens, 768]

    sentence_embedding = torch.mean(encoded_layers[11], 1)
    print("sentence embedding size", sentence_embedding.shape)
    print("Our final sentence embedding vector of shape:"), sentence_embedding[0].shape[0]
    print(sentence_embedding)
    return sentence_embedding[0]

if __name__ == '__main__':

    target_list = ["二月二龙抬头", "龙", "吃猪头肉", "理发", "剪头发", "二月初二", "理了个发", "二月二，龙抬头开启一年好兆头",
                   "猪头", "周日", "星期天", "理发店", "龙台头", "雨天", "下雨天", "降温", "冷死", "节日快乐", "好日子", "过节",
                   "开工大吉", "开门红", "初八", "大年初八", "开门大吉", "正月初八", "复工", "搞钱", "发发发"]

    # contextA = "越野"
    # contextB = "体育"
    #
    # x = get_bert_embedding(contextA)
    # y = get_bert_embedding(contextB)

    for i in range(len(target_list)-1):
        j = i+1
        while j < len(target_list):
            x = get_bert_embedding(target_list[i])
            y = get_bert_embedding(target_list[j])
            j += 1

            similarity = torch.cosine_similarity(x, y, dim=0)
            print('similarity', target_list[i], target_list[j], round(similarity.cpu().numpy().tolist(), 3))
            break
        break


