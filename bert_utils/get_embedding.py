# coding: UTF-8

import torch
from pytorch_pretrained import BertTokenizer, BertModel, BertForMaskedLM

bert_path = ".bert_pretrain"
PAD, CLS = '[PAD]', '[CLS]'  # padding符号, bert中综合信息符号
pad_size = 32

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

    tokens_tensor = torch.tensor([token_ids])
    segments_tensors = torch.tensor([mask])
    # Load pre-trained model (weights)
    bert_model = BertModel.from_pretrained(bert_path)
    # Put the model in "evaluation" mode, meaning feed-forward operation.
    bert_model.eval()

    with torch.no_grad():
        encoded_layers, pooled = bert_model(tokens_tensor, attention_mask=segments_tensors, output_all_encoded_layers=False)

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


if __name__ == '__main__':
    context = "越野"
    get_bert_embedding(context)

