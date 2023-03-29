import os

from transformers import pipeline

os.environ["CUDA_VISIBLE_DEVICES"] = '4,5,6,7'
os.environ["TOKENIZERS_PARALLELISM"] = 'false'



def text_generation():

    generator = pipeline("text-generation", model="uer/gpt2-chinese-poem")
    results = generator(
        "[CLS] 万 叠 春 山 积 雨 晴 ，",
        max_length=40,
        num_return_sequences=2,
    )
    print(results)


if __name__ == "__main__":
    text_generation()