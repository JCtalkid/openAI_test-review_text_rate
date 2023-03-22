"""
TODO:Скрипт должен реализовывать режими пототковой, предзагрузочной или комбинированной обработки отзывов.
В случае потоковой обработки, что полезно, если количество отзывов большое, дублируется запрос на вводе данных.
В случае предзагрузки обрабатываемых данных, может возникнуть лимит на оперативную память и на предел токенов.
Комбинированный метод подразумевает предзагрузить максимальное допустимое количество данных, и отправлять по цепочке.
Комбинированный метод напоминает общественный автобус, который старается увести так много, сколько поместиться.
Использовать _langchain.LLMChain. Использование llama_index не оправданно
TODO:Добавить логи

* Потоковая загрузка файла
* Построчное чтение через операцию yield
* Чтение элемента колонки
* Запись элементов в buffer_data_list до заполнения токенами на buffer_tokens_limit(=70% = max_tokens * 0.7)
* Если предел превышен, сохраняет последнего в следующий буфер на отправку, и продолжает выполнение

* Запрос по API: prompt + [buffer_data_list]
* Получение ответа по API
* Парсинг списка рейтинга через регулярные выражения или через сплит (предпочтительно)
*
* Потоковая запись в файл по индексу последнего элемента в сортированных группах, распределённой по значению рейтинга

"""
# from llama_index import SimpleDirectoryReader, GPTListIndex, readers, GPTSimpleVectorIndex, LLMPredictor, PromptHelper
# from langchain import OpenAI


# from dataclasses import dataclass
import csv
import logging
from re import findall

from transformers import AutoTokenizer
import torch
import openai

from config import *


openai.api_key = os.getenv('OPENAI_API_KEY')
logging.basicConfig(level=logging.DEBUG,
                    format="%(asctime)s %(levelname)s %(message)s",
                    datefmt="%Y-%m-%d %H:%M:%S",
                    filename='log.log'
                    )

BUFFER_TOKENS_LIMIT = MAX_TOKENS * 0.8

# def construct_index(dir_path):
#     max_input_size = 4096
#     num_outputs = 300
#     max_chunk_overlap = 20
#     chunk_size_limit = 600
#     llm_predictor = LLMPredictor(llm=OpenAI(temperature=0, max_tokens=num_outputs))
#     prompt_helper = PromptHelper(max_input_size, num_outputs, max_chunk_overlap, chunk_size_limit=chunk_size_limit)
#
#     documents = SimpleDirectoryReader(dir_path).load_data()
#
#     index = GPTSimpleVectorIndex(
#         documents, llm_predictor=llm_predictor, prompt_helper=prompt_helper
#     )
#
#     index.save_to_disk('index.json')
#
#
# def ask_ai():
#     index = GPTSimpleVectorIndex.load_from_disk('index.json')
#
#     query = input()
#     response = index.query(query, response_mode='compact')
#     print(f"Response: <b>{response.response}</b>")
#
#
# @dataclass
# class Buffer:
#     buffer = []
#
#     buffer_limit = max_tokens * 0.7
#     buffer_token_count = 0
#
#     def loader(self, text: str):
#         if
#         self.buffer.append(text)

def fun_token_counter(txt: str) -> int:
    tokenizer = AutoTokenizer.from_pretrained("gpt2")
    input_ids = torch.tensor(tokenizer.encode(txt)).unsqueeze(0)
    num_tokens = input_ids.shape[1]
    return num_tokens


'''
prompt before GPT optimization = "Act as a system. Range this reviews by rave with rate on a scale from 1 to 10,
where 10 is the most enthusiastic review, 1 is the most negative review.
Return me only a list of rate numbers separate with ','."

prompt after GPT optimization = "Rank reviews on a scale of 1 to 10. Give only the numeric ratings separated by commas."
'''
# Возвращает список из чисел, оценивающих текст по порядкку ввода
def fun_GPT_respons_list(review_list: list[str]) -> list[int]:
    prompt = f"""Act as a system. Range this reviews by rave with rate on a scale from 1 to 10, where 10 is the most enthusiastic review, 1 is the most negative review. Return me only a list of rate numbers separate with ','.\nReviews: {review_list}"""

    # try:
    response = openai.Completion.create(engine=GPT_ENGINE, prompt=prompt, temperature=0)
    # except ConnectionError:
    #     logging.error(f'{ConnectionError}')
    logging.debug(f'{response}')

    # [int(i) for i in response['choices'][0]['text'].split(',')]

    message = response['choices'][0]['text']
    match_list = findall('\d{1,2}', message)

    return match_list

def fun_GPT_respons(review: str) -> int:
    prompt = f"Rank review on a scale of 1 to 10. Give only the numeric rating. Review: {review}"
    response = openai.Completion.create(engine=GPT_ENGINE, prompt=prompt, temperature=0)
    # except ConnectionError:
    #     logging.error(f'{ConnectionError}')
    logging.debug(f'{response}')

    # [int(i) for i in response['choices'][0]['text'].split(',')]

    message = response['choices'][0]['text']
    match_list = findall('\d{1,2}', message)

    return int(match_list[0])

# TODO:
def fun_read_CSV(filename):
    pass

# TODO:
def fun_write_CSV(filename):
    pass

# TODO:
def buffer_zone(text: str, start=False):
    buffer = []
    buffer_token_count = 0

    if buffer_token_count + row_token_count > BUFFER_TOKENS_LIMIT or start:

        for i, v in enumerate(index_list):
            pass
        buffer = [row['review text']]
        buffer_token_count = row_token_count

        rate_list = fun_GPT_respons(buffer)

    return rate_list


if __name__ == '__main__':
    buffer = []
    buffer_token_count = 0

    index_list = []
    rows_rate = [[]] * 10

    with open(FILE_NAME, mode='r') as csvfile:
        reader = csv.DictReader(csvfile)
        csv_heads = [i for i in reader.__next__().keys() if i]
        for index, row in enumerate(reader, 1):
            row_token_count = fun_token_counter(row['review text'])
            index_list.append(index)
            if buffer_token_count + row_token_count > BUFFER_TOKENS_LIMIT:

                for i, v in enumerate(index_list):
                    pass
                buffer = [row['review text']]
                buffer_token_count = row_token_count

                rate_list = fun_GPT_respons(buffer)






