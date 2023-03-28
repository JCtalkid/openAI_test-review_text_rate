"""
Скрипт должен реализовывать режими пототковой, предзагрузочной или комбинированной обработки отзывов.
В случае потоковой обработки, что полезно, если количество отзывов большое, дублируется запрос на вводе данных.
В случае предзагрузки обрабатываемых данных, может возникнуть лимит на оперативную память и на предел токенов.
Комбинированный метод подразумевает предзагрузить максимальное допустимое количество данных, и отправлять по цепочке.
Комбинированный метод напоминает общественный автобус, который старается увести так много, сколько поместиться.
Использовать _langchain.LLMChain. Использование llama_index не оправданно

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

import csv

from openAI_API_service import *


BUFFER_TOKENS_LIMIT = MAX_TOKENS * 0.8
def buffer_zone(text: str, start=False):
    buffer = []
    buffer_token_count = 0

    if buffer_token_count + row_token_count > BUFFER_TOKENS_LIMIT or start:

        for i, v in enumerate(index_list):
            pass
        buffer = [row['review text']]
        buffer_token_count = row_token_count

        rate_list = fun_GPT_respons_list(buffer)

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

                rate_list = fun_GPT_respons_list(buffer)
