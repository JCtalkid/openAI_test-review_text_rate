"""
prompt before GPT optimization = "Act as a system. Range this reviews by rave with rate on a scale from 1 to 10,
where 10 is the most enthusiastic review, 1 is the most negative review.
Return me only a list of rate numbers separate with ','."

prompt after GPT optimization = "Rank reviews on a scale of 1 to 10. Give only the numeric ratings separated by commas."
"""

from dataclasses import dataclass
from re import findall
import logging

from transformers import AutoTokenizer
import torch
import openai

from openAI_settings import *


openai.api_key = os.getenv('OPENAI_API_KEY')
logging.basicConfig(level=logging.DEBUG,
                    format="%(asctime)s %(levelname)s %(message)s",
                    datefmt="%Y-%m-%d %H:%M:%S",
                    filename='API_tokens.log'
                    )


@dataclass
class ResponseWrapperAPIopenAI:
    @dataclass
    class Usage:
        completion_tokens: int
        prompt_tokens: int
        total_tokens: int

    @dataclass
    class Choices:
        finish_reason: str
        index: int
        logprobs: None
        text: str

    def __post_init__(self):
        self.usage = self.Usage(**self.usage)
        self.choices = [self.Choices(**choice) for choice in self.choices]

    choices: list[Choices]
    usage: Usage
    created: int
    id: str
    model: str
    object: str


def fun_token_counter(txt: str) -> int:
    tokenizer = AutoTokenizer.from_pretrained("gpt2")
    input_ids = torch.tensor(tokenizer.encode(txt)).unsqueeze(0)
    num_tokens = input_ids.shape[1]
    return num_tokens


def fun_GPT_response(review: str) -> int:
    prompt = f"Rank review on a scale of 1 to 10. Give only the numeric rating. Review: {review}"
    response = openai.Completion.create(engine=GPT_ENGINE, prompt=prompt, temperature=0)

    logging.debug(f'{response}')

    message = response['choices'][0]['text']
    match_list = findall(r'\d{1,2}', message)

    return int(match_list[0])


def fun_GPT_respons_list(review_list: list[str]) -> list[int]:
    prompt = "Rank reviews on a scale of 1 to 10. Give only the numeric ratings separated by commas. " \
             f"Reviews: {review_list}"

    # try:
    response = openai.Completion.create(engine=GPT_ENGINE, prompt=prompt, temperature=0)
    # except ConnectionError:
    #     logging.error(f'{ConnectionError}')
    logging.debug(f'{response}')

    # [int(i) for i in response['choices'][0]['text'].split(',')]

    message = response['choices'][0]['text']
    match_list = findall('\d{1,2}', message)

    return match_list


def connection_test():
    openai.api_key = os.getenv('OPENAI_API_KEY')

    prompt = 'Hello world'

    response = openai.Completion.create(engine='text-ada-001', prompt=prompt, max_tokens=50)

    logging.debug(f'{response}')

    print(response['choices'][0]['text'])
    print(response)
