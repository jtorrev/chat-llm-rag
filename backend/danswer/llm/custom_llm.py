import json
from collections.abc import Iterator

import requests
from langchain.schema.language_model import LanguageModelInput
from requests import Timeout

from danswer.configs.model_configs import GEN_AI_API_ENDPOINT
from danswer.configs.model_configs import GEN_AI_MAX_OUTPUT_TOKENS
from danswer.llm.interfaces import LLM
from danswer.llm.utils import convert_lm_input_to_basic_string
from danswer.utils.logger import setup_logger


from langchain.prompts.base import StringPromptValue
from langchain.prompts.chat import ChatPromptValue
from langchain.schema import PromptValue
from langchain.schema.messages import AIMessage
from langchain.schema.messages import BaseMessage
from langchain.schema.messages import BaseMessageChunk
from langchain.schema.messages import HumanMessage
from langchain.schema.messages import SystemMessage

from openai import OpenAI

logger = setup_logger()

import replicate

class CustomModelServer(LLM):
    """This class is to provide an example for how to use Danswer
    with any LLM, even servers with custom API definitions.
    To use with your own model server, simply implement the functions
    below to fit your model server expectation

    The implementation below works against the custom FastAPI server from the blog:
    https://medium.com/@yuhongsun96/how-to-augment-llms-with-private-data-29349bd8ae9f
    """

    @property
    def requires_api_key(self) -> bool:
        return False

    def __init__(
        self,
        # Not used here but you probably want a model server that isn't completely open
        api_key: str | None,
        timeout: int,
        endpoint: str | None = GEN_AI_API_ENDPOINT,
        max_output_tokens: int = GEN_AI_MAX_OUTPUT_TOKENS,
    ):
        if not endpoint:
            raise ValueError(
                "Cannot point Danswer to a custom LLM server without providing the "
                "endpoint for the model server."
            )

        self._endpoint = endpoint
        self._max_output_tokens = max_output_tokens
        self._timeout = timeout
       

    def llama(self,input):
        headers = {
            "Content-Type": "application/json",
        }

        # data = {
        #     "inputs": convert_lm_input_to_basic_string(input),
        #     "parameters": {
        #         "temperature": 0.2,
        #         "max_tokens": self._max_output_tokens,
        #     },
        # }
        data = {
                "messages": [{"role": "user", "content": input}],
                "temperature": 0.2,
                "max_tokens": 1024,
        }
        try:
            response = requests.post(
                self._endpoint, headers=headers, json=data#, timeout=self._timeout
            )
            print(f"************************* {response}")
        except Timeout as error:
            raise Timeout(f"Model inference to {self._endpoint} timed out") from error

        response.raise_for_status()
        return response.json()["text"][0].strip()

    def openai(self,input):
        self._client=OpenAI()
        print(f"************************* {input}")
        completion=self._client.chat.completions.create(
            model="gpt-4-0314",
            messages=[{"role": "user", "content": input}]
        )
        return completion.choices[0].message.content

    
    def mixtral(self,input):
        output = replicate.run(
                    "mistralai/mixtral-8x7b-instruct-v0.1:7b3212fbaf88310cfef07a061ce94224e82efc8403c26fc67e8f6c065de51f21",
                    input={
                        # "top_k": 50,
                        # "top_p": 0.9,
                        "prompt": str(input),
                        # "temperature": st.session_state.temperature,
                        # "max_new_tokens": st.session_state.max_tokens,
                        "prompt_template": "<s>[INST] {prompt} [/INST] ",
                        # "presence_penalty": 0,
                        # "frequency_penalty": 0,
                    },
        )
        return "".join(output).strip().removesuffix("<\s>")

    def _execute(self, input: LanguageModelInput) -> str:
        new_input = ""
        if isinstance(input, list) and isinstance(input[0], HumanMessage):
            new_input =  input[0].content
        elif isinstance(input, list) and isinstance(input[0], str):
            new_input = input[0]
        elif isinstance(input, str):
            new_input = input
        else:
            new_input = str(input)
        # return self.llama(new_input)
        return self.mixtral(new_input)
    
    def log_model_configs(self) -> None:
        logger.debug(f"Custom model at: {self._endpoint}")

    def invoke(self, prompt: LanguageModelInput) -> str:
        return self._execute(prompt)

    def stream(self, prompt: LanguageModelInput) -> Iterator[str]:
        yield self._execute(prompt)
