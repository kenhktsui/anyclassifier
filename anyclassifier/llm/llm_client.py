
from abc import abstractmethod
import json
import logging
import os
from typing import Union
import uuid

from llama_cpp import Llama, LlamaGrammar
from openai import AsyncOpenAI, DefaultAsyncHttpxClient
from pydantic import BaseModel, RootModel
from huggingface_hub import hf_hub_download


# Suppress info log from httpx
logging.getLogger("httpx").setLevel(logging.WARNING)

logger = logging.getLogger(__name__)
console_handler = logging.StreamHandler()
console_handler.setFormatter(logging.Formatter('[%(asctime)s] [%(name)s] %(message)s'))
logger.addHandler(console_handler)
logger.setLevel(logging.INFO)


class LLMClient:
    SYSTEM_PROMPT = """You are a helpful, smart, kind, and efficient AI assistant. You always fulfill the user's
                       requests to the best of your ability."""

    def __init__(self,
                 temperature: float = 0.3,
                 max_tokens: int = 4096,
                 ):

        self._temperature = temperature
        self._max_tokens = max_tokens

    @abstractmethod
    async def _call_llm(self, prompt: str, schema: Union[BaseModel, RootModel] = None):
        pass

    async def prompt_llm(self, prompt: str, schema: Union[BaseModel, RootModel] = None):
        request_id = str(uuid.uuid4())
        logger.debug(f'<{request_id}> LLM request, prompt: {prompt}')

        response_content = await self._call_llm(prompt, schema)
        logger.debug(f'<{request_id}> LLM responded, response: {response_content}')
        return response_content


class LlamaCppClient(LLMClient):
    def __init__(self,
                 model_path: str = hf_hub_download("lmstudio-community/Meta-Llama-3.1-8B-Instruct-GGUF",
                                                   "Meta-Llama-3.1-8B-Instruct-Q8_0.gguf"),
                 *args,
                 n_gpu_layers: int = -1,
                 n_ctx: int = 4096,
                 **kwargs):
        super().__init__(*args, **kwargs)

        self._llama_cpp = Llama(model_path=model_path,
                                n_gpu_layers=n_gpu_layers,
                                seed=42,
                                n_ctx=n_ctx,
                                verbose=False,
                                )
        # randomise generation for each run
        self._llama_cpp.set_seed(-1)

    async def _call_llm(self, prompt: str, schema: Union[BaseModel, RootModel] = None):
        output = self._llama_cpp.create_chat_completion(
            messages=[
                {
                    "role": "system",
                    "content": self.SYSTEM_PROMPT
                },
                {
                    "role": "user",
                    "content": prompt
                }
            ],
            max_tokens=self._max_tokens,
            temperature=self._temperature,
            grammar=LlamaGrammar.from_json_schema(json.dumps(schema.model_json_schema(),
                                                             indent=2)) if schema else None
        )
        response_content = output["choices"][0]["message"]["content"]
        if not schema:
            return response_content
        try:
            result = json.loads(response_content)
            return result
        except json.decoder.JSONDecodeError:
            raise ValueError(f"Invalid response from LLM: {response_content}")


class OpenAIClient(LLMClient):
    def __init__(self,
                 *args,
                 model: str = "gpt-4o-mini",
                 api_key: str = os.environ.get("OPENAI_API_KEY"),
                 proxy_url: str = os.getenv("OPENAI_PROXY_URL"),
                 **kwargs):

        super().__init__(*args, **kwargs)

        if api_key is None:
            raise ValueError("openai_api_key must be provided for openai")

        self._openai_model = model
        if proxy_url:
            self._openai = AsyncOpenAI(api_key=api_key,
                                       http_client=DefaultAsyncHttpxClient(proxy=proxy_url))
        else:
            self._openai = AsyncOpenAI(api_key=api_key)

    async def _call_llm(self, prompt: str, schema: Union[BaseModel, RootModel] = None):
        system_prompt = self.SYSTEM_PROMPT
        if schema:
            system_prompt += f"Output a JSON array in a field named 'data', that matches" \
                f"the following schema:\n{json.dumps(schema.model_json_schema(), indent=2)}"

        output = await self._openai.chat.completions.create(
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": prompt}
            ],
            response_format={"type": "json_object"} if schema else None,
            model=self._openai_model,
            temperature=self._temperature,
            max_tokens=self._max_tokens
        )
        response_content = output.choices[0].message.content
        if not schema:
            return response_content
        try:
            result = json.loads(response_content)['data']
            return result
        except json.decoder.JSONDecodeError:
            raise ValueError(f"Invalid response from LLM: {response_content}")
