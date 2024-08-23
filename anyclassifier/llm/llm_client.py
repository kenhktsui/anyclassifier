
import json
import logging
import os
from typing import List, Union
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
                 use_provider: str = "llama_cpp",
                 *kwargs,
                 temperature: float = 0.3,
                 max_tokens: int = 4096,
                 openai_model: str = "gpt-4o-mini",
                 openai_api_key: str = os.environ.get("OPENAI_API_KEY"),
                 openai_proxy_url: str = os.getenv("OPENAI_PROXY_URL"),
                 llama_cpp_model_path: str = hf_hub_download("lmstudio-community/Meta-Llama-3.1-8B-Instruct-GGUF",
                                                             "Meta-Llama-3.1-8B-Instruct-Q8_0.gguf"),
                 llama_cpp_n_gpu_layers: int = -1,
                 llama_cpp_n_ctx: int = 4096,
                 ):

        self._use_provider = use_provider
        self._temperature = temperature
        self._max_tokens = max_tokens

        if self._use_provider == "llama_cpp":
            self._llama_cpp = Llama(model_path=llama_cpp_model_path,
                                    n_gpu_layers=llama_cpp_n_gpu_layers,
                                    seed=42,
                                    n_ctx=llama_cpp_n_ctx,
                                    verbose=False,
                                    )
            # randomise generation for each run
            self._llama_cpp.set_seed(-1)
        elif self._use_provider == "openai":
            if openai_api_key is None or openai_proxy_url is None:
                raise ValueError("openai_api_key and openai_proxy_url must be provided for openai")
            self._openai_model = openai_model
            self._openai = AsyncOpenAI(api_key=openai_api_key,
                                       http_client=DefaultAsyncHttpxClient(proxy=openai_proxy_url),
                                       )

    async def prompt_llm(self, prompt: str, schema: Union[BaseModel, RootModel]) -> List[dict]:
        if self._use_provider == "llama_cpp":
            grammar = LlamaGrammar.from_json_schema(json.dumps(schema.model_json_schema(), indent=2))
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
                grammar=grammar
            )
            response_content = output["choices"][0]["message"]["content"]
        elif self._use_provider == "openai":
            request_id = str(uuid.uuid4())
            logger.debug(f'[LLM Prompt] <{request_id}> OpenAI request, prompt: {prompt}')
            system_prompt = f"{self.SYSTEM_PROMPT}\n\nOutput a JSON array in a field named 'data', that matches" \
                            f"the following schema:\n{json.dumps(schema.model_json_schema(), indent=2)}"

            output = await self._openai.chat.completions.create(
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": prompt}
                ],
                response_format={"type": "json_object"},
                model=self._openai_model,
                temperature=self._temperature,
                max_tokens=self._max_tokens
            )
            response_content = output.choices[0].message.content
            logger.debug(f'[LLM Prompt] <{request_id}> OpenAI responded, response: {response_content}')
        else:
            raise ValueError(f"Invalid model: {self._use_provider}")

        try:
            result = json.loads(response_content)['data']
            return result
        except json.decoder.JSONDecodeError:
            return []
