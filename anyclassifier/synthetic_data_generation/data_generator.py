from typing import List, Union, Optional
from abc import abstractmethod, ABCMeta
import os
import json
import asyncio
from openai import AsyncOpenAI, DefaultAsyncHttpxClient
from tqdm import tqdm
from pydantic import BaseModel, RootModel
from huggingface_hub import hf_hub_download
from datasets import Dataset
from llama_cpp import Llama, LlamaGrammar
from anyclassifier.schema.schema import ItemList, SourceTypeList, SyntheticData, Label
from dotenv import load_dotenv

load_dotenv()


class SyntheticDataGeneratorBase(metaclass=ABCMeta):
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
        self._openai_model = openai_model

        if self._use_provider == "llama_cpp":
            if llama_cpp_model_path is None:
                raise ValueError("llama_cpp_model_path must be provided for llama_cpp")
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
        else:
            raise ValueError(f"Invalid model: {self._use_provider}")

        try:
            result = json.loads(response_content)['data']
            print(result)
            return result
        except json.decoder.JSONDecodeError:
            return []

    @abstractmethod
    def generate(self,
                 instruction: str,
                 labels: List[Label],
                 n_record_to_generate=100,
                 **kwargs) -> Dataset:
        pass


class SyntheticDataGeneratorForSequenceClassification(SyntheticDataGeneratorBase):
    """
    This class adopts a hierarchical generation approach by asking LLM to suggest topic to expand on a particular label
on first level, and suggest subtopic on second level,
It is also inspired by Chan, X., Wang, X., Yu, D., Mi, H., Yu, D., 2024. Scaling synthetic data
creation with 1,000,000,000 personas. URL: https://arxiv.org/abs/
2406.20094, arXiv:2406.20094., where multiple personas are simulated to cover diverse scenario. We adopted a
source-type-driven strategy instead of persona-driven strategy as source type usually reflects a different groups of
target audiences, and it is less fine-grained, which is more suitable for text classification model which is less data
hungry in training.
Each data synthesis is grounded on (instruction, label, subtopic, source_type).
This approach ensures the diversity of synthetic data by design.
    """

    SOURCE_TYPE_PROMPT = f"""I am building a document classifier to {{instruction}} with labels {{labels}}. Suggest {{n}} source type of information for efficient data acquisition.
Output JSON array. Each item contains key "source_type"."""

    EXPAND_LEVEL1_PROMPT = f"""I am building a document classifier to {{instruction}} with labels {{labels}}. I would like to collect collectively exhaustive taxonomy or topic for the label: {{label}} from {{source_type}}.

<instruction>
- Suggest {{n}} taxonomies or topics to further expand on this label.
- Output JSON array. Each item contains key "item" 
</instruction>"""

    EXPAND_LEVEL2_PROMPT = f"""I am building document classifier to {{instruction}} with labels {{labels}}.  I would like to collect collectively exhaustive subtopic under {{topic}} from {{source_type}}.

<instruction>
- Suggest {{n}} subtopic or keywords. 
- Output JSON array. Each item contains key "item" 
</instruction>"""

    DATA_GENERATION_PROMPT = f""""I am building document classifier to {{instruction}} with labels {{labels}}. I would like to collect samples for the label: {{label}}.

<instruction>
- Generate realistic examples for a classification model that will predict label {{label}}.
- Characteristics:
  Topic: {{topic}}.
  Source type: {{source_type}}
- Generate {{n}} example.
- The example shall have a realistic length, and cannot be too short.
- Output JSON array. Each item contains key "text" 
</instruction>"""

    async def generate(self,
                       instruction: str,
                       labels: List[Label],
                       n_record_to_generate: Optional[int] = 100,
                       n_source_type: Optional[int] = None,
                       n_topic: Optional[int] = None,
                       n_subtopic: Optional[int] = None,
                       sample_per_subtopic: Optional[int] = None) -> Dataset:
        """
        Args:
            instruction (`str`):
                Description of use of the classifier (e.g. classify a text's sentiment)
            labels (`List[Label]`):
                The Label.name is used to create instruction
            n_record_to_generate (`int`, *optional*):
                Target number of record to generate. If it is provided, it overrides n_source_type, n_topic, n_subtopic
                and sample_per_subtopic
            n_source_type (`int`, *optional*):

            n_topic (`int`, *optional*):

            n_subtopic (`int`, *optional*):

            sample_per_subtopic (`int`, *optional*):

        Returns:
            Dataset
        """

        if (n_record_to_generate is not None and n_source_type is None
                and n_topic is None and n_subtopic is None and sample_per_subtopic is None):
            n_record_to_label_per_label = n_record_to_generate // len(labels)
            n_topic = 5
            n_source_type = 3
            n_subtopic = 2
            sample_per_subtopic = max(1, int(n_record_to_label_per_label / (n_source_type * n_topic * n_subtopic)))
        elif n_record_to_generate is None and (
            n_topic is None or n_source_type is None or n_subtopic is None and sample_per_subtopic is None
        ):
            raise ValueError("As n_record_to_generate is None, n_topic, n_source_type, n_subtopic and "
                             "sample_per_subtopic must not be None.")

        labels_str = ', '.join([i.desc for i in labels])

        # lowercase the first word and remove full stop at the end.
        instruction = instruction.split(" ")
        instruction[0] = instruction[0].lower()
        instruction = " ".join(instruction)
        if instruction.endswith("."):
            instruction = instruction[:-1]

        source_prompt = self.SOURCE_TYPE_PROMPT.format(instruction=instruction, labels=labels_str, n=n_source_type)
        source_type_list = (await self.prompt_llm(source_prompt, SourceTypeList))[:n_source_type]

        data_record_list = []
        for label in labels:
            for s in tqdm(source_type_list, desc="source_type"):
                topics = (await self.prompt_llm(self.EXPAND_LEVEL1_PROMPT.format(instruction=instruction,
                                                                                 labels=labels_str,
                                                                                 label=label.desc,
                                                                                 source_type=s["source_type"],
                                                                                 n=n_topic),
                                                ItemList))[:n_topic]
                for t in tqdm(topics, desc="topic"):
                    subtopics = (await self.prompt_llm(self.EXPAND_LEVEL2_PROMPT.format(instruction=instruction,
                                                                                        labels=labels_str,
                                                                                        label=label.desc,
                                                                                        source_type=s["source_type"],
                                                                                        n=n_subtopic,
                                                                                        topic=t["item"]),
                                                       ItemList))[:n_subtopic]
                    for st in subtopics:
                        print(len(subtopics))
                        data_record = (await self.prompt_llm(self.DATA_GENERATION_PROMPT.format(instruction=instruction,
                                                                                                labels=labels_str,
                                                                                                label=label.desc,
                                                                                                topic=st["item"],
                                                                                                source_type=s["source_type"],
                                                                                                n=sample_per_subtopic),
                                                             SyntheticData))[:sample_per_subtopic]
                        for d in data_record:
                            d['label'] = label.id
                            d['meta'] = {
                                "source_type": s["source_type"],
                                "topic": t["item"],
                                "subtopic": st["item"]
                            }
                            print(d)
                            data_record_list.extend(data_record)

        return Dataset.from_list(data_record_list)


if __name__ == "__main__":
    tree_constructor = SyntheticDataGeneratorForSequenceClassification(
        "openai",
        openai_api_key=os.environ.get("OPENAI_API_KEY"),
        openai_proxy_url=os.getenv("OPENAI_PROXY_URL"),
        openai_model="gpt-4o-mini",
        temperature=0.3,
        max_tokens=4096,
    )
    dataset = asyncio.run(tree_constructor.generate(
        "classify sentiment of movie review",
        [
            Label(id=0, desc='negative sentiment'),
            Label(id=1, desc='positive sentiment')
        ]
    ))
