from dataclasses import dataclass
from enum import Enum
from typing import List, Union, Optional
from abc import abstractmethod, ABCMeta
import os
import json
import asyncio
import uuid
from openai import AsyncOpenAI, DefaultAsyncHttpxClient
from tqdm import tqdm
from pydantic import BaseModel, RootModel
from huggingface_hub import hf_hub_download
from datasets import Dataset
from llama_cpp import Llama, LlamaGrammar
from anyclassifier.schema.schema import ItemList, SourceTypeList, SyntheticData, Label
from dotenv import load_dotenv
import logging

load_dotenv()

# Suppress info log from httpx
logging.getLogger("httpx").setLevel(logging.WARNING)

# logger.basicConfig(stream=sys.stdout, level=logger.DEBUG, format='[%(asctime)s] [%(name)s] %(message)s')
logger = logging.getLogger(__name__)
console_handler = logging.StreamHandler()
console_handler.setFormatter(logging.Formatter('[%(asctime)s] [%(name)s] %(message)s'))
logger.addHandler(console_handler)
logger.setLevel(logging.INFO)


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

    async def prompt_templated(self, template: str, schema: Union[BaseModel, RootModel], **kwargs) -> str:
        with open(os.path.join(os.path.dirname(__file__), 'prompts', f'{template}_prompt.txt'), 'r') as file:
            prompt_template = file.read()
        prompt = prompt_template.format(**kwargs)
        return await self.prompt_llm(prompt, schema)
    
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

    @abstractmethod
    def generate(self,
                 instruction: str,
                 labels: List[Label],
                 n_record_to_generate=100,
                 **kwargs) -> Dataset:
        pass



@dataclass
class GenerationItem:
    task: str
    label: str
    source_type: str
    topic: Optional[str]
    subtopic: Optional[str]
    
@dataclass
class GenerationConfig:
    instruction: str
    labels_str: str
    labels: List[Label]
    n_topic: int
    n_subtopic: int
    sample_per_subtopic: int
    

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

    def get_prompt(self, template: str, *args, **kwargs):
        with open(os.path.join(os.path.dirname(__file__), 'prompts', f'{template}_prompt.txt'), 'r') as file:
            prompt_template = file.read()
        return prompt_template.format(*args, **kwargs)

    async def generate(self,
                       instruction: str,
                       labels: List[Label],
                       n_record_to_generate: Optional[int] = 100,
                       n_source_type: Optional[int] = None,
                       n_topic: Optional[int] = None,
                       n_subtopic: Optional[int] = None,
                       llm_concurrency: Optional[int] = 10,
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
            
            llm_concurrency (`int`, *optional*):
                Number of concurrent LLM calls to make

            sample_per_subtopic (`int`, *optional*):

        Returns:
            Dataset
        """

        if (n_record_to_generate is not None and n_source_type is None
                and n_topic is None and n_subtopic is None and sample_per_subtopic is None):
            n_record_to_label_per_label = n_record_to_generate // len(labels) # 50
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


        # Generating initial source type list
        source_type_list = await self.prompt_templated('source_type', SourceTypeList, instruction=instruction, labels=labels_str, n=n_source_type)       
        
        # Start concurrent generation with async queue
        queue = asyncio.Queue()

        data_record_list = []

        # Create workers
        generationConfig = GenerationConfig(instruction=instruction, 
                                            labels_str=labels_str,
                                            labels=labels, 
                                            n_topic=n_topic, 
                                            n_subtopic=n_subtopic, 
                                            sample_per_subtopic=sample_per_subtopic)
        logger.info(f'generationConfig: {generationConfig}')
        
        workers: list[asyncio.Task] = []
        for i in range(llm_concurrency):
            task = asyncio.create_task(self.worker(f'worker-{i}', queue, generationConfig, data_record_list))
            workers.append(task)

        # Put initial items into queue
        for label in labels:
            for source_type in source_type_list:
                await queue.put(GenerationItem(task="source_type", label=label.desc, source_type=source_type['source_type'], topic=None, subtopic=None))

        # Wait for all tasks to be completed
        await queue.join()

        # Cancel worker tasks
        for task in workers:
            task.cancel()

        # Wait for all workers to finish
        await asyncio.gather(*workers, return_exceptions=True)
        

        return Dataset.from_list(data_record_list)


    async def worker(self, name: str, queue: asyncio.Queue, config: GenerationConfig, data_record_list: List[dict]):
        while True:
            try:
                item: GenerationItem = await queue.get()

                logger.info(f'[{name}] - item: {item}')
                if item.task == "source_type":
                    topics = await self.prompt_templated('expand_level1', 
                                                         ItemList,
                                                         instruction=config.instruction,
                                                         labels=config.labels_str,
                                                         label=item.label,
                                                         source_type=item.source_type,
                                                         n=config.n_topic)
                    for t in topics[:config.n_topic]:
                        await queue.put(GenerationItem(task="topic", label=item.label,
                                                source_type=item.source_type, topic=t["item"], subtopic=None))

                elif item.task == "topic":
                    subtopics = await self.prompt_templated('expand_level2',
                                                            ItemList,
                                                            instruction=config.instruction,
                                                            labels=config.labels_str,
                                                            label=item.label,
                                                            source_type=item.source_type,
                                                            n=config.n_subtopic,
                                                            topic=item.topic)
                    for st in subtopics[:config.n_subtopic]:
                        await queue.put(GenerationItem(task="subtopic", label=item.label,
                                                source_type=item.source_type, topic=item.topic, subtopic=st["item"]))

                elif item.task == "subtopic":
                    data_record = await self.prompt_templated('data_generation', 
                                                              SyntheticData,
                                                              instruction=config.instruction,
                                                              labels=config.labels_str,
                                                              label=item.label,
                                                              source_type=item.source_type,
                                                              topic=item.subtopic,
                                                              n=config.sample_per_subtopic)
                    for d in data_record[:config.sample_per_subtopic]:
                        d['label'] = item.label
                        d['meta'] = {
                            "source_type": item.source_type,
                            "topic": item.topic,
                            "subtopic": item.subtopic
                        }
                        data_record_list.extend(data_record)
            except Exception as e:
                logger.error(f'[{name}] Error: {e}')
            finally:
                queue.task_done()
                

if __name__ == "__main__":
    tree_constructor = SyntheticDataGeneratorForSequenceClassification(
        "openai",
        openai_api_key=os.environ.get("OPENAI_API_KEY"),
        openai_proxy_url=os.getenv("OPENAI_PROXY_URL"),
        openai_model="gpt-4o-mini",
        temperature=0.3,
        max_tokens=4096,
    )
    dataset: Dataset = asyncio.run(tree_constructor.generate(
        "classify sentiment of movie review",
        [
            Label(id=0, desc='negative sentiment'),
            Label(id=1, desc='positive sentiment')
        ],
        llm_concurrency=3,
        n_record_to_generate=120
    ))
    dataset.to_csv("dataset.csv")
    print(dataset)
