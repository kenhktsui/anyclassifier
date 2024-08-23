from dataclasses import dataclass
from typing import List, Optional, Union
from abc import abstractmethod, ABCMeta
import os
import asyncio
from datasets import Dataset
from pydantic import BaseModel, RootModel
from anyclassifier.llm.llm_client import LLMClient
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

    def __init__(self, llm_client: LLMClient):
        self._llm_client = llm_client

    async def prompt_templated(self, template: str, schema: Union[BaseModel, RootModel], **kwargs) -> str:
        with open(os.path.join(os.path.dirname(__file__), 'prompts', f'{template}_prompt.txt'), 'r') as file:
            prompt_template = file.read()
        prompt = prompt_template.format(**kwargs)
        return await self._llm_client.prompt_llm(prompt, schema)

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
            n_record_to_label_per_label = n_record_to_generate // len(labels)  # 50
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
        source_type_list = await self.prompt_templated('source_type',
                                                       SourceTypeList,
                                                       instruction=instruction,
                                                       labels=labels_str,
                                                       n=n_source_type)

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
                await queue.put(GenerationItem(task="source_type",
                                               label=label.desc,
                                               source_type=source_type['source_type'],
                                               topic=None,
                                               subtopic=None))

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
                        await queue.put(GenerationItem(task="subtopic",
                                                       label=item.label,
                                                       source_type=item.source_type,
                                                       topic=item.topic,
                                                       subtopic=st["item"]))

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
    llm_client = LLMClient(
        "openai",
        openai_api_key=os.environ.get("OPENAI_API_KEY"),
        openai_proxy_url=os.getenv("OPENAI_PROXY_URL"),
        openai_model="gpt-4o-mini",
        temperature=0.3,
        max_tokens=4096,
    )
    tree_constructor = SyntheticDataGeneratorForSequenceClassification(llm_client)
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
