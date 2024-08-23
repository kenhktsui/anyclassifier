import asyncio
import os
import sys
from abc import abstractmethod, ABCMeta
from typing import Optional, List
import re
from collections import Counter
import uuid
from tqdm import tqdm
import logging
from anyclassifier.llm.llm_client import LLMClient, LlamaCppClient, OpenAIClient
from anyclassifier.annotation.prompt import AnnotationPrompt
from anyclassifier.schema.schema import Label
from datasets import Dataset, load_dataset  # it is import to load llama_cpp first before datasets to prevent error like https://github.com/abetlen/llama-cpp-python/issues/806


logging.basicConfig(stream=sys.stdout, level=logging.INFO, format='[%(asctime)s] [%(name)s] %(message)s')


class AnnotatorBase(metaclass=ABCMeta):
    def __init__(self):
        self.regex_pattern = None

    def prepare_regex_pattern(self, labels: List[Label]):
        labels_str = "|".join(str(l.id) for l in labels)
        self.regex_pattern = re.compile(rf'Label:\s*({labels_str})')

    @abstractmethod
    def annotate(self, text: str) -> str:
        pass

    def parse_output(self, text: str) -> Optional[str]:
        match = self.regex_pattern.search(text)
        if match:
            return int(match.group(1))
        return None


class LLMAnnotator(AnnotatorBase):
    def __init__(self, llm_client: LLMClient, prompt: AnnotationPrompt):
        super().__init__()
        self.prepare_regex_pattern(prompt.label_definition)
        self._prompt = prompt
        self._llm_client = llm_client

    async def annotate(self, text: str) -> str:
        output = await self._llm_client.prompt_llm(self._prompt.get_prompt(text))
        return output

    async def annotate_dataset(self,
                               dataset: Dataset,
                               text_col: str = "text",
                               n_record: int = 200,
                               max_length_for_labeling: int = 1500,
                               shuffle: bool = True,
                               llm_concurrency: int = 10) -> Dataset:
        # shuffle the data to randomise potential bias in data collection process
        if shuffle:
            dataset = dataset.shuffle(10000)

        selected_dataset = dataset.select(range(n_record))

        label_list = []

        semaphore = asyncio.Semaphore(llm_concurrency)

        tasks = []

        # for d in tqdm(selected_dataset, desc="Annotating dataset"):

        progress = tqdm(desc="Annotating dataset", total=n_record)

        async def annotate_record(d):
            async with semaphore:
                trace_id = str(uuid.uuid4())
                logging.debug(f"<{trace_id}> Annotating {d[text_col][:max_length_for_labeling]}")
                llm_output = await self.annotate(d[text_col][:max_length_for_labeling])
                label = self.parse_output(llm_output)
                logging.debug(f"<{trace_id}> Label: {label}")
                label_list.append(label)
                progress.update(1)

        for d in selected_dataset:
            tasks.append(asyncio.create_task(annotate_record(d)))

        await asyncio.gather(*tasks)

        selected_dataset = selected_dataset.add_column("label", label_list)
        selected_dataset = selected_dataset.filter(lambda x: x.get("label") is not None)
        logging.info(f"""Count of labels
{Counter(selected_dataset["label"]).most_common(len(self._prompt.label_definition))}
        """)
        return selected_dataset


if __name__ == "__main__":
    llm_client = OpenAIClient(
        api_key=os.environ.get("OPENAI_API_KEY"),
        proxy_url=os.getenv("OPENAI_PROXY_URL"),
        model="gpt-4o-mini",
        temperature=0.3,
        max_tokens=256,
    )
    # llm_client = LlamaCppClient(temperature=0.3,
    #                             max_tokens=256)

    prompt = AnnotationPrompt(
        task_description="Classify a text's sentiment.",
        label_definition=[
            Label(id=1, desc='positive sentiment'),
            Label(id=0, desc='negative sentiment')
        ]
    )
    annotator = LLMAnnotator(llm_client, prompt)
    output = asyncio.run(annotator.annotate("It is one of best I ever watched."))
    print(output)

    dataset = load_dataset("stanfordnlp/imdb", split="train")
    # mock unlabeled data
    dataset = dataset.remove_columns("label")
    annotated_dataset = asyncio.run(annotator.annotate_dataset(dataset, n_record=10, llm_concurrency=3))
    print(annotated_dataset)
    print(annotated_dataset[0])
