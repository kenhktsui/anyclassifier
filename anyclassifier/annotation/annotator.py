import sys
from abc import abstractmethod, ABCMeta
from typing import Union, Optional, List
import re
from collections import Counter
from tqdm import tqdm
import logging
from llama_cpp import Llama
from datasets import Dataset  # it is import to load llama_cpp first before datasets to prevent error like https://github.com/abetlen/llama-cpp-python/issues/806
from huggingface_hub import hf_hub_download
from anyclassifier.annotation.prompt import AnnotationPrompt, Label


logging.basicConfig(stream=sys.stdout, level=logging.INFO)


class AnnotatorBase(metaclass=ABCMeta):
    def __init__(self):
        self.regex_pattern = None

    def prepare_regex_pattern(self, labels: List[Label]):
        labels_str = "|".join(l.name for l in labels)
        self.regex_pattern = re.compile(rf'Label:\s*({labels_str})')

    @abstractmethod
    def annotate(self, text: str) -> str:
        pass

    def parse_output(self, text: str) -> Optional[str]:
        match = self.regex_pattern.search(text)
        if match:
            return match.group(1)
        return None


class LlamaCppAnnotator(AnnotatorBase):
    def __init__(self,
                 prompt: AnnotationPrompt,
                 model_path: str = hf_hub_download("lmstudio-community/Meta-Llama-3.1-8B-Instruct-GGUF",
                                                   "Meta-Llama-3.1-8B-Instruct-Q8_0.gguf"),
                 n_gpu_layers: int = -1,
                 n_ctx: int = 2048):
        super().__init__()
        self.prepare_regex_pattern(prompt.label_definition)
        self._prompt = prompt
        self._llm = Llama(model_path=model_path,
                          n_gpu_layers=n_gpu_layers,
                          seed=42,
                          n_ctx=n_ctx,
                          verbose=False
                          )

    def annotate(self, text: str) -> str:
        output = self._llm.create_chat_completion(
            messages=[
                {
                    "role": "user",
                    "content": self._prompt.get_prompt(text)
                }
            ],
            max_tokens=256
        )
        return output["choices"][0]["message"]["content"]

    def annotate_dataset(self,
        dataset: Union[Dataset],
        text_col: str = "text",
        n_record: int = 200,
        max_length_for_labeling: int = 1500,
        shuffle: bool = True
    ) -> Dataset:
        # shuffle the data to randomise potential bias in data collection process
        if shuffle:
            dataset = dataset.shuffle(10000)

        selected_dataset = dataset.select(range(n_record))

        label_list = []
        for d in tqdm(selected_dataset, desc="Annotating dataset"):
            llm_output = self.annotate(d[text_col][:max_length_for_labeling])
            label = self.parse_output(llm_output)
            label_list.append(label)

        selected_dataset = selected_dataset.add_column("label", label_list)
        selected_dataset = selected_dataset.filter(lambda x: x.get("label") is not None)
        logging.info(f"""Count of labels
{Counter(selected_dataset["label"]).most_common(len(self._prompt.label_definition))}        
        """)
        return selected_dataset


if __name__ == "__main__":
    from datasets import load_dataset
    from anyclassifier.annotation.prompt import Label
    prompt = AnnotationPrompt(
        task_description="Classify a text's sentiment.",
        label_definition=[
            Label(name="1", desc='positive sentiment'),
            Label(name="0", desc='negative sentiment')
        ]
    )
    annotator = LlamaCppAnnotator(prompt)
    print(annotator.annotate("It is one of best I ever watched."))

    dataset = load_dataset("stanfordnlp/imdb", split="train")
    # mock unlabeled data
    dataset = dataset.remove_columns("label")
    annotated_dataset = annotator.annotate_dataset(dataset, n_record=5)
    print(annotated_dataset)
    print(annotated_dataset[0])
