from abc import abstractmethod, ABCMeta
from typing import Union, Optional
import re
from tqdm import tqdm
from llama_cpp import Llama
from datasets import Dataset  # it is import to load llama_cpp first before datasets to prevent error like https://github.com/abetlen/llama-cpp-python/issues/806
from huggingface_hub import hf_hub_download
from anyclassifier.annotation.prompt import AnnotationPrompt


class AnnotatorBase(metaclass=ABCMeta):

    regex_pattern = re.compile(r'Label:\s*(.+)')
    @abstractmethod
    def annotate(self, text: str) -> str:
        pass

    @classmethod
    def parse_output(cls, text: str) -> Optional[str]:
        match = cls.regex_pattern.search(text)
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
                         n_record: int = 1000) -> Dataset:
        selected_dataset = dataset.select(range(n_record))

        label_list = []
        for d in tqdm(selected_dataset):
            llm_output = self.annotate(d[text_col])
            label = self.parse_output(llm_output)
            label_list.append(label)

        selected_dataset = selected_dataset.add_column("label", label_list)
        selected_dataset = selected_dataset.filter(lambda x: x.get("label") is not None)
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
