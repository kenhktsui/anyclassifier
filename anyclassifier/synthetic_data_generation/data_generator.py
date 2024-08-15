from typing import List, Union, Optional
from abc import abstractmethod, ABCMeta
import json
from tqdm import tqdm
from pydantic import BaseModel, RootModel
from huggingface_hub import hf_hub_download
from datasets import Dataset
from llama_cpp import Llama, LlamaGrammar
from anyclassifier.schema.schema import ItemList, SourceTypeList, SyntheticData, Label


class SyntheticDataGeneratorBase(metaclass=ABCMeta):
    SYSTEM_PROMPT = """You are a helpful, smart, kind, and efficient AI assistant. You always fulfill the user's requests to the best of your ability."""

    def __init__(self,
                 model_path: str = hf_hub_download("lmstudio-community/Meta-Llama-3.1-8B-Instruct-GGUF",
                                                   "Meta-Llama-3.1-8B-Instruct-Q8_0.gguf"),
                 n_gpu_layers: int = -1,
                 n_ctx: int = 4096
                 ):
        self._llm = Llama(model_path=model_path,
                          n_gpu_layers=n_gpu_layers,
                          seed=42,
                          n_ctx=n_ctx,
                          verbose=False,
                          )
        # randomise generation for each run
        self._llm.set_seed(-1)

    def prompt_llm(self, prompt: str, schema: Union[BaseModel, RootModel]) -> List[dict]:
        grammar = LlamaGrammar.from_json_schema(json.dumps(schema.model_json_schema()))
        output = self._llm.create_chat_completion(
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
            max_tokens=4096,
            temperature=0.3,
            grammar=grammar
        )
        try:
            return json.loads(output["choices"][0]["message"]["content"])
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

    def generate(self,
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
        source_type_list = self.prompt_llm(source_prompt, SourceTypeList)[:n_source_type]

        data_record_list = []
        for label in labels:
            for s in tqdm(source_type_list, desc="source_type"):
                topics = self.prompt_llm(self.EXPAND_LEVEL1_PROMPT.format(instruction=instruction,
                                                                          labels=labels_str,
                                                                          label=label.desc,
                                                                          source_type=s["source_type"],
                                                                          n=n_topic),
                                         ItemList)[:n_topic]
                for t in tqdm(topics, desc="topic"):
                    subtopics = self.prompt_llm(self.EXPAND_LEVEL2_PROMPT.format(instruction=instruction,
                                                                                 labels=labels_str,
                                                                                 label=label.desc,
                                                                                 source_type=s["source_type"],
                                                                                 n=n_subtopic,
                                                                                 topic=t["item"]),
                                                ItemList)[:n_subtopic]
                    for st in subtopics:
                        data_record = self.prompt_llm(self.DATA_GENERATION_PROMPT.format(instruction=instruction,
                                                                                         labels=labels_str,
                                                                                         label=label.desc,
                                                                                         topic=st["item"],
                                                                                         source_type=s["source_type"],
                                                                                         n=sample_per_subtopic),
                                                      SyntheticData)[:sample_per_subtopic]
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
    tree_constructor = SyntheticDataGeneratorForSequenceClassification()
    tree_constructor.generate(
        "classify sentiment of movie review",
        [
            Label(id=0, desc='negative sentiment'),
            Label(id=1, desc='positive sentiment')
        ]
    )
