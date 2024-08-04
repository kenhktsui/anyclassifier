import sys
from typing import List, Dict, Union, Literal, Optional, Callable, Any
import logging
from datasets import Dataset
from huggingface_hub import interpreter_login
from setfit import SetFitModel, TrainingArguments, Trainer as SetFitTrainer
from anyclassifier.annotation.prompt import Label, AnnotationPrompt, Example
from anyclassifier.annotation.annotator import LlamaCppAnnotator
from anyclassifier.fasttext_wrapper import (
    FastTextConfig, FastTextTrainer, FastTextForSequenceClassification, FastTextTrainingArguments
)


logging.basicConfig(stream=sys.stdout, level=logging.INFO)


def train_anyclassifier(
    instruction: str,
    labels: List[Label],
    annotator_model_path: str,
    unlabeled_dataset: Dataset,
    column_mapping: Dict[str, str] = {"text": "text"},
    model_type: Literal["setfit", "fasttext", "transformers"] = "setfit",
    few_shot_examples: Optional[List[Example]] = None,
    base_model: Optional[str] = "sentence-transformers/paraphrase-mpnet-base-v2",
    num_epochs: Optional[int] = 5,
    batch_size: Optional[int] = 16,
    n_record_to_label: int = 100,
    max_length_for_labeling: int = 1500,
    test_size: float = 0.3,
    metric: Union[str, Callable[["Dataset", "Dataset"], Dict[str, float]]] = "accuracy",
    metric_kwargs: Optional[Dict[str, Any]] = None,
    push_dataset_to_hub: bool = False,
    dataset_repo_id: Optional[str] = None,
    is_dataset_private: Optional[bool] = True,
) -> Union[FastTextTrainer, SetFitTrainer]:
    """
    Train any classifier without labelled data.
    Args:
        instruction (`str`):
            The instruction to LLM annotator
        labels (`List[Label]`):
            The labels including name and desc you want to classify
        annotator_model_path (`str`):
            The path of LLM annotator model to be used by llama.cpp
        unlabeled_dataset ('Dataset'):
            The unlabeled dataset you want to label.
        column_mapping (`Dict[str, str]`, *optional*):
            A mapping from the column names in the dataset to the column names expected by the model.
            The expected format is a dictionary with the following format:
            `{"text_column_name": "text", "label_column_name: "label"}`.
        model_type (Literal["setfit", "fasttext", "transformers"], *optional*):
            Modeling approach -
        few_shot_examples (`List[Example]`, *optional*):
            Few shot examples if you want to provide examples in annotation prompt.
        base_model (`str`, *optional*):
            The base model to be used in setfit/ transformers. Not applicable to fastText model.
        num_epochs (`int`, *optional*):
            No of epochs to train model
        batch_size (`int`, *optional*):
            Batch size to train model
        n_record_to_label (`int`, *optional*):
            No of record for LLM to label
        max_length_for_labeling (`int`, *optional*):
            Max length on character level to avoid exceeding context length of LLM and faster annotation. In general,
            how limiting truncating document affects the accuracy of annotation process depending on various
            factors, like complexity of classification, location of key information. If the same topic is conveyed
            throughout a document (e.g. sentiment analysis, domain classification), the impact is expected to be low.
        test_size (`float`, *optional*):
            Proportion of labeled data to evaluation
        metric (`str` or `Callable`, *optional*, defaults to `"accuracy"`):
            The metric to use for evaluation. If a string is provided, we treat it as the metric
            name and load it with default settings. If a callable is provided, it must take two arguments
            (`y_pred`, `y_test`) and return a dictionary with metric keys to values.
        metric_kwargs (`Dict[str, Any]`, *optional*):
            Keyword arguments passed to the evaluation function if `metric` is an evaluation string like "f1".
            For example useful for providing an averaging strategy for computing f1 in a multi-label setting.
        push_dataset_to_hub (`bool`, *optional*):
            Whether to push dataset to huggingface hub for reuse, highly recommended to do so.
        dataset_repo_id (`str`, *optional*):
            Huggingface dataset repo id if you want to push
        is_dataset_private (`bool`, *optional*):
            Whether you want to make the dataset private
    """
    if push_dataset_to_hub:
        interpreter_login(new_session=False)
        assert dataset_repo_id is not None, "dataset_repo_id must be provided when push_dataset_to_hub is True."

    # labeling dataset
    prompt = AnnotationPrompt(
        task_description=instruction,
        label_definition=labels,
        few_shot_examples=few_shot_examples
    )
    annotator = LlamaCppAnnotator(prompt, annotator_model_path)
    label_dataset = annotator.annotate_dataset(
        unlabeled_dataset,
        n_record=n_record_to_label,
        max_length_for_labeling=max_length_for_labeling
    )

    label_dataset = label_dataset.train_test_split(test_size=test_size)

    if push_dataset_to_hub:
        label_dataset.push_to_hub(dataset_repo_id, private=is_dataset_private)

    # training
    if model_type == "fasttext":
        id2label = {i: l.name for i, l in enumerate(labels)}
        label2id = {label: idx for idx, label in id2label.items()}
        config = FastTextConfig(id2label=id2label, label2id=label2id)
        model = FastTextForSequenceClassification(config)
        args = FastTextTrainingArguments("fasttext_model", epoch=num_epochs)
        trainer = FastTextTrainer(
            model=model,
            args=args,
            train_dataset=label_dataset["train"],
            eval_dataset=label_dataset["test"],
            metric=metric,
            metric_kwargs=metric_kwargs,
            column_mapping={**column_mapping, "label": "label"},
        )

        # Train and evaluate
        trainer.train()
        metrics = trainer.evaluate(label_dataset["test"])
        logging.info(metrics)
        return trainer

    elif model_type == "setfit":
        model = SetFitModel.from_pretrained(
            base_model,
            labels=[l.name for l in labels],
        )

        args = TrainingArguments(
            output_dir="setfit",
            batch_size=batch_size,
            num_epochs=num_epochs,
            evaluation_strategy="epoch",
            save_strategy="epoch",
            load_best_model_at_end=True,
        )
        trainer = SetFitTrainer(
            model=model,
            args=args,
            train_dataset=label_dataset["train"],
            eval_dataset=label_dataset["test"],
            metric=metric,
            metric_kwargs=metric_kwargs,
            column_mapping={**column_mapping, "label": "label"},
        )

        # Train and evaluate
        trainer.train()
        metrics = trainer.evaluate(label_dataset["test"])
        logging.info(metrics)
        return trainer
    else:
        raise NotImplementedError("other approach is not implemented yet")
