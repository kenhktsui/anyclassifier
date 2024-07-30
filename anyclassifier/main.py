from typing import List, Dict, Union, Literal, Optional
from datasets import Dataset
from huggingface_hub import interpreter_login
from setfit import SetFitModel, TrainingArguments, Trainer as SetFitTrainer
from anyclassifier.annotation.prompt import Label, AnnotationPrompt
from anyclassifier.annotation.annotator import LlamaCppAnnotator
from anyclassifier.fasttext_wrapper import FastTextConfig, FastTextTrainer, FastTextForSequenceClassification, FastTextTrainingArguments


def build_anyclassifier(
    instruction: str,
    annotator_model_path: str,
    labels: List[Label],
    unlabeled_dataset: Dataset,
    column_mapping: Dict[str, str] = {"text": "text"},
    model_type: Literal["setfit", "fasttext", "transformers"] = "setfit",
    base_model: Optional[str] = "sentence-transformers/paraphrase-mpnet-base-v2",
    test_size: float = 0.3,
    n_record_to_label: int = 100,
    push_dataset_to_hub: bool = False,
    dataset_repo_id: Optional[str] = None,
    is_dataset_private: Optional[bool] = True
    ) -> Union[FastTextTrainer, SetFitTrainer]:

    if push_dataset_to_hub:
        interpreter_login()
        assert dataset_repo_id is not None, "dataset_repo_id must be provided when push_dataset_to_hub is True."

    # labeling dataset
    prompt = AnnotationPrompt(
        task_description=instruction,
        label_definition=labels,
    )
    annotator = LlamaCppAnnotator(prompt, annotator_model_path)
    label_dataset = annotator.annotate_dataset(unlabeled_dataset, n_record=n_record_to_label)

    label_dataset = label_dataset.train_test_split(test_size=test_size)

    if push_dataset_to_hub:
        label_dataset.push_to_hub(dataset_repo_id, private=is_dataset_private)

    # training
    if model_type == "fasttext":
        config = FastTextConfig()
        model = FastTextForSequenceClassification(config)
        args = FastTextTrainingArguments("fasttext_model")
        trainer = FastTextTrainer(
            model=model,
            args=args,
            train_dataset=label_dataset["train"],
            eval_dataset=label_dataset["test"],
            column_mapping={**column_mapping, "label": "label"},
        )

        # Train and evaluate
        trainer.train()
        metrics = trainer.evaluate(label_dataset["test"])
        print(metrics)
        return trainer

    elif model_type == "setfit":
        model = SetFitModel.from_pretrained(
            base_model,
            labels=[l.name for l in labels],
        )

        args = TrainingArguments(
            output_dir="setfit",
            batch_size=16,
            num_epochs=1,
            evaluation_strategy="epoch",
            save_strategy="epoch",
            load_best_model_at_end=True,
        )
        trainer = SetFitTrainer(
            model=model,
            args=args,
            train_dataset=label_dataset["train"],
            eval_dataset=label_dataset["test"],
            metric="accuracy",
            column_mapping={**column_mapping, "label": "label"},
        )

        # Train and evaluate
        trainer.train()
        metrics = trainer.evaluate(label_dataset["test"])
        print(metrics)
        return trainer
    else:
        raise NotImplementedError("other approach is not implemented yet")
