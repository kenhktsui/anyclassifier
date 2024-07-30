import os
from dataclasses import dataclass, asdict
import warnings
from pathlib import Path
from typing import Dict, Optional
import fasttext
from datasets import Dataset
from setfit.trainer import ColumnMappingMixin
from anyclassifier.fasttext_wrapper.config import FastTextConfig
from anyclassifier.fasttext_wrapper.model import FastTextForSequenceClassification
from anyclassifier.fasttext_wrapper.utils import replace_newlines


# For Python 3.7 compatibility
try:
    from typing import Literal
except ImportError:
    from typing_extensions import Literal


def get_fasttext_cache_path():
    if os.name == 'nt':  # Windows
        base_path = os.path.expanduser('~\\AppData\\Local')
    elif os.name == 'posix':  # Unix-based systems (Linux, macOS)
        base_path = os.path.expanduser('~/.cache')
    else:
        raise OSError("Unsupported operating system")
    return str(Path(base_path) / 'fasttext_wrapper' / 'datasets')


@dataclass
class FastTextTrainingArguments:
    output_dir: str
    lr: float = 0.05
    epoch: int = 5
    neg: int = 5
    loss: Literal["ns", "hs", "softmax", "ova"] = "ns"
    lrUpdateRate = 100
    seed: int = 0
    data_txt_path: str = get_fasttext_cache_path()


class FastTextTrainer(ColumnMappingMixin):
    """Trainer to train a SetFit model.

    Args:
        model (`FastTextForSequenceClassification`):
            The model to train.
        args (`TrainingArguments`, *optional*):
            The training arguments to use.
        train_dataset (`Dataset`):
            The training dataset.
        eval_dataset (`Dataset`, *optional*):
            The evaluation dataset.
        column_mapping (`Dict[str, str]`, *optional*):
            A mapping from the column names in the dataset to the column names expected by the model.
            The expected format is a dictionary with the following format:
            `{"text_column_name": "text", "label_column_name: "label"}`.
    """

    def __init__(
        self,
        model: FastTextForSequenceClassification,
        args: FastTextConfig,
        train_dataset: Optional["Dataset"] = None,
        eval_dataset: Optional["Dataset"] = None,
        column_mapping: Optional[Dict[str, str]] = None,
    ) -> None:
        if args is not None and not isinstance(args, FastTextTrainingArguments):
            raise ValueError("`args` must be a `FastTextTrainingArguments` instance.")
        self.training_args = asdict(args)
        self.output_dir = self.training_args.pop("output_dir")
        self.data_txt_path = self.training_args.pop("data_txt_path")
        self.column_mapping = column_mapping
        if train_dataset:
            self._validate_column_mapping(train_dataset)
            if self.column_mapping is not None:
                train_dataset = self._apply_column_mapping(train_dataset, self.column_mapping)
        self.train_dataset = train_dataset

        if eval_dataset:
            self._validate_column_mapping(eval_dataset)
            if self.column_mapping is not None:
                eval_dataset = self._apply_column_mapping(eval_dataset, self.column_mapping)
        self.eval_dataset = eval_dataset

        self.model = model

    def _convert_dataset_to_fasttext_data(self, dataset: Dataset, filename: str) -> str:
        os.makedirs(self.data_txt_path, exist_ok=True)
        txt_path = os.path.join(self.data_txt_path, filename)
        with open(txt_path, "w") as f:
            for d in dataset:
                text = d[self.column_mapping["text"]]
                label = d[self.column_mapping["label"]]
                f.write(f"__label__{label} {replace_newlines(text)}")
                f.write("\n")
        return txt_path

    @staticmethod
    def print_results(N, p, r, header=""):
        print(header)
        print("N\t" + str(N))
        print("P@{}\t{:.3f}".format(1, p))
        print("R@{}\t{:.3f}".format(1, r))

    def train(
        self,
        args: Optional[FastTextTrainingArguments] = None,
        **kwargs,
    ) -> None:
        """
        Main training entry point.

        Args:
            args (`TrainingArguments`, *optional*):
                Temporarily change the training arguments for this training call.
        """
        if len(kwargs):
            warnings.warn(
                f"`{self.__class__.__name__}.train` does not accept keyword arguments anymore. "
                f"Please provide training arguments via a `TrainingArguments` instance to the `{self.__class__.__name__}` "
                f"initialisation or the `{self.__class__.__name__}.train` method.",
                DeprecationWarning,
                stacklevel=2,
            )

        if self.train_dataset is None:
            raise ValueError(
                f"Training requires a `train_dataset` given to the `{self.__class__.__name__}` initialization."
            )

        train_dataset_path = self._convert_dataset_to_fasttext_data(self.train_dataset, "train.txt")
        model = fasttext.train_supervised(
            input=train_dataset_path,
            **self.model.config.to_dict(),
            **self.training_args
        )
        self.model.model = model
        self.print_results(*self.model.model.test(train_dataset_path), "train metrics")
        os.remove(train_dataset_path)

        if self.eval_dataset is not None:
            eval_dataset_path = self._convert_dataset_to_fasttext_data(self.train_dataset, "eval.txt")
            self.print_results(*self.model.model.test(eval_dataset_path), "test metrics")
            os.remove(eval_dataset_path)

        self.model.save_pretrained(self.output_dir)

    def evaluate(self, dataset: Optional[Dataset] = None) -> Dict[str, float]:
        """
        Computes the metrics for a given classifier.

        Args:
            dataset (`Dataset`, *optional*):
                The dataset to compute the metrics on. If not provided, will use the evaluation dataset passed via
                the `eval_dataset` argument at `Trainer` initialization.

        Returns:
            `Dict[str, float]`: The evaluation metrics.
        """
        self._validate_column_mapping(dataset)
        if self.column_mapping is not None:
            dataset = self._apply_column_mapping(dataset, self.column_mapping)
        dataset_path = self._convert_dataset_to_fasttext_data(dataset, "evaluate.txt")
        n, precision, recall = self.model.model.test(dataset_path)
        self.print_results(n, precision, recall, "")
        os.remove(dataset_path)
        return {
            "precision": precision,
            "recall": recall
        }

    def push_to_hub(self, repo_id: str, **kwargs) -> str:
        """Upload model checkpoint to the Hub using `huggingface_hub`.

        See the full list of parameters for your `huggingface_hub` version in the\
        [huggingface_hub documentation](https://huggingface.co/docs/huggingface_hub/package_reference/mixins#huggingface_hub.ModelHubMixin.push_to_hub).

        Args:
            repo_id (`str`):
                The full repository ID to push to, e.g. `"tomaarsen/setfit-sst2"`.
            config (`dict`, *optional*):
                Configuration object to be saved alongside the model weights.
            commit_message (`str`, *optional*):
                Message to commit while pushing.
            private (`bool`, *optional*, defaults to `False`):
                Whether the repository created should be private.
            api_endpoint (`str`, *optional*):
                The API endpoint to use when pushing the model to the hub.
            token (`str`, *optional*):
                The token to use as HTTP bearer authorization for remote files.
                If not set, will use the token set when logging in with
                `transformers-cli login` (stored in `~/.huggingface`).
            branch (`str`, *optional*):
                The git branch on which to push the model. This defaults to
                the default branch as specified in your repository, which
                defaults to `"main"`.
            create_pr (`boolean`, *optional*):
                Whether or not to create a Pull Request from `branch` with that commit.
                Defaults to `False`.
            allow_patterns (`List[str]` or `str`, *optional*):
                If provided, only files matching at least one pattern are pushed.
            ignore_patterns (`List[str]` or `str`, *optional*):
                If provided, files matching any of the patterns are not pushed.

        Returns:
            str: The url of the commit of your model in the given repository.
        """
        if "/" not in repo_id:
            raise ValueError(
                '`repo_id` must be a full repository ID, including organisation'
            )
        commit_message = kwargs.pop("commit_message", "Add FastText model")
        return self.model.push_to_hub(repo_id, commit_message=commit_message, **kwargs)
