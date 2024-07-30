import os
from typing import Optional, Union, List
from transformers.utils.hub import PushToHubMixin, cached_file, is_remote_url, download_url
from transformers.utils import CONFIG_NAME
import fasttext
from anyclassifier.fasttext_wrapper.config import FastTextConfig
from anyclassifier.fasttext_wrapper.utils import replace_newlines


FASTTEXT_WEIGHTS_NAME = "model.bin"


class FastTextForSequenceClassification(PushToHubMixin):
    config_class = FastTextConfig

    def __init__(self, config: Optional[FastTextConfig] = None, model=None):
        self.config = config
        self.model = model

    def predict(self, text_list: List[str]):
        if self.model is None:
            raise ValueError("Model is not yet trained, please pass to FastTextTrainer for training.")
        return self.model.predict([replace_newlines(t) for t in text_list])

    @classmethod
    def from_pretrained(
        cls,
        pretrained_model_name_or_path: str,
        cache_dir: Optional[Union[str, os.PathLike]] = None,
        force_download: bool = False,
        local_files_only: bool = False,
        token: Optional[Union[str, bool]] = None,
        revision: str = "main",
    ):
        config = None
        is_local_dir = os.path.isdir(pretrained_model_name_or_path)
        if is_local_dir:
            archive_file = os.path.join(pretrained_model_name_or_path, FASTTEXT_WEIGHTS_NAME)
        elif os.path.isfile(pretrained_model_name_or_path):
            archive_file = pretrained_model_name_or_path
        elif is_remote_url(pretrained_model_name_or_path):
            archive_file = download_url(pretrained_model_name_or_path)
        else:
            # set correct filename
            try:
                # Load from URL or cache if already cached
                cached_file_kwargs = {
                    "cache_dir": cache_dir,
                    "force_download": force_download,
                    "local_files_only": local_files_only,
                    "token": token,
                    "revision": revision,
                    "_raise_exceptions_for_gated_repo": False,
                    "_raise_exceptions_for_missing_entries": False,
                }
                archive_file = cached_file(pretrained_model_name_or_path, FASTTEXT_WEIGHTS_NAME, **cached_file_kwargs)
                config_file = cached_file(pretrained_model_name_or_path, CONFIG_NAME, **cached_file_kwargs)
                config = FastTextConfig(config_file)

            except EnvironmentError:
                # Raise any environment error raise by `cached_file`. It will have a helpful error message adapted
                # to the original exception.
                raise
            except Exception as e:
                # For any other exception, we throw a generic error.
                raise EnvironmentError(
                    f"Can't load the model for '{pretrained_model_name_or_path}'. If you were trying to load it"
                    " from 'https://huggingface.co/models', make sure you don't have a local directory with the"
                    f" same name. Otherwise, make sure '{pretrained_model_name_or_path}' is the correct path to a"
                    f" directory containing a file named {FASTTEXT_WEIGHTS_NAME}."
                ) from e

        model = fasttext.load_model(archive_file)

        return cls(config=config, model=model)

    def save_pretrained(self, checkpoint_file_path: str, **kwargs):
        # ignore max_shard_size and safe_serialization, not applicable to fasttext_wrapper
        os.makedirs(checkpoint_file_path, exist_ok=True)
        self.model.save_model(os.path.join(checkpoint_file_path, FASTTEXT_WEIGHTS_NAME))
        self.config.to_json_file(os.path.join(checkpoint_file_path, CONFIG_NAME), use_diff=True)
