![GitHub License](https://img.shields.io/github/license/kenhktsui/anyclassifier)

# AnyClassifier - One Line To Build Any Classifier Without Data
Have you ever wanted to build a classifier without any labeled data? What it takes now is just a few lines 🤯.  
**AnyClassifier** is a framework that helps you build a classifier **without** any label.   
As a machine learning engineer, one of the unavoidable and heavy lifting issues is to build and filter a high quality labelled data.  
By leveraging LLM 🤖 annotation, one can now label data at a better quality and at lightning speed ever.   
This is built for **machine learning engineer and software engineer**, by **machine learning engineer** 👨🏻‍💻.   


## Feature
- One line to build any classifier that you don't have data 🤯
- Smoothness integration with transformers, setfit, fasttext and datasets
  - [setfit](https://github.com/huggingface/setfit): for limited data 🤗
  - [fastText](https://github.com/facebookresearch/fastText): for blazingly fast inference without GPU ⚡️
  - [transformers](https://github.com/huggingface/transformers): for generic purpose
- Huggingface-like interface for fastText that supports push_to_hub, saving and loading.

## QuickStart in Colab
[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/drive/1LB8PUTT9wM1Qb2cY-6Dx-RNiqmyCvRr1?usp=sharing)

## Installation
### Metal Backend
```shell
CMAKE_ARGS="-DGGML_CUDA=on" pip install anyclassifier
```

### CUDA Backend
```shell
CMAKE_ARGS="-DGGML_METAL=on" pip install anyclassifier
```

### CPU
```shell
CMAKE_ARGS="-DGGML_BLAS=ON -DGGML_BLAS_VENDOR=OpenBLAS" pip install anyclassifier
```

### Developer's installation
```shell
pip install -e .
```

## Usage
### Download Llama3 (please accept the terms and condition of llama3.1 license in [here](https://huggingface.co/meta-llama/Meta-Llama-3.1-8B-Instruct) beforehand)
```python
from huggingface_hub import hf_hub_download


hf_hub_download("lmstudio-community/Meta-Llama-3.1-8B-Instruct-GGUF", "Meta-Llama-3.1-8B-Instruct-Q8_0.gguf")
```

### One Liner

```python
from huggingface_hub import hf_hub_download
from anyclassifier import build_anyclassifier
from anyclassifier.annotation.prompt import Label

unlabeled_dataset  # a datasets.Dataset class can be from your local json/ csv, or from huggingface hub.

# Magic One Liner!
trainer = build_anyclassifier(
  "Classify a text's sentiment.",
  hf_hub_download("lmstudio-community/Meta-Llama-3.1-8B-Instruct-GGUF", "Meta-Llama-3.1-8B-Instruct-Q8_0.gguf"),
  [
    Label(name='1', desc='positive sentiment'),
    Label(name='0', desc='negative sentiment')
  ],
  unlabeled_dataset,
  column_mapping={"text": "text"},
  model_type="setfit",  # can be set to fastText
  push_dataset_to_hub=True,  # we recommend to push your dataset to huggingface, so that you won't lose it
  dataset_repo_id="user_id/test",
  is_dataset_private=True
)
# Share Your Model!
trainer.push_to_hub("user_id/any_model")
```

### To Use Model

```python
# FastText
from anyclassifier.fasttext_wrapper import FastTextForSequenceClassification

model = FastTextForSequenceClassification.from_pretrained("user_id/any_model")
preds = model.predict(["i loved the spiderman movie!", "pineapple on pizza is the worst 🤮"])
print(preds)

# SetFit
from setfit import SetFitModel

model = SetFitModel.from_pretrained("user_id/any_model")
preds = model.predict(["i loved the spiderman movie!", "pineapple on pizza is the worst 🤮"])
print(preds)
```

### Label Dataset

```python
from datasets import load_dataset
from anyclassifier.annotation.prompt import AnnotationPrompt, Label
from anyclassifier.annotation.annotator import LlamaCppAnnotator

unlabeled_dataset = load_dataset("somepath")
prompt = AnnotationPrompt(
  task_description="Classify a text's sentiment.",
  label_definition=[
    Label(name='1', desc='positive sentiment'),
    Label(name='0', desc='negative sentiment')
  ]
)
annotator = LlamaCppAnnotator(prompt)
label_dataset = annotator.annotate_dataset(unlabeled_dataset, n_record=1000)
label_dataset.push_to_hub('user_id/any_data')

```

See examples:  

| model_type | example                                  | resulting model                                                                  | dataset                                                                      |
|------------|------------------------------------------|----------------------------------------------------------------------------------|------------------------------------------------------------------------------|
| setfit     | [link](examples/train_setfit_model.py)   | [link](https://huggingface.co/kenhktsui/anyclassifier_setfit_demo)               | [link](https://huggingface.co/datasets/kenhktsui/anyclassifier_dataset_demo) |
| fasttext   | [link](examples/train_fasttext_model.py) | [link](https://huggingface.co/kenhktsui/fasttext_test)(probably need more label) | [link](https://huggingface.co/datasets/kenhktsui/anyclassifier_dataset_demo) |



## Future Roadmap
- High Quality Data:
  - Prompt validation
  - Label validation - inter-model annotation
- High Quality Model
  - Auto error analysis
  - Auto model documentation
  - Auto synthetic data
