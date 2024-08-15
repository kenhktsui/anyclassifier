![GitHub License](https://img.shields.io/github/license/kenhktsui/anyclassifier?)![PyPI - Downloads](https://img.shields.io/pypi/dm/anyclassifier?)![PyPI - Version](https://img.shields.io/pypi/v/anyclassifier?)

# ∞🧙🏼‍♂️AnyClassifier - One Line To Build Any Classifier Without Data, And A Step Towards The First AI ML Engineer
![image](assets/Traditional_ML_Cycle.png)

![image](assets/AnyClassifier.png)

>Have you ever wanted/ been requested to build a classifier without any data? What it takes now is just one line 🤯.   

**AnyClassifier** is a framework that helps you build a classifier **without** any label/ **even with no data**, with as limited coding as possible.    
As a machine learning engineer, one of the unavoidable but the most heavy lifting issues is to build and curate a high quality labelled data.   
By leveraging LLM 🤖 annotation with permissive license, one can now label and even generate training data at a better quality and at lightning speed ever.    
This is inspired by some of the challenges I faced daily in work and doing open source - it is built for **machine learning engineer and software engineer**, by **machine learning engineer** 👨🏻‍💻.    
By providing a higher level abstraction, this project's mission is to further **democratizes** AI to everyone, with **ONE LINE**.   
The project is still in experimental mode, but I found it worked in some of my use cases. Feedbacks welcome, and feel free to contribute. See [Roadmap](#-roadmap).  
Together let's build more useful models.

## 🚀 Features
- One line to build any classifier that you don't have data (**No** label/ data) 🤯
- **Competitive** result even when **No** data is provided because of synthetic data 🤯🤯. See [Benchmark](#benchmark)
- Why one line? Not only it is easy to be used by Human but also it can easily be used by other LLM as a function call, easily to be integrated with any **agentic flow**
- Synthetic Data Generation and Annotation Module for standalone usage
- Smoothness integration with transformers, setfit, fasttext and datasets
  - [setfit](https://github.com/huggingface/setfit): for limited data (e.g. 100) 🤗
  - [fastText](https://github.com/facebookresearch/fastText): for blazingly fast inference (1000 docs/s) without GPU ⚡️
- Huggingface-like interface for fastText that supports push_to_hub, saving and loading (let's not forget this amazing model before transformers architecture).

## 🏁 QuickStart in Apple Silicon - Train a model in 5 min!
<details>
<summary>Expand</summary>

```shell
# install (cp39 = python3.9, other valid values are cp310, cp311, cp312)
curl -L -O https://github.com/abetlen/llama-cpp-python/releases/download/v0.2.88-metal/llama_cpp_python-0.2.88-cp39-cp39-macosx_11_0_arm64.whl
pip install llama_cpp_python-0.2.88-cp39-cp39-macosx_11_0_arm64.whl
rm llama_cpp_python-0.2.88-cp39-cp39-macosx_11_0_arm64.whl
pip install anyclassifier
# run
cd examples
python train_setfit_model.py
```
</details>

![image](assets/demo.gif)
## 🏁 QuickStart in Colab

| Dataset                       | Colab Link                                                                                                                                                          |
|-------------------------------|---------------------------------------------------------------------------------------------------------------------------------------------------------------------|
| No Data!                      | [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/drive/1Wt_IlilfTqBbyn3gZQ3kITjObrVAypyi?usp=sharing) |
| imdb sentiment classification | [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/drive/1LB8PUTT9wM1Qb2cY-6Dx-RNiqmyCvRr1?usp=sharing) |


## 🛠️ Usage
### Download a small LLM (please accept the respective terms and condition of model license beforehand)
[meta-llama/Meta-Llama-3.1-8B-Instruct](https://huggingface.co/meta-llama/Meta-Llama-3.1-8B-Instruct)  
[google/gemma-2-9b](https://huggingface.co/google/gemma-2-9b)
```python
from huggingface_hub import hf_hub_download

# meta-llama/Meta-Llama-3.1-8B-Instruct
hf_hub_download("lmstudio-community/Meta-Llama-3.1-8B-Instruct-GGUF", "Meta-Llama-3.1-8B-Instruct-Q8_0.gguf")

# google/gemma-2-9b
hf_hub_download("lmstudio-community/gemma-2-9b-it-GGUF", "gemma-2-9b-it-Q8_0.gguf")
```

### One Liner For No Data

```python
from huggingface_hub import hf_hub_download
from anyclassifier import build_anyclassifier
from anyclassifier.schema import Label

# Magic One Line!
trainer = build_anyclassifier(
  "Classify a text's sentiment.",
  [
    Label(id=1, desc='positive sentiment'),
    Label(id=0, desc='negative sentiment')
  ],
  hf_hub_download("lmstudio-community/Meta-Llama-3.1-8B-Instruct-GGUF", "Meta-Llama-3.1-8B-Instruct-Q8_0.gguf")
)
# Share Your Model!
trainer.push_to_hub("user_id/any_model")
```

### One Liner For Unlabeled Data
```python
from huggingface_hub import hf_hub_download
from anyclassifier import build_anyclassifier
from anyclassifier.schema import Label

unlabeled_dataset  # a huggingface datasets.Dataset class can be from your local json/ csv, or from huggingface hub.

# Magic One Line!
trainer = build_anyclassifier(
  "Classify a text's sentiment.",
  [
    Label(id=1, desc='positive sentiment'),
    Label(id=0, desc='negative sentiment')
  ],
  hf_hub_download("lmstudio-community/Meta-Llama-3.1-8B-Instruct-GGUF", "Meta-Llama-3.1-8B-Instruct-Q8_0.gguf"),  # as you like
  unlabeled_dataset
)
# Share Your Model!
trainer.push_to_hub("user_id/any_model")
```

### To Use Model

```python
# SetFit
from setfit import SetFitModel

model = SetFitModel.from_pretrained("user_id/any_model")
preds = model.predict(["i loved the spiderman movie!", "pineapple on pizza is the worst 🤮"])
print(preds)

# FastText
from anyclassifier.fasttext_wrapper import FastTextForSequenceClassification

model = FastTextForSequenceClassification.from_pretrained("user_id/any_model")
preds = model.predict(["i loved the spiderman movie!", "pineapple on pizza is the worst 🤮"])
print(preds)
```

## 🔧 Installation
It is using llama.cpp as backend, and build wheel can take a lot of time (10min+), as such, we also provide an instruction to install with pre-built wheel.
<details>
<summary>Metal Backend (Apple's GPU) - cp39 = python3.9, other valid values are cp310, cp311, cp312</summary>

```shell
curl -L -O https://github.com/abetlen/llama-cpp-python/releases/download/v0.2.88-metal/llama_cpp_python-0.2.88-cp39-cp39-macosx_11_0_arm64.whl
pip install llama_cpp_python-0.2.88-cp39-cp39-macosx_11_0_arm64.whl
rm llama_cpp_python-0.2.88-cp39-cp39-macosx_11_0_arm64.whl
pip install anyclassifier
```
</details>

<details>
<summary>Colab (T4) Prebuilt Wheel</summary>

```shell
curl -L -O https://github.com/abetlen/llama-cpp-python/releases/download/v0.2.88-cu124/llama_cpp_python-0.2.88-cp310-cp310-linux_x86_64.whl
pip install llama_cpp_python-0.2.88-cp310-cp310-linux_x86_64.whl
rm llama_cpp_python-0.2.88-cp310-cp310-linux_x86_64.whl
pip install anyclassifier
```
</details>

<details>
<summary>CUDA Backend (Please read [llama-cpp-python](https://llama-cpp-python.readthedocs.io/en/latest/#installation))</summary>

```shell
CMAKE_ARGS="-DGGML_METAL=on" pip install anyclassifier
```
</details>

<details>
<summary>CPU</summary>

```shell
CMAKE_ARGS="-DGGML_BLAS=ON -DGGML_BLAS_VENDOR=OpenBLAS" pip install anyclassifier
```
</details>

<details>
<summary>Developer's installation</summary>

```shell
pip install -e .
```
</details>

## Other Usages
### To Label a Dataset

```python
from datasets import load_dataset
from anyclassifier.annotation.prompt import AnnotationPrompt
from anyclassifier.schema import Label
from anyclassifier.annotation.annotator import LlamaCppAnnotator

unlabeled_dataset = load_dataset("somepath")
prompt = AnnotationPrompt(
  task_description="Classify a text's sentiment.",
  label_definition=[
      Label(id=0, desc='negative sentiment'),
      Label(id=1, desc='positive sentiment')
  ]  
)
annotator = LlamaCppAnnotator(prompt)
label_dataset = annotator.annotate_dataset(unlabeled_dataset, n_record=1000)
label_dataset.push_to_hub('user_id/any_data')
```

### To Generate a Synthetic Dataset
```python
from anyclassifier.schema import Label
from anyclassifier.synthetic_data_generation import SyntheticDataGeneratorForSequenceClassification

tree_constructor = SyntheticDataGeneratorForSequenceClassification()
dataset = tree_constructor.generate(
    "Classify a text's sentiment.",
    [
        Label(id=0, desc='negative sentiment'),
        Label(id=1, desc='positive sentiment')
    ]
)
dataset.push_to_hub('user_id/any_data')
```

See more examples:  

| approach | model_type | example                                          | resulting model   | dataset   |
|----------|------------|--------------------------------------------------|----------------|---------------|
|synthetic data generation| setfit     | [link](examples/train_setfit_model_synthetic.py) | [link](https://huggingface.co/kenhktsui/setfit_test_syn)               | [link](https://huggingface.co/datasets/kenhktsui/test_syn) |
|annotation| setfit     | [link](examples/train_setfit_model.py)           | [link](https://huggingface.co/kenhktsui/anyclassifier_setfit_demo)               | [link](https://huggingface.co/datasets/kenhktsui/anyclassifier_dataset_demo) |
|annotation| fasttext   | [link](examples/train_fasttext_model.py)         | [link](https://huggingface.co/kenhktsui/fasttext_test)(probably need more label) | [link](https://huggingface.co/datasets/kenhktsui/anyclassifier_dataset_demo) |

## 🧪Benchmark
Model performance of synthetic data is at par, if not exceeds, that of real data.

|dataset | approach                               |  model_type                                                       | metrics         |
|---|----------------------------------------|----------------------------------------|-----------------|
|imdb| full training dataset                  | [lvwerra/distilbert-imdb](https://huggingface.co/lvwerra/distilbert-imdbb) | Accuracy: 92.8% |
|imdb| synthetic data generation (28 samples) |setfit                                                                     | Accuracy: 88.8% |
|imdb| annotation (30 records)                |setfit                                                                    | Accuracy: 85.9% |
|zeroshot/twitter-financial-news-sentiment| full training dataset                  | [nickmuchi/finbert-tone-finetuned-fintwitter-classification](https://huggingface.co/nickmuchi/finbert-tone-finetuned-fintwitter-classification) | F1: 0.884       |
|zeroshot/twitter-financial-news-sentiment| synthetic data generation (46 samples) |setfit                                                                     | F1: 0.606       |
|zeroshot/twitter-financial-news-sentiment| annotation (42 records)                |setfit                                                                    | F1: 0.668       |


## 🗺️ Roadmap
- High Quality Data:
  - Prompt validation
  - Label validation - inter-model annotation
- High Quality Model
  - Auto error analysis
  - Auto model documentation
  - Auto synthetic data
- More Benchmarking

# 👋 Contributing
- build models with AnyClassifier
- suggest features
- create issue/ PR
- benchmarking synthetic data generation vs annotation vs full training dataset

# 📩 Follow Me For Update:
[X](https://x.com/kenhktsui)/ [huggingface](https://huggingface.co/kenhktsui)/ [github](https://github.com/kenhktsui)
