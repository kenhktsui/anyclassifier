from huggingface_hub import hf_hub_download
from datasets import load_dataset
from anyclassifier.schema import Label
from anyclassifier import train_anyclassifier
from setfit import SetFitModel


HF_HANDLE = "user_id"


dataset = load_dataset("ccdv/arxiv-classification")
# mock unlabeled data
unlabeled_dataset = dataset["train"].remove_columns("label")

trainer = train_anyclassifier(
        "Classify an academic paper into topics.",
    [
        Label(id=0, desc='Commutative Algebra'),
        Label(id=1, desc='Computer Vision'),
        Label(id=2, desc='Artificial Intelligence'),
        Label(id=3, desc='Systems and Control'),
        Label(id=4, desc='Group Theory'),
        Label(id=5, desc='Computational Engineering'),
        Label(id=6, desc='Programming Languages'),
        Label(id=7, desc='Information Theory'),
        Label(id=8, desc='Data Structures'),
        Label(id=9, desc='Neural and Evolutionary'),
        Label(id=10, desc='Statistics Theory'),
    ],
    hf_hub_download("lmstudio-community/Meta-Llama-3.1-8B-Instruct-GGUF", "Meta-Llama-3.1-8B-Instruct-Q8_0.gguf"),
    unlabeled_dataset,
    column_mapping={"text": "text"},
    model_type="setfit",
    n_record_to_label=270,
    num_epochs=5,
    push_dataset_to_hub=True,
    is_dataset_private=True,
    metric="f1",
    metric_kwargs={"average": "micro"},
    dataset_repo_id=f"{HF_HANDLE}/test_arxiv_classification"
)
full_test_data = dataset["test"]

print(trainer.evaluate(full_test_data))

trainer.push_to_hub(f"{HF_HANDLE}/setfit_test_arxiv_classification", private=True)

model = SetFitModel.from_pretrained(f"{HF_HANDLE}/setfit_test_arxiv_classification")
