from huggingface_hub import hf_hub_download
from datasets import load_dataset
from anyclassifier.schema import Label
from anyclassifier import train_anyclassifier
from setfit import SetFitModel


HF_HANDLE = "user_id"


dataset = load_dataset("fancyzhx/ag_news")

trainer = train_anyclassifier(
        "Classify a news into categories.",
    [
        Label(id=0, desc='World'),
        Label(id=1, desc='Sports'),
        Label(id=2, desc='Business'),
        Label(id=3, desc='Sci/Tech'),
    ],
    hf_hub_download("lmstudio-community/Meta-Llama-3.1-8B-Instruct-GGUF", "Meta-Llama-3.1-8B-Instruct-Q8_0.gguf"),
    column_mapping={"text": "text"},
    model_type="setfit",
    n_record_to_generate=60,
    num_epochs=5,
    push_dataset_to_hub=True,
    is_dataset_private=True,
    metric="accuracy",
    dataset_repo_id=f"{HF_HANDLE}/test_ag_news_syn"
)
full_test_data = dataset["test"]

print(trainer.evaluate(full_test_data))

trainer.push_to_hub(f"{HF_HANDLE}/test_ag_news_syn", private=True)

model = SetFitModel.from_pretrained(f"{HF_HANDLE}/test_ag_news_syn")
