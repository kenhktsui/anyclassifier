from huggingface_hub import hf_hub_download
from datasets import load_dataset
from anyclassifier.schema import Label
from anyclassifier import train_anyclassifier
from setfit import SetFitModel


HF_HANDLE = "user_id"


dataset = load_dataset("lmsys/toxic-chat", "toxicchat0124")
dataset = dataset.rename_column("toxicity", "label")
dataset = dataset.rename_column("user_input", "text")
# mock unlabeled data
unlabeled_dataset = dataset["train"].remove_columns("label")


trainer = train_anyclassifier(
        "Classify if a LLM prompt or chat is toxic.",
    [
        Label(id=0, desc='Non toxic'),
        Label(id=1, desc='Toxic')
    ],
    hf_hub_download("lmstudio-community/Meta-Llama-3.1-8B-Instruct-GGUF", "Meta-Llama-3.1-8B-Instruct-Q8_0.gguf"),
    unlabeled_dataset,
    column_mapping={"text": "text"},
    model_type="setfit",
    n_record_to_label=60,
    num_epochs=5,
    push_dataset_to_hub=True,
    is_dataset_private=True,
    metric="f1",
    metric_kwargs={"average": "micro"},
    dataset_repo_id=f"{HF_HANDLE}/test_toxic_chat"
)
full_test_data = dataset["test"]

print(trainer.evaluate(full_test_data))

trainer.push_to_hub(f"{HF_HANDLE}/setfit_test_toxic_chat", private=True)

model = SetFitModel.from_pretrained(f"{HF_HANDLE}/setfit_test_toxic_chat")