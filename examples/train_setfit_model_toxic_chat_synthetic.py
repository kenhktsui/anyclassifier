from huggingface_hub import hf_hub_download
from datasets import load_dataset
from anyclassifier.schema import Label
from anyclassifier import train_anyclassifier
from setfit import SetFitModel


HF_HANDLE = "user_id"


dataset = load_dataset("lmsys/toxic-chat", "toxicchat0124")
dataset = dataset.rename_column("toxicity", "label")
dataset = dataset.rename_column("user_input", "text")

trainer = train_anyclassifier(
        "Classify if a LLM prompt or chat is toxic.",
    [
        Label(id=0, desc='Non toxic'),
        Label(id=1, desc='Toxic')
    ],
    hf_hub_download("lmstudio-community/Meta-Llama-3.1-8B-Instruct-GGUF", "Meta-Llama-3.1-8B-Instruct-Q8_0.gguf"),
    column_mapping={"text": "text"},
    model_type="setfit",
    n_record_to_generate=60,
    num_epochs=5,
    push_dataset_to_hub=True,
    is_dataset_private=True,
    metric="f1",
    metric_kwargs={"average": "micro"},
    dataset_repo_id=f"{HF_HANDLE}/test_toxic_chat_syn"
)
full_test_data = dataset["test"]
print(full_test_data)

print(trainer.evaluate(full_test_data))

trainer.push_to_hub(f"{HF_HANDLE}/setfit_test_toxic_chat_syn", private=True)

model = SetFitModel.from_pretrained(f"{HF_HANDLE}/setfit_test_toxic_chat_syn")
