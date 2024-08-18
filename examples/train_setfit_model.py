from huggingface_hub import hf_hub_download
from datasets import load_dataset
from anyclassifier.schema import Label
from anyclassifier import train_anyclassifier
from setfit import SetFitModel


HF_HANDLE = "user_id"


dataset = load_dataset("stanfordnlp/imdb")
# mock unlabeled data
unlabeled_dataset = dataset["train"].remove_columns("label")

trainer = train_anyclassifier(
    "Classify a text's sentiment.",
    [
        Label(id=1, desc='positive sentiment'),
        Label(id=0, desc='negative sentiment')
    ],
    hf_hub_download("lmstudio-community/Meta-Llama-3.1-8B-Instruct-GGUF", "Meta-Llama-3.1-8B-Instruct-Q8_0.gguf"),
    unlabeled_dataset,
    column_mapping={"text": "text"},
    model_type="setfit",
    n_record_to_label=40,
    num_epochs=5,
    push_dataset_to_hub=True,
    is_dataset_private=True,
    dataset_repo_id=f"{HF_HANDLE}/test"
)
full_test_data = dataset["test"]

print(trainer.evaluate(full_test_data))

trainer.push_to_hub(f"{HF_HANDLE}/setfit_test", private=True)

model = SetFitModel.from_pretrained(f"{HF_HANDLE}/setfit_test")
# Run inference
text = ["i loved the spiderman movie!", "pineapple on pizza is the worst 🤮"]
preds = model.predict(text)
print(text)
print(preds)
# ['1', '0']
