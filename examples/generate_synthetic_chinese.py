from huggingface_hub import hf_hub_download
from anyclassifier.annotation.prompt import Label
from anyclassifier.synthetic_data_generation import SyntheticDataGeneratorForSequenceClassification


HF_HANDLE = "user_id"

data_gen = SyntheticDataGeneratorForSequenceClassification(
    hf_hub_download("lmstudio-community/Meta-Llama-3.1-8B-Instruct-GGUF", "Meta-Llama-3.1-8B-Instruct-Q8_0.gguf")
)
dataset = data_gen.generate(
    "Classify a Chinese text's sentiment.",
    [
        Label(id=0, name='0', desc='negative sentiment'),
        Label(id=1, name='1', desc='positive sentiment')
    ]
)
dataset.push_to_hub(f"{HF_HANDLE}/traditional_chinese_sentiment_syn")
