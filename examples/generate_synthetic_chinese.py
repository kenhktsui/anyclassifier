import asyncio
from huggingface_hub import hf_hub_download
from anyclassifier.annotation.prompt import Label
from anyclassifier.llm.llm_client import LlamaCppClient, OpenAIClient
from anyclassifier.synthetic_data_generation import SyntheticDataGeneratorForSequenceClassification

HF_HANDLE = "user_id"

llm_client = LlamaCppClient()
# or llm_client = OpenAIClient()

data_gen = SyntheticDataGeneratorForSequenceClassification(llm_client)

dataset = asyncio.run(data_gen.generate(
    "Classify a Chinese text's sentiment.",
    [
        Label(id=0, name='0', desc='negative sentiment'),
        Label(id=1, name='1', desc='positive sentiment')
    ]
))
dataset.push_to_hub(f"{HF_HANDLE}/traditional_chinese_sentiment_syn")
