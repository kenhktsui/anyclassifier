from datasets import load_dataset
from setfit import SetFitModel, TrainingArguments, Trainer as SetFitTrainer

args = TrainingArguments()

### stanfordnlp/imdb
dataset = load_dataset("stanfordnlp/imdb")
full_test_data = dataset["test"]
model = SetFitModel.from_pretrained(f"kenhktsui/setfit_test_syn")
trainer = SetFitTrainer(model=model, args=args, eval_dataset=full_test_data, metric="accuracy")
print("kenhktsui/setfit_test_syn", trainer.evaluate(full_test_data))
model = SetFitModel.from_pretrained(f"kenhktsui/setfit_test")
trainer = SetFitTrainer(model=model, args=args, eval_dataset=full_test_data, metric="accuracy")
print("kenhktsui/setfit_test", trainer.evaluate(full_test_data))

### zeroshot/twitter-financial-news-sentiment
dataset = load_dataset("zeroshot/twitter-financial-news-sentiment")
full_test_data = dataset["validation"]
model = SetFitModel.from_pretrained(f"kenhktsui/setfit_test_twitter_news_syn")
trainer = SetFitTrainer(model=model, args=args, eval_dataset=full_test_data, metric="f1", metric_kwargs={"average": "weighted"})
print("kenhktsui/setfit_test_twitter_news_syn", trainer.evaluate(full_test_data))
model = SetFitModel.from_pretrained(f"kenhktsui/setfit_test_twitter_news")
trainer = SetFitTrainer(model=model, args=args, eval_dataset=full_test_data, metric="f1", metric_kwargs={"average": "weighted"})
print("kenhktsui/setfit_test_twitter_news", trainer.evaluate(full_test_data))

###  ccdv/arxiv-classification
dataset = load_dataset("ccdv/arxiv-classification")
full_test_data = dataset["test"]
model = SetFitModel.from_pretrained(f"kenhktsui/setfit_test_arxiv_classification_syn")
trainer = SetFitTrainer(model=model, args=args, eval_dataset=full_test_data, metric="accuracy")
print("kenhktsui/setfit_test_arxiv_classification_syn", trainer.evaluate(full_test_data))
model = SetFitModel.from_pretrained(f"kenhktsui/setfit_test_arxiv_classification")
trainer = SetFitTrainer(model=model, args=args, eval_dataset=full_test_data, metric="accuracy")
print("kenhktsui/setfit_test_arxiv_classification", trainer.evaluate(full_test_data))

### ToxicChat
dataset = load_dataset("lmsys/toxic-chat", "toxicchat0124")
dataset = dataset.rename_column("toxicity", "label")
dataset = dataset.rename_column("user_input", "text")
full_test_data = dataset["test"]
model = SetFitModel.from_pretrained(f"kenhktsui/setfit_test_toxic_chat_syn")
trainer = SetFitTrainer(model=model, args=args, eval_dataset=full_test_data, metric="f1")
print("kenhktsui/setfit_test_toxic_chat_syn", trainer.evaluate(full_test_data))
model = SetFitModel.from_pretrained(f"kenhktsui/setfit_test_toxic_chat")
trainer = SetFitTrainer(model=model, args=args, eval_dataset=full_test_data, metric="f1")
print("kenhktsui/setfit_test_toxic_chat", trainer.evaluate(full_test_data))

###  fancyzhx/ag_news
dataset = load_dataset("ag_news")
full_test_data = dataset["test"]
model = SetFitModel.from_pretrained(f"kenhktsui/setfit_test_ag_news_syn")
trainer = SetFitTrainer(model=model, args=args, eval_dataset=full_test_data, metric="accuracy")
print("kenhktsui/setfit_test_ag_news_syn", trainer.evaluate(full_test_data))
model = SetFitModel.from_pretrained(f"kenhktsui/setfit_test_ag_news")
trainer = SetFitTrainer(model=model, args=args, eval_dataset=full_test_data, metric="accuracy")
print("kenhktsui/setfit_test_ag_news", trainer.evaluate(full_test_data))

### We also reproduced f1 as we are not sure the definition of reported f1, as it can be micro, macro, or weighted.
### replicate toxicchat-t5-large-v1.0 f1
import evaluate
from transformers import AutoModelForSeq2SeqLM, AutoTokenizer

data = load_dataset("lmsys/toxic-chat", "toxicchat0124", split="test")

checkpoint = "lmsys/toxicchat-t5-large-v1.0"
device = "mps"
tokenizer = AutoTokenizer.from_pretrained("t5-large")
model = AutoModelForSeq2SeqLM.from_pretrained(checkpoint).to(device)

def predict_toxic(text):
    prefix = "ToxicChat: "
    inputs = tokenizer.encode(prefix + text, return_tensors="pt").to(device)
    outputs = model.generate(inputs)
    output = tokenizer.decode(outputs[0], skip_special_tokens=True)
    if output == 'positive':
        return 1
    elif output == 'negative':
        return 0
    return None


data = data.map(lambda x: {"toxicity_pred": predict_toxic(x["user_input"])})
metric = evaluate.load("f1")
print(metric.compute(predictions=data["toxicity_pred"], references=data["toxicity"]))
print(metric.compute(predictions=data["toxicity_pred"], references=data["toxicity"], average="macro"))
print(metric.compute(predictions=data["toxicity_pred"], references=data["toxicity"], average="micro"))

### replicate nickmuchi/finbert-tone-finetuned-fintwitter-classification f1
import torch
from transformers import AutoTokenizer, AutoModelForSequenceClassification

tokenizer = AutoTokenizer.from_pretrained("nickmuchi/finbert-tone-finetuned-fintwitter-classification")
model = AutoModelForSequenceClassification.from_pretrained("nickmuchi/finbert-tone-finetuned-fintwitter-classification")

def predict_financial_sentiment(text):
    with torch.no_grad():
        output = model(**tokenizer(text, return_tensors='pt')).logits.argmax().item()
    return output

data = load_dataset("zeroshot/twitter-financial-news-sentiment", split="validation")
data = data.map(lambda x: {"label_pred": predict_financial_sentiment(x["text"])})
metric = evaluate.load("f1")
print(metric.compute(predictions=data["label_pred"], references=data["label"], average="macro"))
print(metric.compute(predictions=data["label_pred"], references=data["label"], average="micro"))
