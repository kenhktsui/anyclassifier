# Synthetic Data Generation For Text Classification
## Motivation
Synthetic data becomes more and more common because it can minimize data drift between training data and production data. It is also common when pretraining LLM in order to boost LLM performance.
It is required because some training data is very difficult or costly to collect.
In this article, we present an approach for text classification.

## Design
We adopt a hierarchical generation approach by asking LLM to suggest topic to expand on a particular label on first level, and suggest subtopic on second level,
It is also inspired by [1], where multiple personas are simulated to cover diverse scenario. We adopted source-type-driven strategy instead of persona-driven strategy as source type usually reflects a different groups of target audiences, and it is less fine-grained, which is more suitable for text classification model which is less data
hungry in training.  
Each data synthesis is grounded on unique combination of (instruction, label, subtopic, source_type).
By enumerating on all these combination, it ensures the diversity of synthetic data by design.

![image](../assets/SyntheticDataGeneratorForSequenceClassification.png)

## Prompt
### Prompt to Expand Source Type
```txt
I am building a document classifier to {{instruction}} with labels {{labels}}. Suggest {{n}} source type of information for efficient data acquisition.
Output JSON array. Each item contains key "source_type".
```
### Prompt to Expand Topic from Label
```txt
I am building a document classifier to {{instruction}} with labels {{labels}}. I would like to collect collectively exhaustive taxonomy or topic for the label: {{label}} from {{source_type}}.

<instruction>
- Suggest {{n}} taxonomies or topics to further expand on this label.
- Output JSON array. Each item contains key "item" 
</instruction>```

## Example
![image](../assets/Sentiment_Analysis_Data_Generation.png)
```
### Prompt to Expand Subtopic from Topic
```txt
I am building document classifier to {{instruction}} with labels {{labels}}.  I would like to collect collectively exhaustive subtopic under {{topic}} from {{source_type}}.

<instruction>
- Suggest {{n}} subtopic or keywords. 
- Output JSON array. Each item contains key "item" 
</instruction>
```

### Prompt to Generate 
```txt
I am building document classifier to {{instruction}} with labels {{labels}}. I would like to collect samples for the label: {{label}}.

<instruction>
- Generate realistic examples for a classification model that will predict label {{label}}.
- Characteristics:
  Topic: {{topic}}.
  Source type: {{source_type}}
- Generate {{n}} example.
- The example shall have a realistic length, and cannot be too short.
- Output JSON array. Each item contains key "text" 
</instruction>
```

## Example
### imdb
| Label              | Examples                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                          |
|:-------------------|:--------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------|
| negative sentiment | <ul><li>'I was really looking forward to this movie, but unfortunately, it was a complete disappointment. The plot was predictable and lacked any real tension. The characters were underdeveloped and their motivations were unclear. The pacing was slow and dragged on for far too long. Overall, I would not recommend this movie to anyone.'</li><li>"I'm extremely disappointed with the service I received at this restaurant. The hostess was unfriendly and unhelpful, and our server seemed completely overwhelmed. We had to ask multiple times for basic things like water and utensils. The food was overpriced and not even that good. Definitely will not be returning."</li><li>"I'm extremely disappointed with my recent purchase from this store. The quality of the product is subpar and the price is way too high. I paid $200 for a cheap-looking item that broke after just a week of use. Not worth the money at all. 1/10 would not recommend."</li></ul>               |
| positive sentiment | <ul><li>"I just got tickets to see my favorite artist in concert and I'm beyond thrilled! The energy in the crowd is going to be electric! #concertseason #musiclover"</li><li>"I just had the most amazing experience at this restaurant! The service was lightning fast, and the food was prepared to perfection. Our server, Alex, was attentive and friendly, making sure we had everything we needed. The bill was reasonable, and we left feeling satisfied and eager to come back. 5 stars isn't enough, I'd give it 10 if I could!"</li><li>'The action scenes in this movie are absolutely mind-blowing! The stunts are incredibly well-choreographed and the special effects are top-notch. I was on the edge of my seat the entire time, cheering on the heroes as they fought to save the world. The cast is also excellent, with standout performances from the lead actors. Overall, I would highly recommend this movie to anyone who loves action-packed thrill rides.'</li></ul> |
[Click here to inspect more.](https://huggingface.co/datasets/kenhktsui/test_syn)


### zeroshot/twitter-financial-news-sentiment
| Label   | Examples                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                     |
|:--------|:---------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------|
| Bearish | <ul><li>"I'm getting out of the market before it's too late. The Dow is plummeting and I don't see any signs of recovery. The economic indicators are all pointing to a recession and I'm not willing to risk my retirement savings. I've been in this market for years, but I think it's time to cut my losses and move to cash. Has anyone else seen this coming?"</li><li>"Bloomberg: Tesla's Earnings Miss Estimates as Revenue Declines 20% - The electric vehicle maker's quarterly earnings fell short of analysts' expectations, with revenue plummeting 20% to $24.58 billion. The company's gross margin also declined, sparking concerns about its ability to maintain profitability. Tesla's stock price tumbled 8% in after-hours trading, as investors reacted to the disappointing results. The miss is a setback for CEO Elon Musk, who has been under pressure to deliver consistent growth. Analysts had expected Tesla to report earnings of $0.55 per share, but the company came in at $0.35 per share. The revenue decline was driven by a 15% drop in automotive sales, as well as a 10% decline in energy generation and storage sales. The company's guidance for the current quarter also fell short of expectations, with Tesla forecasting revenue of $24 billion, below the consensus estimate of $25.5 billion. The earnings miss is a reminder that the electric vehicle market remains highly competitive, and that Tesla faces significant challenges in maintaining its market share."</li><li>"I'm getting out of the market before it's too late. The Dow just plummeted 500 points and I'm not willing to risk losing my shirt. The economic indicators are all pointing to a recession and I'm not convinced that the Fed can stem the tide. Anyone else bailing out before the ship sinks?"</li></ul> |
| Neutral | <ul><li>"Earnings Report: Tech Giant's Revenue Beats Expectations, But Margins Narrow. Despite a 10% increase in revenue, the company's net income fell short of forecasts due to higher operating expenses. The stock price remained relatively stable, with investors taking a cautious approach ahead of the company's upcoming product launch."</li><li>'Just got back from a meeting with my financial advisor and we discussed the latest quarterly earnings reports. No major surprises, just a steady performance from the market. Nothing to get too excited about, but nothing to worry about either. #stockmarket #investing'</li><li>'The proposed merger between Coca-Cola and Coca-Cola European Partners is expected to create a beverage giant with a combined market value of over $100 billion. While the deal is still pending regulatory approval, analysts are cautiously optimistic about its potential to drive growth and increase efficiency. The merged entity is likely to benefit from economies of scale and a stronger presence in key markets, but some investors are concerned about the potential impact on jobs and supply chains. As the deal moves forward, investors will be closely watching for any signs of regulatory hurdles or other issues that could delay or derail the merger.'</li></ul>                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                     |
| Bullish | <ul><li>"Just got my hands on the new #Robinhood app update! They've finally added a feature to track my portfolio's performance in real-time. Huge shoutout to the @RobinhoodApp team for listening to user feedback and delivering on their promises! #investing #stockmarket #bullish"</li><li>"Just got out of the @AAPL earnings call and I'm feeling super bullish about the company's future. The guidance on EPS is looking strong and I think we're on the verge of a major breakout. #AAPL #StockMarket #Earnings"</li><li>"Just heard that the Federal Reserve is considering cutting interest rates to stimulate the economy. This is a huge positive for the stock market and I'm expecting a strong rally in the coming weeks. Anyone else feeling bullish about the market right now?"</li></ul>                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                              |
[Click here to inspect more.](https://huggingface.co/datasets/kenhktsui/test_twitter_financial_news_syn)

## Usage
You can either specify `n_record_to_generate` or all of the params `n_source_type`, `n_topic`, `n_subtopic`, `sample_per_subtopic`.
If first is used, our implementation aims at generate such no of record, but it might not be exact because of unpredictability of LLM.
If later is used, the result no of record = n_source_type * n_topic * n_subtopic * sample_per_subtopic. It may be required for some problems, for example, where you see more source_type and less in topic.

## Benchmarking
We tested on two datasets - imdb and zeroshot/twitter-financial-news-sentiment. 
Model performance of synthetic data is at par/ marginal lower than that of real data, which is not bad because the testing data is usually morel similar to training (real) data than synthetic data. See [Benchmark](../README.md#benchmark).
We will continue to add more benchmark on other datasets.

## Reference
1. Chan, X., Wang, X., Yu, D., Mi, H., Yu, D., 2024. Scaling synthetic data creation with 1,000,000,000 personas. URL: https://arxiv.org/abs/2406.20094, arXiv:2406.20094.