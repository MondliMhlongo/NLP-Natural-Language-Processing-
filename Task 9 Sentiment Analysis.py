import pandas as pd
import matplotlib.pyplot as plt
import nltk
from nltk.sentiment.vader import SentimentIntensityAnalyzer

nltk.download("vader_lexicon")

dfFramer = pd.read_csv("C:\\Data Set For Task\\3) Sentiment dataset.csv")
print(dfFramer.columns)
print(dfFramer.head())
# dfFramer["Reviews"] = dfFramer["Reviews"].fillna("").astype(str)

#Using the sentiment analyser to interpret the comments
sentimentAnalysis = SentimentIntensityAnalyzer()

#a function to assign scores and labels for each comment
def get_sentiment_label(text):

    scoreCommission = sentimentAnalysis.polarity_scores(text)["compound"]

    if scoreCommission >= 0.05:
        return "Positive"
    elif scoreCommission <= -0.05:
        return "Negative"
    else:
        return "Neutral"
    
# dfFramer["Compound_Score"] = dfFramer["Reviews"].apply(lambda x: sentimentAnalysis.polarity_scores(x)["compound"])
# dfFramer["Sentiment"] = dfFramer["Reviews"].apply(get_sentiment_label)

print("===Results===")
#print(dfFramer[["Reviews", "Compound_Score", "Sentiment"]])

#Visualising data
dfFramer["Sentiment"] = dfFramer["Sentiment"].str.strip()
sentiment_counter = dfFramer["Sentiment"].value_counts()

#Visualising the results through Matplotlib
plt.figure(figsize=(8, 6))

#Colour Alignment
color_map = {"Positive": "Green", "Negative": "Red", "Neutral": "Gray"}
bar_colouring = [color_map.get(sentiment.strip().title(), "gray") for sentiment in sentiment_counter.index]
print(sentiment_counter.index.tolist())
plt.bar(sentiment_counter.index, sentiment_counter.values, color=bar_colouring, edgecolor="black")

#chart formatting
plt.title("Customer Review Sentiment", fontsize=14, fontweight="bold")
plt.xlabel("Category", fontsize=13)
plt.ylabel("No of Reviews", fontsize=13)
plt.grid(axis="y", linestyle="--", alpha=0.8)

#display the plots
plt.tight_layout()
plt.show()