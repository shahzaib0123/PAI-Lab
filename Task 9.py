from nltk.sentiment.vader import SentimentIntensityAnalyzer
import nltk

nltk.download('vader_lexicon')

sia = SentimentIntensityAnalyzer()

sentences = [
    "I love this phone, it is amazing!",
    "This is the worst movie I have ever seen.",
    "The food was okay, nothing special.",
    "Absolutely fantastic experience, highly recommend!",
    "I hate waiting in long lines.",
    "The weather is fine today.",
    "This laptop is great but the battery is bad."
]

print("Sentiment Analysis using VADER")
print("-" * 50)

for sentence in sentences:
    scores = sia.polarity_scores(sentence)

    if scores['compound'] >= 0.05:
        sentiment = "Positive"
    elif scores['compound'] <= -0.05:
        sentiment = "Negative"
    else:
        sentiment = "Neutral"

    print(f"Text      : {sentence}")
    print(f"Scores    : {scores}")
    print(f"Sentiment : {sentiment}")
    print("-" * 50)
