"""Main script for news project. """
import pandas as pd
from pathlib import Path

# Define paths
PROJECT_DIR = Path(__file__).parent
DATA_DIR = Path(__file__).parent.parent / "data"

# Load data
news_df = pd.read_csv(DATA_DIR / "raw" / "news_dataset.csv")

# Inspect data
print(news_df.head())
print(news_df.tail())
print(news_df.info())
print(news_df.describe())
print("\n")

# How many articles are real and how many are fake?
real_count = news_df.loc[news_df["label"] == 1].shape[0]
fake_count = news_df.loc[news_df["label"] == 0].shape[0]

print(f"Real articles: {real_count}")
print(f"Fake articles: {fake_count}")

# How many different authors are there?
num_authors = news_df["author"].nunique()
print(f"Number of authors: {num_authors}")


# Authors with the most fake articles
def get_author_counts(df):
    """Get counts of articles per author."""
    author_counts = df.groupby(by="author", as_index=False)["text"].count()
    author_counts = author_counts.rename(columns={"text": "article_count"})
    author_counts = author_counts.sort_values("article_count", ascending=False)
    author_counts = author_counts.reset_index(drop=True)
    return author_counts


fake_news = news_df.loc[news_df["label"] == 0, :]
real_news = news_df.loc[news_df["label"] == 1, :]
fake_authors = get_author_counts(fake_news)
real_authors = get_author_counts(real_news)

print("Authors with the most fake articles:")
print(fake_authors.head(20))
print("Authors with the most real articles:")
print(real_authors.head(20))


# Most common words in fake articles
def get_word_count(df, column_name):
    """Get word count for a column of text."""
    list_list_words = df[column_name].str.split(" ").to_list()
    # flatten list of lists
    list_words = []
    for sublist in list_list_words:
        try:
            for word in sublist:
                list_words.append(word)
        except TypeError:
            continue
    word_count = pd.DataFrame({"word": list_words, "count": [1] * len(list_words)})
    word_count = word_count.groupby("word", as_index=False).count()
    word_count = word_count.sort_values("count", ascending=False).reset_index(drop=True)
    return word_count


# count most common words in fake articles
fake_word_count = get_word_count(fake_news, "text")
# count most common words in real articles
real_word_count = get_word_count(real_news, "text")

print("Most common words in fake articles:")
print(fake_word_count.head(20))

print("Most common words in real articles:")
print(real_word_count.head(20))

# Most common words in fake articles that are not in real articles
# keep only first words ("most common") and drop duplicates
fake_word_count = fake_word_count.iloc[:500, :]
real_word_count = real_word_count.iloc[:500, :]
fake_word_count = fake_word_count.merge(
    real_word_count, on="word", how="left", suffixes=("_fake", "_real")
)
fake_word_count = fake_word_count.loc[
    fake_word_count["count_real"].isnull(), :
].reset_index(drop=True)
print("Most common words in fake articles that are not 'much' in real articles:")
print(fake_word_count.head(50))

# The rest is left as an exercise for the reader :)
