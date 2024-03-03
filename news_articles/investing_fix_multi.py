import pandas as pd
# Load the CSV file
print('loading file')
df = pd.read_csv('investing_articles.csv')
print('Dropping duplicates...')
df = df.drop_duplicates(subset='url', keep='first')
print('processing text')
# Replace newlines in the 'article_text' column with spaces
df['article_text'] = df['article_text'].str.replace('\n', ' ', regex=False)
print('saving file')
# Save the modified DataFrame back to a CSV file
df.to_csv('investing_articles_cleaned.csv', index=False)