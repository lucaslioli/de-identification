from nltk.corpus import stopwords                       # Methods to remove stop words

# A function to preprocess and make a base clean of text
def text_cleaner(text, language):
    # Getting the set of stop words
    stop = set(stopwords.words(language))

    # Getting the set of punctuation
    pont={'.', ',', '_', '^', '*'}

    # To remove the stop words from the text
    text = ' '.join([w for w in text.split() if w not in stop])

    # Remove the (setted) ponctuation from the text
    text = ''.join(c for c in text if c not in pont)

    # Removes the line breaks
    text = text.replace('\n', ' ').replace('\r', '')

    text = text.lstrip().rstrip().replace(u'\ufeff', '')

    return text