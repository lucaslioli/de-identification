import glob                             # Read files from folder
import string                           # Remove the ponctuation from the texts
from pprint import pprint               # Data pretty printer
import progressbar                      # Print progress bar
from bs4 import BeautifulSoup as bs     # Parse XML files
from nltk.corpus import stopwords       # Methods to remove stop words
from nltk import pos_tag                # Methods generating Part-of-Speech tags

def prepare_data(path):
    print("Preparing data for training...")
    progressbar.streams.wrap_stderr()

    # Get all XML files from the path
    files = glob.glob("{}/*.xml".format(path))

    docs = []

    # Process all files in the folder
    for file in progressbar.progressbar(files):
        # Read data file and parse the XML
        with open(file, mode='r', encoding='utf-8-sig') as infile:
            soup = bs(infile, "html.parser")

        # Get the main tag in the file
        elem = soup.find("meddocan")

        # Get the pacient note text
        record = elem.find("text").text

        pos = []
        pos.append(0)

        # Build array with all positions of PHI annotated
        for phi in elem.find("tags").findAll():
            pos.append(int(phi.get('start')))
            pos.append(int(phi.get('end')))

        pos.append(len(record)-1)

        texts = []

        for i in range(len(pos)-1):
            info = text_cleaner(record[pos[i] : pos[i+1]])

            if i % 2 == 0:
                label = "NOT"
            else:
                label = "PHI"

            if(len(info) == 0 and label != "PHI"):
                continue

            for w in info.split(" "):
                if len(w) > 0:
                    texts.append((w, label))

        docs.append(texts)
    return docs

def text_cleaner(text):
    # Getting the set of stop words
    stop = set(stopwords.words('spanish'))

    # Getting the set of punctuation
    pont=set(string.punctuation)

    # To remove the stop words from the text
    text = ' '.join([w for w in text.split() if w not in stop])

    # To remove the ponctuation from the text
    text = ''.join(c for c in text if c not in pont)

    text = text.lstrip().rstrip().replace(u'\ufeff', '')

    return text

def pos_tagging(docs):
    print("Generating Part-of-Speech tags...")

    data = []
    for doc in progressbar.progressbar(docs):

        # Obtain the list of tokens in the document
        tokens = [t for t, label in doc]

        # Perform POS tagging
        tagged = pos_tag(tokens)

        # Take the word, POS tag, and its label
        data.append([(w, pos, label) for (w, label), (word, pos) in zip(doc, tagged)])
    return data

if __name__ == '__main__':
    docs = prepare_data("MEDDOCAN/train-set/xml")
    data = pos_tagging(docs)
    pprint(data[0][:10])