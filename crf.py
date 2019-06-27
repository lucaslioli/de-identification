import sys                                              # Recive params from line command
import glob                                             # Read files from folder
import string                                           # Remove the ponctuation from the texts
import pycrfsuite                                       # Python implementation of Conditional Random Fields
import progressbar                                      # Print progress bar
import numpy as np                                      # Used in the evaluation process
from pprint import pprint                               # Data pretty printer
from bs4 import BeautifulSoup as bs                     # Parse XML files
from nltk.corpus import stopwords                       # Methods to remove stop words
from nltk import pos_tag                                # Methods generating Part-of-Speech tags
from sklearn.model_selection import train_test_split    # Divide train and test set
from sklearn.metrics import classification_report       # Used to evaluate the results
from sklearn.metrics import confusion_matrix            # Print confusion matrix

FOLDER = "MEDDOCAN"
TRAIN = "MEDDOCAN/train-set/xml"
TEST = "MEDDOCAN/test-set/xml"
MODEL = 'crf.model'

# A function to prepare the data to fit with the need format
def prepare_data(path = FOLDER):
    print("\nPreparing data from {}/ ...".format(path))
    progressbar.streams.wrap_stderr()

    # Get all XML files from a folder
    files = glob.glob("{}/**/*.xml".format(path), recursive=True)

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

# A function to preprocess and make a base clean of text
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

# A function to generate the Part-of-Speech tags for each document
def pos_tagging(docs):
    print("\nGenerating Part-of-Speech tags...")

    data = []
    for doc in progressbar.progressbar(docs):

        # Obtain the list of tokens in the document
        tokens = [t for t, label in doc]

        # Perform POS tagging
        tagged = pos_tag(tokens)

        # Take the word, POS tag, and its label
        data.append([(w, pos, label) for (w, label), (word, pos) in zip(doc, tagged)])
    return data

# A function generating the features
def word2features(doc, i):
    word = doc[i][0]
    postag = doc[i][1]

    # Common features for all words
    features = [
        'bias',
        'word.lower=' + word.lower(),
        'word[-3:]=' + word[-3:],
        'word[-2:]=' + word[-2:],
        'word.isupper=%s' % word.isupper(),
        'word.istitle=%s' % word.istitle(),
        'word.isdigit=%s' % word.isdigit(),
        'postag=' + postag
    ]

    # Features for words that are not at the beginning of a document
    if i > 0:
        word1 = doc[i-1][0]
        postag1 = doc[i-1][1]
        features.extend([
            '-1:word.lower=' + word1.lower(),
            '-1:word.istitle=%s' % word1.istitle(),
            '-1:word.isupper=%s' % word1.isupper(),
            '-1:word.isdigit=%s' % word1.isdigit(),
            '-1:postag=' + postag1
        ])
    else:
        # Indicate that it is the 'beginning of a document'
        features.append('BOS')

    # Features for words that are not at the end of a document
    if i < len(doc)-1:
        word1 = doc[i+1][0]
        postag1 = doc[i+1][1]
        features.extend([
            '+1:word.lower=' + word1.lower(),
            '+1:word.istitle=%s' % word1.istitle(),
            '+1:word.isupper=%s' % word1.isupper(),
            '+1:word.isdigit=%s' % word1.isdigit(),
            '+1:postag=' + postag1
        ])
    else:
        # Indicate that it is the 'end of a document'
        features.append('EOS')

    return features

# A function for extracting features in documents
def extract_features(doc):
    return [word2features(doc, i) for i in range(len(doc))]

# A function for generating the list of labels for each document
def get_labels(doc):
    return [label for (token, postag, label) in doc]

# A function to group the process of extract features and get the labels
def features_and_labels(pos_tag_data):
    print("\nPreparing data for training and testing...")
    features = [extract_features(doc) for doc in progressbar.progressbar(pos_tag_data)]

    labels = [get_labels(doc) for doc in pos_tag_data]

    # Return as x_train, x_test, y_train, y_test
    return train_test_split(features, labels, test_size=0.2)

# Training CRF model
def crf_model_training(x_train, y_train, print_verbose = False):
    trainer = pycrfsuite.Trainer(verbose=print_verbose)

    print("\nTraining model...")

    bar = progressbar.ProgressBar(max_value=progressbar.UnknownLength)
    i = 1

    # Submit training data to the trainer
    for xseq, yseq in zip(x_train, y_train):
        trainer.append(xseq, yseq)
        bar.update(i)
        i+=1

    # Set the parameters of the model
    trainer.set_params({
        # coefficient for L1 penalty
        'c1': 0.1,

        # coefficient for L2 penalty
        'c2': 0.01,  

        # maximum number of iterations
        'max_iterations': 200,

        # whether to include transitions that
        # are possible, but not observed
        'feature.possible_transitions': True
    })

    # Provide a file name as a parameter to the train function, such that
    # the model will be saved to the file when training is finished
    trainer.train(MODEL)

# Evaluate the results from model from predictions
def results_evaluation(y_test, y_pred, print_check = False):

    if(print_check):
        print("\nChecking results...")

        # Let's take a look at a random sample in the testing set
        i = 12
        for x, y in zip(y_pred[i], [x[1].split("=")[1] for x in x_test[i]]):
            print("%s (%s)" % (y, x))

    print("\n\nResults Evaluation:")

    # Create a mapping of labels to indices
    labels = {"PHI": 1, "NOT": 0}

    # Convert the sequences of tags into a 1-dimensional array
    predictions = np.array([labels[tag] for row in y_pred for tag in row])
    truths = np.array([labels[tag] for row in y_test for tag in row])

    # Print out the classification report
    print(classification_report(
        truths, predictions,
        target_names=["NOT", "PHI"]))

    print("\nConfusion Matrix:")
    
    print(confusion_matrix(truths, predictions))

if __name__ == '__main__':
    args = {}
    args['print_verbose'] = len(sys.argv) >= 3 if True else False
    args['print_check'] = len(sys.argv) >= 2 if True else False

    docs = prepare_data(FOLDER)
    
    pos_tag_data = pos_tagging(docs)

    x_train, x_test, y_train, y_test = features_and_labels(pos_tag_data)

    crf_model_training(x_train, y_train, args['print_verbose'])

    tagger = pycrfsuite.Tagger()
    tagger.open(MODEL)
    y_pred = [tagger.tag(xseq) for xseq in x_test]

    results_evaluation(y_test, y_pred, args['print_check'])