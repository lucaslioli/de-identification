import sys                                              # Recive params from line command
import glob                                             # Read files from folder
import pycrfsuite                                       # Python implementation of Conditional Random Fields
import progressbar                                      # Print progress bar
import numpy as np                                      # Used in the evaluation process
from bs4 import BeautifulSoup as bs                     # Parse XML files
from nltk.tag.stanford import StanfordPOSTagger         # Methods generating Part-of-Speech tags
from sklearn.model_selection import train_test_split    # Divide train and test set
from sklearn.metrics import classification_report, confusion_matrix # Evaluate the results and print the confusion matrix

from config import *
from resources.text_processing import simple_cleaner

# A function to prepare the data to fit with the need format
def prepare_data(path):
    print("\nPreparing data from {}/ ...".format(path))

    # Get all XML files from a folder
    files = glob.glob("{}/**/*.xml".format(path), recursive=True)

    docs = []

    # Process all files in the folder
    for file in progressbar.progressbar(files):
        # Read data file and parse the XML
        with open(file, mode='r', encoding='utf-8-sig') as infile:
            soup = bs(infile, "html.parser")

        # Get the pacient note text
        record = soup.find("text").text

        # Array that will save all positions to define PHI
        pos = []

        # Add the first position
        pos.append(0)

        # Build array with all positions of PHI annotated
        for phi in soup.find("tags").findAll():
            pos.append(int(phi.get('start')))
            pos.append(int(phi.get('end')))

        # Add the last position
        pos.append(len(record)-1)

        # Array that will save all the words with each label
        texts = []

        for i in range(len(pos)-1):
            # Entire phrase between the defined positions
            info = record[pos[i] : pos[i+1]]
            
            info = simple_cleaner(info)

            # Considering that the first word will never be a PHI,
            # when the position is EVEN, the label is NOT
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

# A function to generate the Part-of-Speech tags for each document
def pos_tagging(docs, stanford_path, pos_tagger):
    print("\nGenerating Part-of-Speech tags...")

    # Configuring Stanford NLP POS tagger
    path_to_model = "{}/models/{}.tagger".format(stanford_path, pos_tagger)
    path_to_jar = "{}/stanford-postagger.jar".format(stanford_path)

    tagger = StanfordPOSTagger(model_filename=path_to_model, path_to_jar=path_to_jar)
    # Setting higher memory limit for long sentences
    tagger.java_options='-mx8192m'

    data = []
    for doc in progressbar.progressbar(docs):
        # Obtain the list of tokens in the document
        tokens = [t for t, label in doc]

        try:
            # Perform POS tagging
            tagged = tagger.tag(tokens)
        except:
            continue

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
def crf_model_training(x_train, y_train, f_model = 'crf.model', print_verbose = False):
    trainer = pycrfsuite.Trainer(verbose=print_verbose)

    print("\nTraining model...")

    # Submit training data to the trainer
    for xseq, yseq in zip(x_train, y_train):
        trainer.append(xseq, yseq)

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
    trainer.train(f_model)
    print("Model Trained!\n")

# Evaluate the results from model from predictions
def results_evaluation(x_test, y_test, y_pred, print_check = False):

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
    print("\n")

if __name__ == '__main__':
    if(len(sys.argv) < 2):
        print("You need to inform the dataset.")
        exit(1)

    args = set_params(sys.argv)

    if(len(args) == 0):
        print("This is not valid dataset.")
        exit(1)
    
    progressbar.streams.wrap_stderr()

    docs = prepare_data(args['DATA_PATH'])
    
    pos_tag_data = pos_tagging(docs, args['STANFORD_PATH'], args['POS_TAGGER'])

    x_train, x_test, y_train, y_test = features_and_labels(pos_tag_data)

    crf_model_training(x_train, y_train, args['CRF_MODEL'], args['PRINT_VERBOSE'])

    tagger = pycrfsuite.Tagger()
    tagger.open(args['CRF_MODEL'])
    y_pred = [tagger.tag(xseq) for xseq in x_test]

    results_evaluation(x_test, y_test, y_pred, args['PRINT_CHECK'])