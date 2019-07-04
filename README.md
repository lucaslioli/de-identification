# De-identification Experiments

Experiments using CRF in Python for de-identification of clinical records. The dataset used, as well as the gazetteers, has been disclosed by [MEDDOCAN](http://temu.bsc.es/meddocan/) (Medical Document Anonymization task).

To replicate the experiments, download the clinical records from [MEDDOCAN datasets](http://temu.bsc.es/meddocan/index.php/data/) and paste it to the folder ```MEDDOCAN/```. Also, download the gazetteers from [MEDDOCAN resources](http://temu.bsc.es/meddocan/index.php/resources/) and paste it to the folder ```gazetteer/```, both inside the repository folder.

The initial experiments using CRF in Python were made based on Albert Au Yeung's article, "[Performing Sequence Labelling using CRF in Python](http://www.albertauyeung.com/post/python-sequence-labelling-with-crf/)". Currently, many modifications have been done, such as the POS tagger used, CRF model used and the code organization.

## Requirements

To run the scripts it's needed to install the dependencies, described in file ```requirements.txt```.

    $ sudo pip install -r requirements.txt

After install all requirements it's also nedeed to download the following packages from [NLTK](https://www.nltk.org/).

    $ python3
    >>> import nltk
    >>> nltk.download('averaged_perceptron_tagger')
    >>> nltk.download('stopwords')

### Part-of-Speech Tagger

As part of the requirements, it's needed to download the respectively POS tagger from [Stanford NLP](https://nlp.stanford.edu/software/tagger.html) and extract the content into any folder. The experiments using MEDDOCAN dataset have been done using the full version 3.8.0, described as ```new Spanish and French UD models```.

## Settings

To increase the number of experiments and compare the results using other datasets, some params were defined to allow an easy change of the dataset used. The file ```config.py``` can be ajusted as your needs, as the following settings.

* ```LANGUAGE``` - Dataset language
* ```DATA_PATH``` - Path to folder were the dataset is located
* ```STANFORD_PATH``` - Path to the folder were Stanford NLP POS Tagger is located
* ```POS_TAGGER``` - The suitable tagger for dataset language (located inside ```models``` folder) 
* ```CRF_MODEL``` - The name that will be used to save the CRF model.

## Run

To run the algorithm, use the following command. The option ```check``` is used to indicate the nedeed to print part of the labels tagged in the end. The option ```verbose``` indicate the needed to print the verbose during the training.

    $ python3 crf.py <dataset> [-check] [-verbose]