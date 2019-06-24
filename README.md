# De-identification Experiments

Experiments using CRF in Python for de-identification of clinical records. The dataset used, as well as the gazetteers, has been disclosed by [MEDDOCAN](http://temu.bsc.es/meddocan/) (Medical Document Anonymization task).

To replicate the experiments, download the clinical records from [MEDDOCAN datasets](http://temu.bsc.es/meddocan/index.php/data/) and paste it to the folder ```MEDDOCAN/```. Also, download the gazetteers from [MEDDOCAN resources](http://temu.bsc.es/meddocan/index.php/resources/) and paste it to the folder ```gazetteer/```, both inside the repository folder.

The initial experiments using CRF in Python were made based on Albert Au Yeung's article, "[Performing Sequence Labelling using CRF in Python](http://www.albertauyeung.com/post/python-sequence-labelling-with-crf/)".

## Requirements

To run the scripts it's needed to install the dependencies, described in file ```requirements.txt```.

    $ sudo pip install -r requirements.txt

After install all requirements it's also nedeed to download the following packages from [NLTK](https://www.nltk.org/).

    $ python3
    >>> import nltk
    >>> nltk.download('averaged_perceptron_tagger')
    >>> nltk.download('stopwords')
