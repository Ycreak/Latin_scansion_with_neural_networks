# Scanning Latin poetry with machine learning
![alt text](https://github.com/Ycreak/Latin_scansion_with_neural_networks/blob/main/docs/banner.png "Ancient Tetris professional")

## About
Welcome to the project for scanning Latin poetry with machine learning. The goal of this project is to train machine learning models on dactylic meters and test the generalisability to other meters such as the anapest and glyconeus. 

The project consists of three parts: 

1. Creating data sets for machine learning tools
2. Training and testing an CRF on Latin poetry
3. Training and testing an LSTM on Latin poetry

For more information about this project, see the [LUCAS website](https://www.universiteitleiden.nl/en/humanities/centre-for-digital-humanities/projects/small-grant-projects#developing-an-intelligent-metrical-analysis-tool-for-latin-poetry) and my [thesis](https://github.com/Ycreak/Latin_scansion_with_neural_networks/blob/main/thesis.pdf) about the project.

### Table of Contents  
+ [Requirements](#Requirements)  
+ [Dataset creation](#Dataset)  
+ [Conditional Random Fields](#CRF)  
+ [Long short-term memory](#LSTM)  

<a name="Requirements"/>

## Requirements
The programs are written entirely in Python. Its dependencies can be found in requirements.txt. As always, you can install all dependencies in your Python environment via pip using the following command:

```console 
pip install -r requirements.txt
```

<a name="Dataset"/>

## Creating a data set
Before machine learning can be trained and tested on Latin poetry, we need to create a data set that is machine readable. For this, we used a syllable-label list. Here, every syllable in a line of poetry is accompanied by its label, being either _long_, _short_, or _elision_. We represent every line with a list, with tuples for every syllable-label combination. To illustrate, the first three syllables of Vergil's _Aeneid_ would look as follows:

```python
[(ar, long), (ma, short), (vi, short)]
```

An entire text of combinations of texts would then be represented of a list of lists, with each nested list being a line of poetry. To create such a list, two methods can be employed, as seen below.

### Using Pedecerto
The [Pedecerto](https://www.pedecerto.eu/public/) project uses a rule-based approach to scan dactylic meter. We use their scansions to create our dataset. The scansions are made available in the form of XML files on [their website](https://www.pedecerto.eu/public/pagine/autori). Now do the following:

1. Download the XML files of which you want to create your dataset. 
2. Place the downloaded XML files in the **scansions/pedecerto/** folder.
3. Call the **convert_pedecerto_xml_to_syllable_sequence_files** function from the **data_creation** module. For example:

```console 
from lsnn import dataset_creation

dataset_creation.Pedecerto_parser().convert_pedecerto_xml_to_syllable_sequence_files(
    input_folder = conf.PEDECERTO_SCANSION_FOLDER,
    output_folder = conf.SEQUENCE_LABELS_FOLDER    
)
```
This will create pickle files with syllable-label lists and save them to the **scansions/sequence_labels/**.

4. (Optional) You can combine the syllable label list of multiple texts and authors by running the following code:

```console 
TODO
```

This will store the created file called **combined.pickle** in the **scansions/sequence_labels/** folder. 

**IMPORTANT** Make sure that all your pickled syllable-label lists are stored in the **scansions/sequence_labels/** folder, as this is the folder the machine learning tools will use when searching for datasets. One can edit the paths via the **config.py** file.

_Note: the tool that creates the syllable-label lists will only process hexameters and pentameters, as these are the focus of the Pedecerto project. Any corrupt lines will be discarded, as well as lines containing words that cannot be syllabified by the syllabifier provided by BNagy and his [MQDQ-parser](https://github.com/bnagy/mqdq-parser) (found under pedecerto/syllabifier.py and pedecerto/rhyme.py)._

### Using Anceps
The [Anceps]([https://www.pedecerto.eu/public/](https://github.com/Dargones/anceps)) project uses a constraint-based approach to scan iambic trimeter. We also provide tools to convert these scansions into syllable-label lists. However, these lists will contain the extra label _anceps_ as Anceps does not resolve these labels automatically.

1. Create scansion files using the Anceps tools, or download complete scansions from the [Senecan Trimeter and Humanist Tragedy repository](https://github.com/QuantitativeCriticismLab/AJP-2022-Senecan-Trimeter).
2. Put the JSON files in the **scansions/anceps/** folder and run the following command:

```console
from lsnn import dataset_creation

dataset_creation.Anceps_parser().convert_anceps_json_to_syllable_sequence_files(
    input_folder = conf.ANCEPS_SCANSION_FOLDER,
    output_folder = conf.SEQUENCE_LABELS_FOLDER
)
```

This will create a pickled syllable_label file of each JSON in the **scansions/anceps** folder. To combine these files, the combining code described in the previous section can be used.

<a name="CRF"/>

## Running the CRF Network
There are two options to run the CRF network. 

1. The first one uses k-fold cross validation and runs on a single pickled file. 
2. The second option is to train on one pickled file and test on another.

Both options can be found in the example file called **test.py**.

<a name="LSTM"/>

## Running the LSTM Network
There are, like with the CRF model, two options to run the LSTM network. 

1. The first one uses k-fold cross validation and runs on a single pickled file. 
2. The second option is to train on one pickled file and test on another.

Both options can be found in the example file called **test.py**.
