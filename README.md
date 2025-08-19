# Scanning Latin poetry with machine learning
![alt text](https://github.com/Ycreak/Latin_scansion_with_neural_networks/blob/main/docs/banner.png "Ancient Tetris professional")

## About
Welcome to the project for scanning Latin poetry with machine learning. The goal of this project is to train machine learning models on dactylic and iambic meters and test the generalisability to other meters such as the anapest and glyconeus. 

The project consists of two parts: 

1. Datalake to create a dataset from multiple sources (Pedecerto, Anceps, Hypotactic) on which to train our model on. 
2. The actual neural network that will train and test on the created dataset.

For more information about this project, see the [LUCAS website](https://www.universiteitleiden.nl/en/humanities/centre-for-digital-humanities/projects/small-grant-projects#developing-an-intelligent-metrical-analysis-tool-for-latin-poetry) and my [thesis](https://theses.liacs.nl/pdf/2021-2022-NoldenL.pdf) about the project.

### Table of Contents  
+ [Requirements](#Requirements)  
+ [Datalake](#Datalake)  
+ [Long short-term memory](#LSTM)  

<a name="Requirements"/>

## Requirements
The programs are written entirely in Python. Its dependencies can be found in requirements.txt. As always, you can install all dependencies in your Python environment via pip using the following command:

```console 
pip install -r requirements.txt
```

<a name="Datalake"/>

## Datalake
The datalake follows the medallion architecture, where the landing zone will retrieve material from the sources. The raw layer will import the source data as is, but will convert it to json format. The clean layer will clean the data, after which the enriched layer will harmonize the multiple sources into one uniform dataframe. The curated layer is currently not implemented, as the only client is the LSTM model, but this might change in future.

The whole idea of the datalake is to quickly harmonize datasources and for each line of poetry, provide the syllables, words and characters that make up the line, as well as the label (_long_, _short_, or _elision_) for each syllable.

<a name="LSTM"/>
## LSTM
Currently the only neural network implemented is the LSTM, as it seemed to work best. We designed a couple of experiments which can be run. The model itself is situated in its own file and can be tweaked.
