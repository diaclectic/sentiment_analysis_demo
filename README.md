
# Sentiment Analysis Demo

This demo is of a recurrent neural net (RNN) that uses Word2Vec vectors in order to classify text as either positive or negative sentiment. After training, the model is demonstrated using Hillary Clinton's email dataset. A widget is also provided to enter arbitrary text.

## Prequisites
Java version 1.8.0_171 (x64 - model loads a massive ~6 Gb set of Word2Vec vectors into memory)
5.4 GB disk space in your primary drive (saves uncompressed resources and trained model weights to speed up widget usage)

## About the model
Pretrained vectors for Word2Vec used are from the Google News word vectors (released under Apache License 2.0 and retreived from here: https://code.google.com/p/word2vec/ )

Training and test datasets used are from the Stanford large movie review dataset of IMDB reviews (no license provided, but widely used and the only requirement specified is for citing Stanford when publishing papers; retrieved from here: http://ai.stanford.edu/~amaas/data/sentiment/ ). While not terribly similar to email, this should contain more than sufficient data to determine whether particular words tend to be used in positive or negative sentences/documents.

Demo of the model is done using Hillary Clinton's email dataset, retrieved from here: https://www.kaggle.com/kaggle/hillary-clinton-emails

Data is copied to a temp folder and extracted there. Once model is trained the first time, it also is saved in the temp folder. Subsequent runs will skip extracting the data and training the model if these files are present already.

## Disclaimer
1. The last layer of the model uses a softmax activation function to provide a clean distinction between the binary classification (positive vs negative sentiment), but at the expense of the probabilities of each. This results in the output indicating a rather high number in either the positive or negative estimate, and a rather low number for the other. This is NOT a probability estimate, as evidence by the stark contrast between relatively similar inputs. (i.e. 'This is okay.' vs 'This is not okay.')
2. Model is not yet well tuned, but should be sufficient to demonstrate use case.

## Building the demo
Package was built with Eclipse with the VM arguments set to "-Xms5120m -Xmx6144m", but just uses mvn package argument under the hood:
```bash
# export MAVEN_OPTS="-Xms5120m -Xmx6144m"
# Windows: Win+Pause ; Add MAVEN_OPTS environment variable for user ; set this to "-Xms5120m -Xmx6144m" (without quotes)
mvn clean dependency:copy-dependencies package
```

## Running the demo
First, you need to train the model. This is done in one class that copies the source files to a temp directory, extracts them there, trains the model, then saves the model in the temp directory. All subsequent runs load the extracted resources and trained model to save time.

NOTE: After training, this will iterate over Hillary Clinton's email dataset, printing the sentiment of each message. This is for demonstration purposes only, and the process can be killed at this point with no consequences.
```bash
cd target
java -Xms5120m -Xmx6144m -cp sentiment_analysis_demo-0.0.1-SNAPSHOT-bin.jar com.globalrelay.nlp.sentiment_analysis_demo.Word2VecSentimentRNN
```

Then, you can run the demo widget. This also loads the extracted resources and trained model, then opens a window to demonstrate the model output.
```bash
cd target
java -Xms5120m -Xmx6144m -cp sentiment_analysis_demo-0.0.1-SNAPSHOT-bin.jar com.globalrelay.nlp.sentiment_analysis_demo.SentimentAnalysisDemo
```
