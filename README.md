# Code Samples

Hello!

There are two code samples in this repository! 

These scripts are for fine-tuning a BERT model for multilabel classification using a source dataset and adapting it to a target dataset. 

One main distinction is that the one_vs_all classifer only learns one class at a time and requires that we train N classifiers for N labels. 

Transfer Learning / Self Training: 

Our source dataset (GQA) is a labelled dataset. Therefore, we can use the question text and knowledge gap labels to fine-tune the model. To adapt the model for the target dataset (which has no labels), we take the following approach. After N steps of training, predict labels for the target dataset. For all examples, were we can predict label(s) higher than some threshold, T, adapt these as a part of the training process.  

This is a type of transfer learning!

Happy to chat more! Feel free to open an issue, if you don't have my email!

Happy Coding!
