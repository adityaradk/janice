![](https://raw.githubusercontent.com/adityaradk/janice/master/logo.png)
___


Janice is **j**ust **an**other **i**ntelligent **c**lassifier for **e**xoplanets. It was created as a final project for a CMU internship. Created by [Aditya](https://github.com/adityaradk), [Shrinidhi](https://github.com/SrinidhiSunkara2000), and Veeralakshmi.

## Getting Started

### What does Janice do?

Janice was created to identify [exoplanets](https://en.wikipedia.org/wiki/Exoplanet) from [transit photometry](https://en.wikipedia.org/wiki/Methods_of_detecting_exoplanets#Transit_photometry) light curves using [machine learning](https://en.wikipedia.org/wiki/Machine_learning) techniques. Specifically, we apply neural networks (an [MLP](https://en.wikipedia.org/wiki/Multilayer_perceptron) and a [CNN](https://en.wikipedia.org/wiki/Convolutional_neural_network)), a [Naive Bayes](https://en.wikipedia.org/wiki/Naive_Bayes_classifier) classifier, a [random forest](https://en.wikipedia.org/wiki/Random_forest) classifier, an [AdaBoost](https://en.wikipedia.org/wiki/AdaBoost) classifier, a [K-Nearest Neigbors](https://en.wikipedia.org/wiki/K-nearest_neighbors_algorithm) classifier, and a [Support Vector Machine](https://en.wikipedia.org/wiki/Support_vector_machine) (SVM), as well as [ensembles](https://en.wikipedia.org/wiki/Ensemble_learning) of these models.

To train these models, we used the "Exoplanet Hunting in Deep Space" dataset, hosted on [Kaggle](https://www.kaggle.com/keplersmachines/kepler-labelled-time-series-data). It was created by GitHub user [WΔ](https://github.com/winterdelta). This dataset is a selection of denoised observations captured by the [Kepler](https://en.wikipedia.org/wiki/Kepler_space_telescope) observatory, hosted on the [Mikulski Archive](https://archive.stsci.edu/). 

### Setup

After creating a local copy of the repo, install the dependencies:

```
pip install -r requirements.txt
```

Note that the code is written in Python 3 and the commands (`pip3` and `python3`) may differ depending on your installation.

### Recommended Usage

It is recommended that the python files are run from the parent directory using the `-m` tag.

For example, to train the CNN, run:

    python -m models.cnn.train

Most of the python files also make use of an argument parser. For usage info, use the `-h` tag.

## Further Information

For more details, please see the [report](https://raw.githubusercontent.com/adityaradk/janice/master/report.pdf) or view the READMEs in the folders.
