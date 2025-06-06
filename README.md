# Troy Tech Internship

## Project Overview

### Details

- Authors: Jaden Lee
- Course: Troy Tech Internship
- Advisors: Rich Liem, Tushar Puri
- Created Date: December 13, 2024
- Repository Link: https://github.com/lee1jaden/troy-tech-internship
- Copyright: This code is the property of its author and may not be copied or edited for any use without the express written consent of its owner.

### Description

This project was completed from June 1, 2021 to July 2, 2021 to fulfill the 150 hour requirement of the Troy Tech Internship. The ideas were provided by Tushar Puri, the CEO of Pegasus One Software based in Fullerton, CA. Most work was done independently and has been uploaded to GitHub with improvements. The program itself implements a machine learning model to identify handwritten characters from the MNIST Database and progresses into attempting to identify text in more natural environments such as address numbers on the siding of a house and entire handwritten words.

## Running the Project

### Installation

The setup instructions assume the code can be run from the dev container described in the repository. This can be done using Docker Desktop and VS Code with the _Dev Containers_ extension. This will build a container with the necessary dependencies to begin the project.

Add a Python virtual environment at the top level of the project and install the dependencies listed in the _requirements.txt_ file.

### Instructions

All datasets are expected to be placed in the top-level _/data_ folder without renaming from the downloaded versions. All commands should be run from the top-level directory using Make.

1. Handwritten Letter Recognition: The images are 28x28 pixels and represented as flattened grayscale values in a CSV dataset. [Download here](https://www.kaggle.com/datasets/sachinpatel21/az-handwritten-alphabets-in-csv-format). Open a terminal and run `make build DATA=letters` to build a model that can predict handwritten letters at 96% accuracy. It may take a few minutes. Conversely, create a convolutional model with command `make build DATA=letters-cnn` that reaches up to 98.5% accuracy.
1. Handwritten Digit Recognition: The images adhere to the same convention as the handwritten character recognition images and are downloaded from a dataset included in the Keras library. To train a neural network to identify them, run `make build DATA=digits`.
1. Handwritten Character Recognition: The letter and digit datasets are concatenated to form one handwritten character set. Running `make build DATA=chars` will produce a model that can accurately distinguish a given character 97% of the time.
1. Handwritten Word Recognition: (not completed)
1. House Number Recognition: This model predicts 32x32 colored images from two .mat files. [Download 'Format 2' from here without the extra files](http://ufldl.stanford.edu/housenumbers/). From the console, run `make build DATA=houses` to train a model that can predict the house numbers at close to an 85% clip.

After building each model, see the console for the results of running the model against a random testing dataset.

To visualize the results on a random sample, run a specific Make command and view the produced image file in the _/examples_ folder. The associated model will automatically be trained and saved to the _/models_ folder if not previously done.

1. Handwritten Letter Recognition: `make plot DATA=letters` or `make plot DATA=letters-cnn` depending on the model you intend to use.
1. Handwritten Digit Recognition: `make plot DATA=digits`
1. Handwritten Character Recognition: `make plot DATA=chars`
1. House Number Recognition: `make plot DATA=houses`

Note: Any `make build` or `make plot` command that does not specify the DATA field will cause an error.
