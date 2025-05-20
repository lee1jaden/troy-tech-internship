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

1. Handwritten Letter Recognition: The images are 28x28 pixels and represented as flattened grayscale values in a CSV dataset. [Download here](https://www.kaggle.com/datasets/sachinpatel21/az-handwritten-alphabets-in-csv-format). Open a terminal and run `make letters-model` to build a model that can predict handwritten letters at 96% accuracy. It may take a few minutes. Conversely, create a convolutional model with command `make letters-model-cnn` that reaches up to 98.5% accuracy.
1. Handwritten Digit Recognition: The images adhere to the same convention as the handwritten character recognition images and are downloaded from a dataset included in the Keras library. To train a neural network to identify them, run `make digits-model`.
1. Handwritten Character Recognition: The letter and digit datasets are concatenated to form one handwritten character set. Running `make chars-model` will produce a model that can accurately distinguish a given character 97% of the time.
1. Handwritten Word Recognition: (not completed)
1. House Number Recognition: (not completed)

After building each model, see the console for the results of running the model against a random testing dataset.

To visualize the results on a random sample, run a specific Make command after the corresponding model is saved.

1. Handwritten Letter Recognition: `make letters-plot` or `make letters-plot-cnn` depending on the model you intend to use.
1. Handwritten Digit Recognition: `make digits-plot`
1. Handwritten Character Recognition: `make chars-plot`
