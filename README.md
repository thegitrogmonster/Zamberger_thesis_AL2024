<!-- HEADER -->

<!-- Introduction -->




This Repository is used to investigate the Application of Active Learning for predicting the age of wood from FT-IR spectroscopical data. 
The main focus is the comparison of different sampling strategies, with the uncertainty as a priority. 
Random sampling in an active learning framework is used as the baseline for the comparison. 

TODO: Cite sklearn, and XGBoost
The application for active learning in this scenario will be based on the assumption that a sufficiantly well known model space exists. To represent a few examples of modelbuilding, a few sci-kit learn Regression Methods are tested, as well as XGBoost. 

<!-- USAGE -->
# Usage 

Currently it can be (possibly) only used as a reference to build a similar project. 

<!-- Structure -->
# Structure

The repository is primarily structured as follows: 
* Data Import
* Data Exploration
* Modelbuilding
* Active Learning
* A Library 'al_lib'

<!-- Prerequisites -->
## Prerequisites

This code was developed in a conda environment. To recreate the environment the thesis_conda.yml file can be used: 
 For the full list you can see the thesis_conda.yml for details or create a conda environment from the *spec-file* via:

```
$ conda create --name <ENV_NAME> --file thesis_conda.yml 
```
<!-- Author -->
## Author
Zamberger Bernd
202375[at]fhwn.ac.at