<!-- HEADER -->

<!-- Introduction -->

This Repository is used to investigate the Application of Active Learning for predicting the age of wood from FT-IR spectroscopical data. 
The main focus is the comparison of different sampling strategies, with uncertainty as a priority. 
Random sampling in an active learning framework is used as the baseline for the comparison. 

The code is developed with functionalities from scikit-learn [[1]](#1), especially RandomSearchCV, GridSearchCV, as well as most statistical models. 

The application for active learning in this scenario will be based on the assumption that a sufficiantly well known model space exists. To represent a few examples of modelbuilding, some sci-kit learn Regression Methods are tested, as well as XGBoost[[2]](#2). 

For the active Learning some selection strategies are implemented, but could be extendend with another function definition in the al_lib/selection_criteria.py file.

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
<!-- References -->

## References
<a id="1">[1]</a> 
F. Pedregosa, G. Varoquaux, A. Gramfort, V. Michel, B. Thirion, O. Grisel,
M. Blondel, P. Prettenhofer, R. Weiss, V. Dubourg, J. Vanderplas, A. Passos,
D. Cournapeau, M. Brucher, M. Perrot, and E. Duchesnay, “Scikit-learn: Machine learning in Python,” Journal of Machine Learning Research, vol. 12, pp. 2825–2830,
2011, accessed: 07.06.2024

<a id="2">[2]</a> 
T. Chen and C. Guestrin, “XGBoost: A scalable tree boosting system,”
in Proceedings of the 22nd ACM SIGKDD International Conference on
Knowledge Discovery and Data Mining, ser. KDD ’16. New York, NY,
USA: ACM, 2016, pp. 785–794, accessed: 07.06.2024. [Online]. Available:
http://doi.acm.org/10.1145/2939672.2939785



<!-- Author -->
## Author
Zamberger Bernd
202375[at]fhwn.ac.at