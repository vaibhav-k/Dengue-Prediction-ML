# Dengue-Prediction-ML

Dengue severity prediction using machine learning.

Analysed - 
* 5' and 3' UTR sequences
* Metadata (both genotype sorted, for all 4 genotypes, and genotype independent)
* Proteins (both independently as well as all 11 combined, for all 4 genotypes)

Protein sequences are available for -
* 2K
* Capsid
* Envelope
* NS1
* NS2A
* NS2A
* NS3
* NS4A
* NS4B
* NS5
* PrM

The algorithms used are -
* AdaBoost
* Decision Tree
* A sequential neural network
* K-Nearest Neighbour
* Kernel SVM
* Logistic Regression
* Naive Bayes
* Random Forest
* SVM
* XGBoost

Also carried out Nested Cross-Validation to select the apprpriate hyper-parameters for all the algorithms used and used stratification to account for the imbalance of different severity's data.
