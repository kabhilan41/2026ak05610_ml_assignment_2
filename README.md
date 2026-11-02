# Online Shopper Purchase Prediction

Machine Learning - Assignment 2

- Name: Kabhilan S
- Roll No: 2026AK05610
- Programme: M.Tech (AIML)

Live Streamlit app: https://2026ak05610mlassignment2-bxew9n5ev34m45janfv26d.streamlit.app

GitHub repository: https://github.com/kabhilan41/2026ak05610_ml_assignment_2


a. Problem statement

I have taken the Online Shoppers Purchasing Intention dataset from kaggle. One row of this dataset is
one browsing session of a visitor on an online shopping website, and the last column Revenue tells
whether that session ended in a purchase or not.

Most sessions do not end in a purchase, only around 15% of them do. Because of this the shop is not
able to tell which visitor is really going to buy. If we can predict this from the browsing
behaviour of the session, then offers can be shown only to the visitors who are likely to buy, and
the pages where people are dropping off can be improved.

So this is a binary classification problem. The target column is Revenue.

- Revenue = True means the session ended in a purchase
- Revenue = False means the session did not end in a purchase

I trained 5 classification models on this data and compared them using Accuracy, AUC, Precision,
Recall, F1 score and MCC score. After that the saved models are used in a Streamlit app, where a
test csv file can be uploaded and the results of the models can be seen.


b. Dataset description

Dataset name: Online Shoppers Purchasing Intention Dataset

- Source: Kaggle
- Link: https://www.kaggle.com/datasets/imakash3011/online-shoppers-purchasing-intention-dataset/data
- Number of rows: 12330
- Number of columns: 18 (17 input features + 1 target column)
- Target column: Revenue (True or False)
- Missing values: none
- Class distribution: 10422 sessions are no purchase (84.53%) and 1908 sessions are purchase (15.47%)

The assignment needs minimum 12 features and minimum 500 instances. This dataset has 17 features and
12330 rows, so both conditions are satisfied.

The 17 input features are:

- Administrative, Informational, ProductRelated - number of pages of each type visited in the session
- Administrative_Duration, Informational_Duration, ProductRelated_Duration - total time spent on
  those pages, in seconds
- BounceRates, ExitRates, PageValues - the rates and page value given by Google Analytics for the page
- SpecialDay - how close the visit was to a special day like Valentine's day, value between 0 and 1
- Month, VisitorType, Weekend - month of the visit, new or returning visitor, weekend or not
- OperatingSystems, Browser, Region, TrafficType - coded values of the visitor's OS, browser,
  region and the traffic source

Some points I noticed while looking at the data:

1. The classes are imbalanced. Only 15.47% of the sessions end in a purchase, so accuracy alone will
   not show the real picture. That is why I calculated F1 and MCC also.
2. PageValues is the strongest feature. When PageValues is 0, only 3.85% of the sessions end in a
   purchase. When PageValues is above 20, around 72.56% of the sessions end in a purchase.
3. BounceRates and ExitRates are almost the same column, the correlation between them is 0.913.
4. There are no missing values, so nothing had to be filled.

The full csv file is kept in the data folder. The 20% test part is kept separately as test_data.csv
in the main folder, and this is the file used in the app.


c. GitHub repository link

https://github.com/kabhilan41/2026ak05610_ml_assignment_2

Files in the repository:

- app.py - the Streamlit app
- ml_assignment.ipynb - notebook with EDA, training of the 5 models and evaluation
- requirements.txt - packages needed to run the app
- test_data.csv - test data, uploaded in the app
- data folder - online_shoppers_intention.csv, the full dataset
- model folder - saved models and preprocessors

Files inside the model folder:

- logistic_regression.joblib
- decision_tree.joblib
- knn.joblib
- naive_bayes.joblib
- svm.joblib
- preprocessor_scaled.joblib - one hot encoding and standard scaling, used for LR, KNN and SVM
- preprocessor_plain.joblib - one hot encoding only, used for Decision Tree and Naive Bayes
- results.csv - metrics of all 5 models on the test data

To run the app on your own laptop, install the packages and then run streamlit. These are the two
commands:

pip install -r requirements.txt

streamlit run app.py

The app opens at http://localhost:8501. The test_data.csv of the repository is loaded by default. Any
other csv can also be uploaded from the sidebar, but it must have the 17 input columns and the
Revenue column.


d. Models used

All 5 models are trained on the same training data and tested on the same test data, so the
comparison is fair. The split is 80/20, which gives 9864 rows for training and 2466 rows for testing.
I used random_state=42 in train_test_split so I get the same result every time I run it.

1. Logistic Regression - max_iter=1000
2. Decision Tree Classifier - default tree with random_state=42
3. K-Nearest Neighbor Classifier - n_neighbors=5
4. Naive Bayes Classifier - GaussianNB
5. Support Vector Machine - SVC with rbf kernel and C=1.0

Preprocessing done before training:

- Month, OperatingSystems, Browser, Region, TrafficType and VisitorType are one hot encoded.
- The numeric columns are standardised for Logistic Regression, KNN and SVM. KNN and SVM work on
  distances, so a column like ProductRelated_Duration (values in thousands) would otherwise dominate
  everything else.
- Decision Tree and Naive Bayes are trained on the plain numbers, without scaling. I first scaled for
  all the models, but Gaussian NB gave only 27% accuracy that way. PageValues is very skewed and the
  scaled values were confusing the model. On the plain numbers it gives 73% accuracy.

Comparison table on test_data.csv (2466 rows, none of these rows were used in training):

| ML Model Name | Accuracy | AUC | Precision | Recall | F1 | MCC |
|---|---|---|---|---|---|---|
| Logistic Regression | 0.8808 | 0.8830 | 0.7366 | 0.3586 | 0.4824 | 0.4592 |
| Decision Tree | 0.8658 | 0.7378 | 0.5687 | 0.5524 | 0.5604 | 0.4813 |
| KNN | 0.8759 | 0.7659 | 0.6845 | 0.3691 | 0.4796 | 0.4419 |
| Naive Bayes | 0.7328 | 0.7728 | 0.3249 | 0.6728 | 0.4382 | 0.3229 |
| Support Vector Machine | 0.8869 | 0.8827 | 0.7036 | 0.4660 | 0.5606 | 0.5127 |

Observations on the performance of each model:

| ML Model Name | Observation about model performance |
|---|---|
| Logistic Regression | Accuracy is 88.08% and AUC is 0.8830, so it separates the two classes well. Precision is the best of all models (0.7366), which means when it says purchase it is mostly correct. But recall is only 0.3586, so it misses around 64% of the real purchase sessions. This happens because the classes are imbalanced and the model goes with the majority class. |
| Decision Tree | Accuracy is 86.58% but AUC is low (0.7378), because the tree gives hard 0 or 1 probabilities. It also overfits the training data - training accuracy is 1.0 but test accuracy is 0.8658. Recall (0.5524) is better than Logistic Regression so it catches more purchase sessions, but precision drops. |
| KNN | Accuracy is 87.59% which looks fine, but AUC is only 0.7659 and recall is 0.3691. With k=5 the neighbours of a purchase session are mostly non purchase sessions, because 85% of the data is non purchase. Because of this many purchases are missed. It also needs scaling, otherwise the distance gets dominated by the duration columns. |
| Naive Bayes | Weakest model. Accuracy is 73.28% and precision is only 0.3249. Recall is high (0.6728) but it is marking many sessions as purchase wrongly. The reason is that the features are not independent - BounceRates and ExitRates have a correlation of 0.913 - and this breaks the main assumption of Naive Bayes. |
| Support Vector Machine | Best model on this dataset. Accuracy 88.69%, F1 0.5606 and MCC 0.5127 are the highest of all five models. The rbf kernel is able to handle the non linear relation between PageValues, ExitRates and the purchase. AUC is taken from decision_function because this SVM does not give probabilities. |
| Overall, Winner for your dataset? | Support Vector Machine. It has the best accuracy (0.8869), best F1 (0.5606) and best MCC (0.5127), and its AUC (0.8827) is almost same as Logistic Regression (0.8830). Since the data is imbalanced, MCC is the safer metric to trust, so SVM is the winner here. Logistic Regression is very close and trains much faster, so it can be used if speed matters. |


Streamlit app

The app is deployed on Streamlit Community Cloud. It has the following:

1. Dataset upload option - a csv file can be uploaded from the sidebar. If no file is uploaded then
   the test_data.csv of the repository is used.
2. Model selection dropdown - any of the 5 models can be selected.
3. Evaluation metrics - Accuracy, AUC, Precision, Recall, F1 and MCC are shown for the selected
   model, and also for all 5 models together in one table.
4. Confusion matrix and classification report - both are shown for the selected model.

The app only loads the saved joblib files from the model folder. It does not train anything again on
the uploaded data. The uploaded file must have the 17 input columns and the Revenue column.
The test_data.csv of the repository is ready to upload directly.


Note

This work was done on the BITS Virtual Lab, and the screenshot of the execution is added in the
submission PDF.
