Title: Feature Extraction and Machine Learning for Hepatocyte Ballooning Detection - Phase 3
Description:
This project aims to enhance the detection of hepatocyte ballooning in liver ultrasound images by combining deep learning feature extraction with traditional machine learning techniques. The work will build upon the previously developed InceptionV3 model.
Key Tasks:

Feature Extraction:

Use the pre-trained InceptionV3 model (from Phase 2) as a feature extractor for liver ultrasound images.
Extract and save features for the entire dataset.


Traditional Machine Learning Classification:

Implement and train the following classifiers using the extracted features:
a. Random Forest
b. Support Vector Machine (SVM)
c. Gradient Boosting (XGBoost)
d. K-Nearest Neighbors (KNN)
Perform basic hyperparameter tuning for each classifier to optimize performance.


Dichotomous Classification Strategy:

Implement a two-step classification approach:
a. Normal vs. Abnormal
b. Few vs. Many (for samples classified as Abnormal)
Apply this strategy using the best performing classifier(s) from step 2.


Model Evaluation and Comparison:

Evaluate each classifier using appropriate metrics (accuracy, sensitivity, specificity, ROC-AUC).
Compare the performance of traditional ML classifiers with the previous deep learning results.
Analyze the effectiveness of the dichotomous classification strategy.


Documentation and Reporting:

Document the methodology, including feature extraction process and classifier implementations.
Prepare a comprehensive report of results, including performance metrics and comparisons.
Update the methods and results sections of the research paper draft.



Deliverables:

Python code for feature extraction, classifier training, and evaluation.
Trained machine learning models and their performance metrics.
Updated sections of the research paper (methods and results).
A brief report summarizing the findings and comparing with previous phases.