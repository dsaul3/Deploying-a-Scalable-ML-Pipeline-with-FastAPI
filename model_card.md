# Model Card

## Model Details

* **Developed by:** Dylan Saul for an Individual student project (*Deploying a Machine Learning Model with FastAPI* through Udacity)
* **Model date:** 2026
* **Model version:** 1
* **Training algorithm:** A tree-based classifier (via `train_model` fuction): `sklearn` `RandomForestClassifier` for classification and `GridSearchCV` for model hyperparameter tuning.
* **Features**:
    * **Numerical:** age, fnlgt, education-num, capital-gain, capital-loss, hours-per-week

    * **Categorical:** workclass, education, marital-status, occupation, relationship, race, sex, native-country
* **Preprocessing:**
    * `OneHotEncoder` for categorical features
    * `LabelBinarizer` for target variable `salary`
* **Label:** Binary classification: less-than-or-equal-to 50k and greater-than 50k
* **License:** Creative Commons Attribution-NonCommercial- NoDerivs 3.0 License, clarified in `LICENSE.txt`
* **Contact:** Project repository owner
 
## Intended Use

### Primary Intended Uses

* Educational demonstration of an end-to-end ML pipeline
* Practice for model training, evaluation, and deployment

### Primary Intended Users

* Reviewers evaluating ML pipeline projects

### Out-of-Scope Use Cases

* Real world applications and decision-making
* Any system affecting individual financial or social outcomes

## Training Data

* Census data provided at: https://archive.ics.uci.edu/dataset/20/census+income
* Training subset: 80% of full dataset
* Demographic and employment attributes

## Evaluation Data

* Census data provided at: https://archive.ics.uci.edu/dataset/20/census+income
* Preproccessed train/test split (80/20), one-hot encoding of categorical values, label binarization of target variable (`salary`)

## Metrics

* Model Performance Measures: Precision, Recall, F1 Score
* Results: Precision: 0.7610, Recall: 0.6384, F1: 0.6944

## Ethical Considerations

* The dataset includes attributes with historical and societal biases that the model may inherit (race, sex, nationality)
* Real-world decision-making using this model could reinforce inequality or discrimination along those attributes

## Caveats and Recommendations

### Caveats

* This model is a basic model without fairness constraints
* The dataset may not reflect most current data
* Not suitable for any practical applications in real-world use-cases

## Recommendations

* Perform fairness analysis
* Mitigate bias
* Use cross validation instead of a single split
* Use datasets that are both the most recent and representitive possible
