
# Model card

## Project context

The real estate company Immo Eliza asked us to create a machine learning model to predict prices of real estate properties in Belgium.

After the scraping, cleaning and analyzing, you are ready to preprocess the data and finally we are going to build a performant machine learning model!

## Data

A clean dataset has been prepared in `data/properties.csv`
- There are about 76 000 properties, roughly equally spread across houses and apartments
- Each property has a unique identifier `id`
- The target variable is `price`
- Variables prefixed with `fl_` are dummy variables (1/0)
- Variables suffixed with `_sqm` indicate the measurement is in square meters
- All missing categories for the categorical variables are encoded as `MISSING`

## Model details

In this analysis, I evaluated three different models for regression tasks:

- Linear Regression: This model assumes a linear relationship between the input features and the target variable. It fits a straight line to the data, aiming to minimize the residual sum of squares between the observed and predicted target values.

- Random Forest Regressor: This is an ensemble learning method that combines multiple decision trees to create a more robust model. Each tree is trained on a random subset of the data and features, and predictions are made by averaging the predictions of individual trees.

- Stacking Method: Stacking is a technique that combines multiple base models with a meta-model to improve predictive performance. Base models are trained independently on the data, and their predictions are used as input features for the meta-model, which learns to make the final predictions.
Linear regression did not performed well on the above data set so i tried other models like random regressor and stacking method. The result was good with both the models, stacking performed slightly better. Finally i choose the stacking method because it combines predictions from multiple models to achieve better accuracy, generalization, and robustness compared to individual models. It's flexible, leveraging the strengths of diverse algorithms, and follows ensemble learning principles, leading to improved predictive performance.
## Performance

Performance metrics for the various models tested:
| Model name              | Train/Test score          |                                      
| --------------          | --------------------------|
| 1. Linear regression    |   Train score - 0.37      |
|                         |   Test score - 0.37       |                                         
|                         |                           | 
| 2. Random forest        |   Train score - 0.95      |
|    regressor            |   Test score - 0.67       |
|                                                     | 
| 3. Stacking method      |   Train score - 0.93      |
|                         |   Test score - 0.71       |
                       




## Limitations
The limitation are -
- Missing Values: Significant missing data across columns may require imputation or removal, potentially leading to information loss.

- High Dimensionality: The dataset has many features post-preprocessing, which can increase computational complexity and risk overfitting.

- Categorical Variables: Handling categorical data with one-hot encoding increases dimensionality and may exacerbate the dimensionality issue.

## Usage

### Dependencies:
The dependencies for training and generating predictions include:

- Python Libraries: scikit-learn, XGBoost, CatBoost
- Data Preprocessing Libraries: pandas, numpy
 ####  Training Scripts:
The training process involves the following steps:

- Data Preprocessing: Preprocess the data by handling missing values, encoding categorical variables, and scaling numerical features.
- Define Base Models: Define a list of base models including RandomForest, LinearRegression, XGBoost, and CatBoost.
- Initialize Stacking Regressor: Initialize a StackingRegressor with the defined base models and a final estimator.
- Train Stacking Regressor: Fit the StackingRegressor on the training data concatenated with the target variable.
- Evaluation: Evaluate the trained model's performance on the training and test datasets.
- Prediction Generation:
### To generate predictions using the trained model:

- Preprocess New Data: Preprocess the new data using the same preprocessing steps applied to the training data.
- Predict: Use the trained StackingRegressor to predict the target variable for the preprocessed new data.
 
```python
def split_data(X, y, test_size=0.2, random_state=None):

def preprocess_data(X_train, X_test):

def train_stacking_regressor(base_models, X_train, y_train_scaled, X_test, y_test_scaled):

def calculate_errors(y_true, y_pred):

```
## Maintainers

- Contact person: Sweta Jain
- Contact no: +32470458300
- Linlked in:www.linkedin.com/in/shweta%2Djain%2D577a3157
