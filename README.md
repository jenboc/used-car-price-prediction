# UK Used Car Price Prediction

An exploratory and predictive analysis of used car prices. We examine the relationships
between vehicle characteristics and listing prices before comparing linear and non-linear
machine-learning models and interpreting the factors leading to their predictions.

## Approach

This project applies EDA and machine-learning regression. Relationships
between vehicle characteristics and listing prices are investigated before
comparing linear regression, random forest, gradient boosting, and neural network
models using cross-validation and held-out test data. We then use permutation importance
to interpret the selected model.

## Results

- Nonlinear models substantially outperformed linear regression.
- Random Forest provided the overall strongest performance in cross-validation, and achieved test
MAE = £902, RMSE = £6,199 and $R^2$ = 0.925
- Permutation importance indicated that engine brake horsepower and registration year
are the most important features and supports the relationships we discovered during EDA.
- Car make and model were less important than one might expect, despite their intuitive
relevance.

## Accessing the Project
### Accessing the Data
Due to the size of the dataset, it is not provided directly in the repository.
It can be retrieved from [Kaggle](https://www.kaggle.com/datasets/guanhaopeng/uk-used-car-market)

1. Download the Dataset from [here](https://www.kaggle.com/datasets/guanhaopeng/uk-used-car-market)
2. Extract the csv as save it in `data/all_car_sales.csv`

### Accessing the Notebooks
The `notebooks/` directory contains the Jupyter notebooks created during the
modelling process and are intended to be sequential.

Before running the notebooks, install the required dependencies:
```
pip install -r requirements.txt
```

## Data and Licensing
Data sourced from [Kaggle](https://www.kaggle.com), from the [UK Used Car Listing Data](https://www.kaggle.com/datasets/guanhaopeng/uk-used-car-market)
dataset, published by [Guanhao Peng](https://www.kaggle.com/guanhaopeng).

> This dataset is licensed under the Creative Commons Zero (CC0) 1.0 Universal licence.
