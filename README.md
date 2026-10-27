# Deep Learning Lab 1: Regression and Classification with PyTorch

This lab implements multilayer perceptrons (MLPs) for two supervised learning tasks: predicting next-day stock closing prices and classifying machine failures. The workflow covers exploratory data analysis, preprocessing, model training, hyperparameter tuning, evaluation, and regularization.

## Tools

Python, Jupyter Notebook, PyTorch, pandas, NumPy, scikit-learn, skorch, imbalanced-learn, Matplotlib, and Seaborn.

## Part 1: Stock Price Regression

**Dataset:** Kaggle NYSE, using `prices-split-adjusted.csv` (851,264 rows, 501 companies, 2010-2016).

- Predict the next trading day's closing price using nine features, including current prices, returns, relative price range, a moving-average ratio, and log-transformed volume.
- Split observations chronologically into 80% training and 20% testing, with scalers fitted only on training data.
- Train an initial `9 -> 64 -> 32 -> 1` MLP with ReLU activations, MSE loss, and SGD.
- Tune the optimizer, learning rate, hidden-layer size, and training duration using `GridSearchCV`, skorch, and three-fold time-series cross-validation.

The selected configuration uses **Adam, a learning rate of 0.01, and 40 epochs**, with hidden layers of 64 and 32 neurons.

| Model | Test RMSE ($) | Predictions within 1% of actual price |
| --- | ---: | ---: |
| Baseline: tomorrow's close = today's close | 2.179 | 55.3% |
| Initial MLP: SGD, 20 epochs | 2.656 | 35.8% |
| Tuned MLP: Adam, 40 epochs | 2.246 | 52.9% |

Tuning improved the MLP, but it did not outperform the persistence baseline. L2 weight decay, dropout, batch normalization, and dropout combined with L2 did not improve the tuned model in the tested configurations.

## Part 2: Machine Failure Classification

**Dataset:** Kaggle Machine Predictive Maintenance Classification, using `predictive_maintenance.csv` (10,000 rows, six classes).

- Remove nine contradictory rows, drop identifiers and the label-related `Target` column, and encode machine type and failure labels.
- Use a stratified 80/20 split and standardize the five sensor features using training statistics.
- Apply **SMOTENC only to the training set**, increasing each failure class to 1,000 samples while keeping the test distribution unchanged.
- Train an initial `6 -> 64 -> 32 -> 6` MLP with cross-entropy loss and SGD.
- Tune the architecture and training parameters using three-fold `GridSearchCV`, scored by macro F1.

The selected configuration uses **Adam, a learning rate of 0.001, 50 epochs, and hidden layers of 128 and 64 neurons**.

| Test metric | Tuned MLP |
| --- | ---: |
| Accuracy | 95.40% |
| Macro precision | 49.26% |
| Macro recall | 62.35% |
| Macro F1 | 54.61% |
| Weighted F1 | 96.40% |

The model detects several failure types but misses all Random Failures and Tool Wear Failures in the test set. Since approximately 96.5% of samples belong to **No Failure**, accuracy alone hides poor minority-class performance. Dropout, L2 weight decay, and early stopping were also explored.

## Limitations and Improvements

Classification cross-validation was performed after oversampling, which can make validation scores optimistic. SMOTENC should instead be applied separately within each training fold. Early stopping also monitored test loss; a separate validation set is needed for model selection and an unbiased final test evaluation. The smallest failure classes contain very few test examples, making their metrics uncertain.

## Running the Notebook

Install the dependencies from the project root:

```bash
python -m pip install -r requirements.txt
```

Place the datasets at the paths expected by the notebook:

- `../NewYorkStock/prices-split-adjusted.csv`
- `../machine-predictive/predictive_maintenance.csv`

Open the lab notebook in Jupyter and run its cells in order. The reported experiments were run locally on CPU; results may vary with random initialization.
