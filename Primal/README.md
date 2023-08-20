# Primal Space
This implementation solves the problem in the primal space

# Installation
Dependencies
- Numpy
- Statsmodels (Optional if one wants to use stacked generalization)

# Usage
This implementation can solve multiclass problem. It can perform cross validation, stacked generalization and a grid search on gamma.
```python
import primal

# Load train set
X_train, Y_train = load_train_data()
# Load test set
X_test, Y_test = load_test_data()

# Initialize the model
model = primal.svm(gamma = 1)
# Fit the model
model.fit(X_train, Y_train)
# Compute the output of the model and get the score (accuracy)
score = model.score(X_test, Y_test)
```