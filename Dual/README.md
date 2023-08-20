# Dual Space

This implementation solves the problem in the dual space

# Installation

## Only Python
Dependencies
- Numpy
- Statsmodels (Optional if one wants to use stacked generalization)

## Python and C++
Dependencies
- Numpy
- Statsmodels (Optional if one wants to use stacked generalization)
- Pybind11
- Eigen3

Run
```bash
g++ -O3 -march=native -shared $(python3 -m pybind11 --includes) pywrap.cpp svm.cpp -o asksvm_utils.so
```

# Usage
This implementation can solve multiclass problem. It can perform cross validation, stacked generalization and a grid search on gamma. A callable kernel function can be passed as an argument or a precomputed kernel
```python
import asksvm

# Load train set, X_train is supposed to be a precomputed kernel matrix
X_train, Y_train = load_train_data()
# Load test set, X_test is supposed to be a precomputed kernel matrix
X_test, Y_test = load_test_data()

# Initialize the model
model = asksvm.svm(gamma = 1, kernel = "precomputed")
# Fit the model
model.fit(X_train, Y_train)
# Compute the output of the model and get the score (accuracy)
score = model.score(X_test, Y_test)
```