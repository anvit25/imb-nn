# imb-nn
A modified Nearest Neighbour classifier tailored for datasets with imbalanced classes. The details on be found [here](https://arxiv.org/abs/2206.10866). Implemented via scikit-learn, so should be compatible with most sklearn structures (pipelines, cross validation etc)

Example Code
```python
from proposed import Proposed
from sklearn.metrics import classification_report # Optional

# load_data is not implemented, you must load your own data
# X should be a (n,p) numpy array
# y should be a (n,) numpy binary vector
X_train, X_test, y_train, y_test = load_data()

model = Proposed(n_neighbors = 5)
model.fit(X_train, y_train)

y_pred = model.predict(X_test)
print(classification_report(y_test, y_pred)
```
