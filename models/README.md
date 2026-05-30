# models/

Holds the trained model artifact:

```
models/price_classifier.pkl
```

This file **is committed** (~16 MB) so the app runs out-of-the-box — including the
one-click Streamlit Community Cloud deploy, where the host needs the model present.
(Any *other* `*.pkl` is git-ignored.)

It's a joblib dict bundling the fitted scikit-learn pipeline + XGBoost model, the
feature reference, the item catalog, the class list, and the training metrics — so the
app derives all of its valid values and metadata straight from this one file.

> Originally from Project 1 (`makanpredict/models/price_classifier.pkl`), re-trainable
> there with `python -m src.train`.
