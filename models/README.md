# models/

Place the trained model artifact here:

```
models/price_classifier.pkl
```

It is **not** committed to git (≈16 MB binary, ignored in `.gitignore`). Copy it from
Project 1 (`makanpredict/models/price_classifier.pkl`) or re-train it there with
`python -m src.train`.

The `.pkl` is a joblib dict bundling the fitted scikit-learn pipeline + XGBoost model,
the feature reference, the item catalog, the class list, and the training metrics — so
the API derives all of its valid values and metadata straight from this one file.
