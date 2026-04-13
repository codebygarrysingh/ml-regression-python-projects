# ML Regression Engineering

> End-to-end regression implementations in Python — from mathematical foundations through production-quality pipelines. Covers linear algebra, gradient descent, feature engineering, and model evaluation with a practitioner's engineering rigour.

[![Python](https://img.shields.io/badge/Python_3.11-3776AB?style=flat&logo=python&logoColor=white)](https://python.org)
[![NumPy](https://img.shields.io/badge/NumPy-013243?style=flat&logo=numpy&logoColor=white)](https://numpy.org)
[![Scikit-Learn](https://img.shields.io/badge/Scikit--Learn-F7931E?style=flat&logo=scikitlearn&logoColor=white)](https://scikit-learn.org)
[![Jupyter](https://img.shields.io/badge/Jupyter-F37626?style=flat&logo=jupyter&logoColor=white)](https://jupyter.org)

---

## What This Covers

These notebooks are engineered for practitioners who need to understand *why* algorithms work, not just *how* to call them — a prerequisite for production ML and LLM engineering work.

---

## Module 1: Linear Regression — First Principles

**Mathematical foundation first.** Understanding gradient descent, cost functions, and feature scaling is directly applicable to fine-tuning LLMs, designing loss functions for custom models, and diagnosing training instability in neural networks.

### Cost Function Landscape

```python
def mean_squared_error(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """
    MSE = (1/2m) Σ (ŷᵢ - yᵢ)²
    The 1/2 factor is a calculus convenience — it cancels the 2 from
    the power rule when computing the gradient, giving cleaner updates.
    """
    m = len(y_true)
    return (1 / (2 * m)) * np.sum((y_pred - y_true) ** 2)
```

### Gradient Descent — Vectorised Implementation

```python
def gradient_descent(
    X: np.ndarray,    # (m, n) feature matrix
    y: np.ndarray,    # (m,) target vector
    alpha: float,     # learning rate
    iterations: int,
) -> tuple[np.ndarray, list[float]]:
    """
    Vectorised gradient descent — O(m·n) per iteration vs O(m·n²) naive loop.
    Key insight: the gradient is just X.T @ (Xw - y) / m
    """
    m, n = X.shape
    w = np.zeros(n)
    cost_history = []

    for _ in range(iterations):
        predictions = X @ w
        errors = predictions - y
        gradient = (X.T @ errors) / m
        w -= alpha * gradient
        cost_history.append(mean_squared_error(y, X @ w))

    return w, cost_history
```

### Feature Scaling — Why It's Non-Negotiable

```
Without scaling:           With StandardScaler:
  feature_1: [0, 100000]    feature_1: [-1.5, 1.5]
  feature_2: [0.001, 0.01]  feature_2: [-1.5, 1.5]

Gradient descent with unscaled features:
  → Elongated loss surface → zigzag convergence → 10-100× more iterations
  → Numerically identical to poorly conditioned matrix inversion
```

---

## Module 2: Linear Algebra for ML

Direct application of linear algebra to ML — why matrix operations underpin everything from regression to transformers:

| Concept | ML Application |
|---|---|
| Matrix multiplication | Forward pass in neural networks |
| Eigendecomposition | PCA, understanding covariance structure |
| Singular Value Decomposition | Low-rank approximations, LLM weight compression (LoRA) |
| Solving Ax = b | Normal equations, least squares |
| Vector spaces | Embedding spaces, semantic similarity |

---

## Module 3: Regression Toolkit — Production Patterns

```python
class RegressionPipeline:
    """
    Production-ready regression pipeline:
    - Handles missing values before fitting (no leakage)
    - Scales features on training data only
    - Supports polynomial feature expansion
    - Evaluates with multiple metrics (RMSE, MAE, R², MAPE)
    """

    def __init__(self, degree: int = 1, regularisation: str = "ridge"):
        self.scaler = StandardScaler()
        self.poly = PolynomialFeatures(degree=degree, include_bias=False)
        self.model = Ridge() if regularisation == "ridge" else Lasso()
        self.is_fitted = False

    def fit(self, X_train: np.ndarray, y_train: np.ndarray) -> "RegressionPipeline":
        X_poly = self.poly.fit_transform(X_train)
        X_scaled = self.scaler.fit_transform(X_poly)  # Fit on train only
        self.model.fit(X_scaled, y_train)
        self.is_fitted = True
        return self

    def evaluate(self, X_test: np.ndarray, y_test: np.ndarray) -> dict[str, float]:
        X_poly = self.poly.transform(X_test)
        X_scaled = self.scaler.transform(X_poly)  # Transform only
        y_pred = self.model.predict(X_scaled)
        return {
            "rmse": np.sqrt(mean_squared_error(y_test, y_pred)),
            "mae": mean_absolute_error(y_test, y_pred),
            "r2": r2_score(y_test, y_pred),
        }
```

---

## Learning Path

These notebooks are structured as a progression:

```
1. Linear Regression (maths) → 2. Linear Algebra → 3. Toolkit
         │                              │                │
         ▼                              ▼                ▼
   Gradient descent             Matrix operations    Production API
   Cost functions               Vectorisation        Cross-validation
   Feature scaling              SVD / PCA            Regularisation
         │
         ▼
   ml-neural-network-projects  (extends to deep learning)
   production-rag-pipeline      (applies to embedding spaces)
```

---

## Related Work

- [ml-neural-network-projects](https://github.com/codebygarrysingh/ml-neural-network-projects) — extends to LSTM and deep learning
- [ml-classification-projects](https://github.com/codebygarrysingh/ml-classification-projects) — logistic regression and classification
- [data-prep-utility](https://github.com/codebygarrysingh/data-prep-utility) — production data preprocessing pipeline

---

## Author

**Garry Singh** — Principal AI & Data Engineer · MSc Oxford

[Portfolio](https://garrysingh.dev) · [LinkedIn](https://linkedin.com/in/singhgarry) · [Book a Consultation](https://calendly.com/garry-singh2902)
