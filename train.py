import argparse
import mlflow
from sklearn.datasets import make_regression
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import train_test_split

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--model_name",
        type=str,
        required=True,
        help="Model name (required)"
    )
    return parser.parse_args()

args = parse_args()

model_name = args.model_name

X, y = make_regression(n_features=4, n_informative=2, random_state=0, shuffle=False)
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

params = {"max_depth": 2, "random_state": 42}
model = RandomForestRegressor(**params)
model.fit(X_train, y_train)

# Log parameters and metrics using the MLflow APIs
mlflow.log_params(params)

y_pred = model.predict(X_test)
mlflow.log_metrics({"mse": mean_squared_error(y_test, y_pred)})

# we use model name as the artifact path for now
mlflow_model = mlflow.sklearn.log_model(
    model,
    model_name,
    input_example=X_train,
    registered_model_name=model_name,
)
