from datetime import datetime

import optuna
from optuna.samplers import TPESampler

from sklearn.metrics import mean_squared_error
from prefect import flow, task
from prefect.task_runners import SequentialTaskRunner

import xgboost as xgb

import mlflow
from mlflow.entities import ViewType
from mlflow.tracking import MlflowClient

from utils import preprocess_data

RS = 2022


@task
def objective(trial, train, valid, y_val):
    params = {
        'max_depth': trial.suggest_int('max_depth', 4, 50, 1),
        'learning_rate': trial.suggest_loguniform('learning_rate', 1e-3, 1),
        'alpha': trial.suggest_loguniform('alpha', 1e-3, 10.0),
        'lambda': trial.suggest_loguniform('lambda', 1e-3, 10.0),
        'min_child_weight': trial.suggest_int('min_child_weight', 1, 10),
        'objective': 'reg:squarederror',
        'eval_metric': trial.suggest_categorical('eval_metric', ['rmse']),
        'seed': RS
    }

    with mlflow.start_run():
        booster = xgb.train(
            params=params,
            dtrain=train,
            num_boost_round=100,
            evals=[(valid, 'validation')],
            early_stopping_rounds=10
        )
        y_pred = booster.predict(valid)
        rmse_valid = mean_squared_error(y_val, y_pred, squared=False)
        mlflow.log_metric("rmse_valid", rmse_valid)

    return rmse_valid


@task
def reg_and_stage_model(tracking_uri: object,
                        experiment_name: object):
    # Run Mlflow client
    client = MlflowClient(tracking_uri=tracking_uri)

    # Search best run
    best_runs = client.search_runs(
        experiment_ids=client.get_experiment_by_name(
            experiment_name).experiment_id,
        filter_string='metrics.rmse_valid < 6.5',
        run_view_type=ViewType.ACTIVE_ONLY,
        max_results=3,
        order_by=["metrics.rmse_valid ASC"]
    )

    # Register and stage best model
    best_model = best_runs[0]
    registered_model = mlflow.register_model(
        model_uri=f"runs:/{best_model.info.run_id}/model",
        name='ny_taxi_trip_duration_pred'
    )

    client.transition_model_version_stage(
        name='ny_taxi_trip_duration_pred',
        version=registered_model.version,
        stage='Staging',
    )

    # Update of staged model
    client.update_model_version(
        name='ny_taxi_trip_duration_pred',
        version=registered_model.version,
        description=f"[{datetime.now()}] "
                    f"The model version {registered_model.version} "
                    f"from experiment {experiment_name} "
                    f"was changed to Staging."
    )


@flow(task_runner=SequentialTaskRunner())
def run(train_path: object,
        val_path: object,
        experiment_name: object,
        tracking_uri: object
        ):
    # Tracking exp
    mlflow.set_tracking_uri(tracking_uri)
    mlflow.set_experiment(experiment_name)
    mlflow.xgboost.autolog()

    # Preprocess data
    X_train, X_val, y_train, y_val, dv = preprocess_data(train_path, val_path)

    # Prepare data for xgboost
    train = xgb.DMatrix(X_train, label=y_train)
    valid = xgb.DMatrix(X_val, label=y_val)

    # Search best params
    sampler = TPESampler(seed=RS)
    study = optuna.create_study(direction='minimize',
                                sampler=sampler)
    study.optimize(
        lambda trial: objective(
            trial,
            train,
            valid,
            y_val
        ),
        n_trials=5
    )


@flow(name='mlflow_staging', task_runner=SequentialTaskRunner())
def main(train_path: object,
         val_path: object
         ) -> object:
    # Setup experiment
    # ctx = get_run_context()
    MLFLOW_TRACKING_URI = "sqlite:///mlflow.db"
    EXPERIMENT_NAME = "nyc-taxi-experiment"

    # Make experiment runs
    run(
        train_path=train_path,
        val_path=val_path,
        experiment_name=EXPERIMENT_NAME,
        tracking_uri=MLFLOW_TRACKING_URI,
    )

    # Registry and stage best model
    reg_and_stage_model(
        tracking_uri=MLFLOW_TRACKING_URI,
        experiment_name=EXPERIMENT_NAME
    )


if __name__ == "__main__":
    params = {
        "train_path": './data/green_tripdata_2021-01.parquet',
        "val_path": './data/green_tripdata_2021-02.parquet'
    }

    main(**params)
