from prefect.deployments import DeploymentSpec
from prefect.orion.schemas.schedules import IntervalSchedule
from prefect.flow_runners import SubprocessFlowRunner
from datetime import timedelta


DeploymentSpec(
    name="deploy-mlflow",
    flow_name='mlflow-staging',
    schedule=IntervalSchedule(interval=timedelta(minutes=10)),
    flow_location="./mlflow_main.py",
    flow_runner=SubprocessFlowRunner(),
    parameters={
        "train_data_path": './data/green_tripdata_2021-01.parquet',
        "valid_data_path": './data/green_tripdata_2021-02.parquet'
    },
    tags=["mlflow_taxi_trip_pred"]
)
