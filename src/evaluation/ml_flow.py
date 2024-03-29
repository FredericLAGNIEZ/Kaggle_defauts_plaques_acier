import mlflow

def log_params_metrics(params, metrics, step=None):
    for param, value in params.items():
        mlflow.log_param(param, value)
    for metric, value in metrics.items():
        mlflow.log_metric(metric, value, step=step)
