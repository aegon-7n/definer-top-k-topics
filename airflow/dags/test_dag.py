from airflow import DAG
from airflow.operators.python import PythonOperator
from airflow.operators.bash import BashOperator
from datetime import datetime

from models.train import train as train
from models.predict import predict as predict

# import sys
# sys.path.append("/opt/airflow/dags")
# sys.path.append("/opt/airflow/plugins")

# from plugins.models.train import main as train
# from plugins.models.predict import main as predict

# import os
# current_directory = str(os.getcwd())
# relative_path = "/plugins/models"
# dir = current_directory + '/' + relative_path

# def run_python_script():
#     script_path = '/path/to/your/script.py'
#     exec(open(script_path).read())

default_args = {
    "owner": "airflow",
    "start_date": datetime(2024, 4, 1),
    "retries": 5,
    "max_active_tis_per_dag": 1
}

piplines = {'train': {"schedule": "1 * * * *", 'func': train},  # At 20:39 on Saturday MSK
            "predict": {"schedule": "2 * * * *", 'func': predict}}  # At 23:48 every day - 3 hours diff

for task_id, params in piplines.items():
    dag = DAG(task_id,
              schedule_interval=params['schedule'],
              max_active_runs=1,
              default_args=default_args
    )
    with dag: 
        task = PythonOperator(
            task_id=f'{task_id}',
            python_callable=params['func'],
            dag=dag
        )      