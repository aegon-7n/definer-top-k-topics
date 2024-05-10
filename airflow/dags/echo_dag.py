from airflow import DAG
from airflow.operators.bash import BashOperator
from airflow.operators.python import PythonOperator
from datetime import datetime

import os
current_directory = os.getcwd()

def my_python_function():
    print(current_directory)

default_args = {
    'owner': 'airflow',
    'start_date': datetime(2024, 5, 1),
    'retries': 1
}

with DAG(
    'echo_dag',
    default_args=default_args,
    schedule_interval='@once',
    catchup=False
) as dag:

    t1 = BashOperator(
        task_id='echo_hi',
        bash_command='echo "Hello"',
    )

    t2 = BashOperator(
        task_id='pwd',
        bash_command='pwd',
    )

    t3 = PythonOperator(
    task_id='run_python_script',
    python_callable=my_python_function,
    dag=dag,
)

    t1 >> t2 >> t3