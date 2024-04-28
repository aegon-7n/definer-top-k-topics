from airflow import DAG
from airflow.operators.bash import BashOperator
from pendulum import today


default_args = {
    "owner": "aegon7n",
    "start_date": today('UTC').add(days=-1),
    "retries": 5,
    "max_active_tis_per_dag": 1
}

pipelines = {"train": {"schedule": "1 * * * *"},  
             "predict": {"schedule": "2 * * * *"}}  

def init_dag(dag, task_id):
    with dag:
        t1 = BashOperator(
            task_id=task_id,
            bash_command=f'python3 /home/aegon/Documents/Development/Projects/topic-sentiment-explorer/{task_id}.py'
        )
    return dag

for task_id, params in pipelines.items():
    dag = DAG(
        task_id,
        schedule=params['schedule'],
        max_active_runs=1,
        default_args=default_args
    )
    init_dag(dag, task_id)
    globals()[task_id] = dag