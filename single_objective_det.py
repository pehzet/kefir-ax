import pandas as pd
from ax.service.ax_client import AxClient, ObjectiveProperties
from ax.modelbridge.generation_strategy import GenerationStrategy, GenerationStep
from ax.service.utils.report_utils import exp_to_df
from ax.modelbridge.registry import Models
import os
from icecream import ic
# Dateipfad zur Excel-Datei
EXCEL_PATH = "experiment_data.xlsx"
VERBOSE = False
# Definition des Experiments
parameters = [
    {"name": "x1", "type": "range", "bounds": [0.0, 10.0], "value_type": "float"},
    {"name": "x2", "type": "range", "bounds": [0.0, 10.0], "value_type": "float"}
]

objectives = { "y_mean" : ObjectiveProperties(minimize=False) }
objective_name = list(objectives.keys())[0]
# AxClient initialisieren

generation_strategy = GenerationStrategy(
    steps=[

        GenerationStep(
            model=Models.BOTORCH_MODULAR,
            num_trials=-1,  # Unbegrenzt viele BO-Schritte
        ),
    ]
)

def create_ax_client():
    ax_client = AxClient(generation_strategy=generation_strategy, verbose_logging=VERBOSE)
    ax_client.create_experiment(
        name="async_experiment",
        parameters=parameters,
        objectives=objectives,
    )
    return ax_client

# Falls die Excel-Datei existiert, lade die bisherigen Daten
def load_existing_data():
    if os.path.exists(EXCEL_PATH):
        df = pd.read_excel(EXCEL_PATH)
        return df
    else:
        raise FileNotFoundError("File does not exist")
      
def add_trials_to_ax(ax_client:AxClient, df:pd.DataFrame):
    for idx, row in df.iterrows():
        parameters = {k: v for k, v in row.items() if k in ["x1", "x2"]}
        param, trial_idx = ax_client.attach_trial(parameters=parameters)
        objective = row["y_mean"] 
        if objective is not None:
            ax_client.complete_trial(trial_index=trial_idx, raw_data=objective)

def append_trial_to_df(df:pd.DataFrame, trial, trial_idx):
    new_row = trial # Später ggf. anpassen
    new_row["trial_index"] = trial_idx
    new_row["y_mean"] = None
    new_row["y_sem"] = None
    return pd.concat([df, pd.DataFrame([new_row])], ignore_index=True)

def save_data(df, path=None):
    if path is None:
        path = EXCEL_PATH
    df.to_excel(path, index=False)

def optimization_run():
    df = load_existing_data()
    ax_client = create_ax_client()
    add_trials_to_ax(ax_client, df)
    trial, trial_idx = ax_client.get_next_trial()
    ic(trial, trial_idx)
    df = append_trial_to_df(df, trial, trial_idx)
    save_data(df)

optimization_run()