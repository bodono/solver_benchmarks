import os
import pandas as pd
import numpy as np
import solvers.statuses as statuses
from solvers.solvers import time_limit

MAX_TIMING = time_limit

def get_data(solvers, output_folder, infeasible_test):
    data = {}

    # Get time and status
    for solver in solvers:
        path = os.path.join('.', 'results', output_folder,
                            solver, 'results.csv')
        df = pd.read_csv(path, index_col='name')

        # Get total number of problems
        n_problems = len(df)
        data[solver] = df

    return data


def make_latex_table(solvers, output_folder, infeasible_test):
    data = get_data(solvers, output_folder, infeasible_test)
    # want this type of output:
    #         solver 1 | solver 2 | solver 3
    # prob 1    time   |  FAIL    |  time 
    # prob 2
    for solver in solvers:
      df = data[solver]
      if infeasible_test:
        df.loc[~df['status'].str.contains('infeasible')] = np.nan
      else:  
        df.loc[df['status'] != 'optimal'] = np.nan
      df = df.rename(columns={'run_time': solver})
      data[solver] = df[solver]
  
    probs = output_folder.replace('_', ' ')   
 
    #df = pd.concat([d for d in data.values()])
    df = pd.DataFrame(data)
    table = df.to_latex(float_format="%.4f", longtable=True, index_names=False,
                        caption=f'Solver times on {probs} problems in seconds.')
    table_path = os.path.join('.', 'results', output_folder, f'{output_folder}_table.tex')
    with open(table_path, "w") as f:
      f.write(table)

    path = os.path.join('.', 'results', output_folder, 'failure_rates.csv')
    df = pd.read_csv(path)
    table = df.to_latex(float_format="%.2f", index=False,
                        caption=f'Solver failure rates on {probs} problems.')
    table_path = os.path.join('.', 'results', output_folder, f'{output_folder}_failure_table.tex')
    with open(table_path, "w") as f:
      f.write(table)

    path = os.path.join('.', 'results', output_folder, 'geom_mean.csv')
    df = pd.read_csv(path)
    table = df.to_latex(float_format="%.2f", index=False,
                        caption=f'Solver geometric means on {probs} problems.')
    table_path = os.path.join('.', 'results', output_folder, f'{output_folder}_geom_mean_table.tex')
    with open(table_path, "w") as f:
      f.write(table)

