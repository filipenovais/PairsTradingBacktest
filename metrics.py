import numpy as np
import pandas as pd

def calculate_std_downstd(df):
    rdf = df.pct_change(1)
    for c in rdf.columns:
        rdf[c][rdf[c] > 0] = 0
        rdf[c] = rdf[c]*rdf[c]
    downvar = rdf.sum()/(len(rdf)-1)
    downstd = np.sqrt(downvar.values)
    return rdf.std().iloc[0]*100, downstd*100

def calculate_expected_return(df):
    rdf = df.pct_change(1)
    expected = rdf.mean().iloc[0]*100
    return expected

def calculate_max_drawdown(df):
    roll_max = df.cummax()
    daily_drawdown = df / roll_max - 1.0
    max_drawdown = (daily_drawdown.cummin().values*100)[-1][0]
    return max_drawdown


def calculate_metrics_analysis(values_df):
    values = values_df.values.T[0]
    
    roi = (values[-1]-values[0])/values[0]*100
    
    max_drawdown = calculate_max_drawdown(values_df)

    std, downstd = calculate_std_downstd(values_df)

    expected = calculate_expected_return(values_df)

    sharpe = expected/std
    
    metrics_index = ['RoI %', 'MaxDrawDown %', 'Expected RoI %', 'Volatility %', 'Sharpe Ratio']
    metrics_values = [round(roi,2), round(max_drawdown,2), round(expected,2), round(std,2), round(sharpe,2)]

    return metrics_index, metrics_values



def evaluation_df(list_name, list_results):
    list_results = [r+100 for r in list_results]
    eval_data = {}
    for model_name, results_df in zip(list_name, list_results):
        metrics_index, metrics_values = calculate_metrics_analysis(results_df)
        eval_data[model_name] = metrics_values
    
    eval_df = pd.DataFrame(eval_data, index=metrics_index)
    return eval_df


