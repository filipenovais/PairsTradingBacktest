import numpy as np
import pandas as pd

def calculate_std_downstd(df):
    # Calculate daily returns from the dataframe
    rdf = df.pct_change(1)
    
    # Calculate the standard deviation of daily returns
    std = rdf.std()
    
    # Adjust positive returns to zero to focus on negative returns only
    negative_returns_only = rdf.copy()
    negative_returns_only[negative_returns_only > 0] = 0
    # Calculate the standard deviation of the adjusted returns
    downstd = negative_returns_only.std()
    
    # Return the overall and downside standard deviations, converted to percentages
    return std.iloc[0] * 100, downstd.iloc[0] * 100

def calculate_expected_return(df):
    # Calculate daily returns
    rdf = df.pct_change(1)
    # Calculate and return the average daily return, converted to percentage
    expected = rdf.mean().iloc[0]*100
    return expected

def calculate_max_drawdown(df):
    # Calculate the cumulative maximum of the dataframe
    roll_max = df.cummax()
    # Calculate daily drawdown
    daily_drawdown = df / roll_max - 1.0
    # Determine the maximum drawdown, converted to percentage
    max_drawdown = (daily_drawdown.cummin().values*100)[-1][0]
    return max_drawdown


def calculate_metrics_analysis(values_df):
    # Extract the first column of values
    values = values_df.values.T[0]
    
    # Calculate Return on Investment (RoI)
    roi = (values[-1]-values[0])/values[0]*100
    
    # Calculate Maximum Drawdown
    max_drawdown = calculate_max_drawdown(values_df.copy())

    # Calculate Volatility and Downside Volatility
    std, downstd = calculate_std_downstd(values_df.copy())

    # Calculate Expected Return
    expected = calculate_expected_return(values_df.copy())

    # Calculate Sharpe Ratio
    sharpe = expected/std

    # Calculate Sortino Ratio
    sortino = expected/downstd

    # Prepare metrics for output
    metrics_index = ['RoI %', 'MaxDrawDown %', 'Expected RoI %', 'Downside Volatility %', 'Sortino Ratio', 'Volatility %', 'Sharpe Ratio']
    metrics_values = [round(roi,2), round(max_drawdown,2), round(expected,2), round(downstd,2), round(sortino,2), round(std,2), round(sharpe,2)]

    return metrics_index, metrics_values


def evaluation_df(list_name, list_results):
    # Adjust input results by adding 100 for percentage calculation
    list_results = [r+100 for r in list_results]
    eval_data = {}
    # Loop through each model and its results to calculate metrics
    for model_name, results_df in zip(list_name, list_results):
        metrics_index, metrics_values = calculate_metrics_analysis(results_df)
        eval_data[model_name] = metrics_values
    
    # Create a dataframe to display evaluation results
    eval_df = pd.DataFrame(eval_data, index=metrics_index)
    return eval_df
