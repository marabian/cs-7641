import sys
import os
import glob
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import pandas as pd

sns.set_theme(style='whitegrid')


def return_data(algo_name, problem_name):
    dfs = []
    for fp in glob.glob(f'data/csv/{algo_name}_{problem_name}_*.csv'):
        df = pd.read_csv(fp)
        dfs.append(df)
    avg = pd.concat(dfs).groupby(level=0).mean()
    return avg

def plot_iters_vs_fitness(problem_name):
    plt.clf()
    
    df = return_data("RHC", problem_name)
    plot=sns.lineplot(x="iters", y="fitness", data=df, legend="full", label="RHC")

    df = return_data("SA", problem_name)
    plot=sns.lineplot(x="iters", y="fitness", data=df, legend="full", label="SA")

    df = return_data("GA", problem_name)
    plot=sns.lineplot(x="iters", y="fitness", data=df, legend="full", label="GA")

    df = return_data("MIMIC", problem_name)
    plot=sns.lineplot(x="iters", y="fitness", data=df, legend="full", label="MIMIC")

    ax = plot.axes
    ax.legend(loc="best")
    plot.set_title(f"{problem_name} - Convergence Graph of Fitness")

    figure=plot.get_figure()
    figure.savefig(os.path.join('data/plot', f'{problem_name}_convergence.png'))

def plot_iters_vs_fevals(problem_name):
    plt.clf()

    df = return_data("RHC", problem_name)

    plot=sns.lineplot(x="iters", y="fevals", data=df, legend="full", label="RHC")

    df = return_data("SA", problem_name)

    plot=sns.lineplot(x="iters", y="fevals", data=df, legend="full", label="SA")

    df = return_data("GA", problem_name)

    plot=sns.lineplot(x="iters", y="fevals", data=df, legend="full", label="GA")

    df = return_data("MIMIC", problem_name)

    plot=sns.lineplot(x="iters", y="fevals", data=df, legend="full", label="MIMIC")

    ax = plot.axes
    ax.legend(loc="best")
    plot.set_title(f"{problem_name} - Function Evaluations vs Iterations")

    figure=plot.get_figure()
    figure.savefig(os.path.join('data/plot', f'{problem_name}_fevals_v_iters.png'))



problem_name = sys.argv[1]
print(f"Plotting {problem_name} graphs")

plot_iters_vs_fitness(problem_name)
plot_iters_vs_fevals(problem_name)