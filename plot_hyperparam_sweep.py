from collections import defaultdict
from os.path import exists

import pandas
import pandas as pd
import wandb
import matplotlib.pyplot as plt
import matplotlib as mpl

# MPL settings, do not modify
mpl.rcParams['figure.dpi'] = 300
mpl.rcParams['font.family'] = 'Microsoft Sans Serif'
mpl.rcParams['font.size'] = 20
mpl.rcParams['axes.linewidth'] = 2
plt.style.use('ggplot')
pd.set_option('display.max_colwidth', 40)
plt.rcParams.update({
    "text.usetex": True,
    "font.family": "sans-serif",
    "font.sans-serif": ["Helvetica"]
})

RESULTS_FILENAME = 'hyperparams.csv'


if __name__ == '__main__':
    if exists(RESULTS_FILENAME):
        runs_df = pandas.read_csv(RESULTS_FILENAME)
    else:
        api = wandb.Api()
        entity, project = 'mikimn', 'interpretable-transformer'  # set to your entity and project
        prefix = 'bert-mask-all-sparsity'
        keys = ['lambda_init', 'lambda_lasso', 'selected', 'eval/accuracy', 'train/dataset/snli_hard.py_eval_accuracy',
                'train/hans_eval_accuracy_0', 'train/hans_eval_accuracy_1']
        names = ['lambda1', 'lambda2', 'Selected', 'Accuracy', 'Hard', 'HANS (+)', 'HANS (-)']
        runs = api.runs(entity + "/" + project)

        name_list = []
        values = defaultdict(list)
        for run in runs:
            if not run.name.startswith(prefix) or run.state == 'running':
                continue
            # .summary contains the output keys/values for metrics like accuracy.
            #  We call ._json_dict to omit large files

            # .config contains the hyperparameters.
            #  We remove special values that start with _.
            stuff = {**run.config, **run.summary._json_dict}

            for key in keys:
                values[key].append(stuff[key])

            # .name is the human-readable name of the run.
            name_list.append(run.name)

        runs_df = pd.DataFrame({
            **values,
            "name": name_list
            }).rename(dict(zip(keys, names)), axis='columns')
        runs_df.to_csv(RESULTS_FILENAME)

    lambda_1_values = sorted(list(runs_df['lambda1'].unique()))
    num_lambda_1 = len(lambda_1_values)
    fig, axes = plt.subplots(ncols=num_lambda_1, sharey=True, figsize=(12, 2.5))
    handles, labels = None, None
    for idx, (ax, lambda1) in enumerate(zip(axes, lambda_1_values)):
        plot_key = 'lambda1'
        x_key = 'lambda2'
        lambda_df = runs_df[(runs_df[plot_key] == lambda1) & (runs_df['Selected'] > 0.1)].sort_values(by=x_key)
        x = lambda_df[x_key]
        y1 = lambda_df['Accuracy']
        y2 = lambda_df['Hard']
        y3 = 1. - lambda_df['Selected']
        ax2 = ax.twinx()
        ax.plot(x, y1, marker='o', c='r', label='Dev.')
        ax.plot(x, y3, marker='.', linestyle='dashed', c='gray', label=r'\% Masked')
        ax.set_xlabel(f'lambda1 = {lambda1}')
        # plt.setp(ax.get_yticklabels(), visible=False)
        ax2.plot(x, y2, marker='^', c='g', label='Hard')
        # plt.setp(ax2.get_yticklabels(), visible=False)
        ax2.grid(None)
        if idx == 0:
            handles, labels = ax.get_legend_handles_labels()
            handles2, labels2 = ax2.get_legend_handles_labels()
            handles += handles2
            labels += labels2

    # plt.ylim(0.8, 0.91)
    # fig.subplots_adjust(bottom=0.3, wspace=0.33)
    fig.legend(handles, labels, loc='lower center', ncol=len(labels), bbox_to_anchor=(0.5, -0.15))
    fig.tight_layout()
    plt.savefig('hyperparams.pdf', bbox_inches='tight')



