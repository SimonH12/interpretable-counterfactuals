import apply
import json
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns


def plot_annotate_cfs(df, ax):
    for i in range(len(df.index)):
        ax.annotate(str(i), (df.iloc[i]['sepal length']+0.05, df.iloc[i]['sepal width']))


def plot_cfs(dataframe, instance, counterfactuals, ax):
    plot_annotate_cfs(counterfactuals, ax)

    dataframe['type'] = np.where(dataframe['label'] == 0, 'class:0', 'class:1')
    instance['type'] = 'instance'
    counterfactuals['type'] = 'counterfactuals'
    
    combined = pd.concat([dataframe, instance, counterfactuals])
    markers= {'class:0': 'o', 'class:1': 'o', 'instance': 'D', 'counterfactuals': 'X'}
    sns.scatterplot(data=combined, x="sepal length", y="sepal width", hue="type", style="type", markers=markers, ax=ax, s=90)
    ax.legend(loc='lower right')


def plot_cfs_3d(dataframe, instance, counterfactuals, ax):
    plot_annotate_cfs(counterfactuals, ax)

    dataframe['type'] = np.where(dataframe['label'] == 0, 'class:0', 'class:1')
    instance['type'] = 'instance'
    counterfactuals['type'] = 'counterfactuals'
    
    combined = pd.concat([dataframe, instance, counterfactuals]).reset_index()
    combined['petal length'] = combined['petal length'].replace({'v': 'small', 's': 'medium', 'p': 'large'})
    markers= {'small':'v', 'medium':'s', 'large':'p'}
    sns.scatterplot(data=combined, x="sepal length", y="sepal width", hue="type", style="petal length", markers=markers, ax=ax, s=90)
    ax.legend(loc='lower right')


def get_df_data(path, path_train = r"datasets\toy_dataset_iris_traindataset.csv"):
    json_file = open(path)

    cf_data = json.load(json_file)
    dataframe = pd.read_csv(path_train)
    instance = pd.DataFrame(cf_data["test_data"][0], columns = cf_data["feature_names_including_target"])
    counterfactuals = pd.DataFrame(cf_data["cfs_list"][0], columns = cf_data["feature_names_including_target"])
    return dataframe,instance,counterfactuals


def plot_dice_gene_abno_prox(plot_cfs, target_class_name, target_class, dataframe, instance, counterfactuals, w=15, h=12):
    metrics = ['abnormality', 'proximity', 'generality', 'obtainability']
    cf_metrics, _, _ = apply.apply_all_metrics(dataframe, instance, counterfactuals, target_class, target_class_name)

    print("\ncf_metrics\n", cf_metrics)
    cf_generality = cf_metrics.sort_values(by=['generality'], ascending=False).reset_index().drop(metrics, axis="columns")
    cf_proximity = cf_metrics.sort_values(by=['proximity']).reset_index().drop(metrics, axis="columns")
    cf_abnormality = cf_metrics.sort_values(by=['abnormality']).reset_index().drop(metrics, axis="columns")

    print("\ncf_abnormality\n", cf_abnormality)
    fig, axs = plt.subplots(2,2)

    plot_cfs(dataframe, instance, counterfactuals, axs[0][0])
    axs[0][0].set_title("(1)  Counterfactuals ordered by DiCE")
    axs[0][0].plot()

    plot_cfs(dataframe, instance, cf_generality, axs[0][1])
    axs[0][1].set_title("(2) Counterfactuals ordered by Generality")
    axs[0][1].plot()

    plot_cfs(dataframe, instance, cf_proximity, axs[1][0])
    axs[1][0].set_title("(3) Counterfactuals ordered by Adjacency")
    axs[1][0].plot()

    plot_cfs(dataframe, instance, cf_abnormality, axs[1][1])
    axs[1][1].set_title("(4) Counterfactuals ordered by Abnormality")
    axs[1][1].plot()

    fig.subplots_adjust(top=0.95, hspace=0.25)
    fig.set_size_inches(w, h)
    plt.savefig('plot_dice_gene_abno_prox.pdf')
    # plt.show()


def plot_dice_obta(plot_cfs_3d, target_class_name, target_class, dataframe, instance, counterfactuals, w=15, h=6):

    ordering_features_iris_3d = {"petal length": ['v', 's', 'p']}

    metrics = ['abnormality', 'proximity', 'generality', 'obtainability']
    cf_metrics, _, _ = apply.apply_all_metrics(dataframe, instance, counterfactuals, target_class, target_class_name, ordering_features_iris_3d)
    cf_obtainability = cf_metrics.sort_values(by=['obtainability']).reset_index().drop(metrics, axis="columns")
    cf_metrics = cf_metrics.drop(metrics, axis="columns")

    fig, (ax1, ax2) = plt.subplots(1,2)

    plot_cfs_3d(dataframe, instance, cf_metrics, ax1)
    ax1.set_title("(1) Counterfactuals ordered by DiCE")
    ax1.plot()

    plot_cfs_3d(dataframe, instance, cf_obtainability, ax2)
    ax2.set_title("(2) Counterfactuals ordered by Obtainability")
    ax2.plot()

    fig.set_size_inches(w, h)
    plt.savefig('plot_dice_obta.pdf')

    # plt.show()


def plot_dice_pareto(plot_cfs_3d, target_class_name, target_class, dataframe, instance, counterfactuals, w=15, h=6):

    ordering_features_iris_3d = {"petal length": ['v', 's', 'p']}

    metrics = ['abnormality', 'proximity', 'generality', 'obtainability']
    cf_metrics, cf_pareto, _ = apply.apply_all_metrics(dataframe, instance, counterfactuals, target_class, target_class_name, ordering_features_iris_3d)
    cf_metrics = cf_metrics.drop(metrics, axis="columns")
    
    fig, (ax1, ax2) = plt.subplots(1,2)

    plot_cfs_3d(dataframe, instance, cf_metrics, ax1)
    ax1.set_title("(1) Counterfactuals ordered by DiCE")
    ax1.plot()

    plot_cfs_3d(dataframe, instance, cf_pareto, ax2)
    ax2.set_title("(2) Counterfactuals optimized")
    ax2.plot()

    fig.set_size_inches(w, h)
    plt.savefig('plot_dice_pareto.pdf')
    # plt.show()


target_class_name = 'label'
target_class = 1 # 0:blue and 1:orange
sns.set_context("notebook", font_scale=1.25)

path_2d = r"json_iris\iris_cfs_used.json"
dataframe_2d, instance_2d, counterfactuals_2d = get_df_data(path_2d)
plot_dice_gene_abno_prox(plot_cfs, target_class_name, target_class, dataframe_2d, instance_2d, counterfactuals_2d)

path_3d = r"json_iris\iris_cfs_3d_used.json"
dataframe_3d, instance_3d, counterfactuals_3d = get_df_data(path_3d, path_train=r"datasets\toy_dataset_iris_3d.csv")
plot_dice_obta(plot_cfs_3d, target_class_name, target_class, dataframe_3d, instance_3d, counterfactuals_3d)
# plot_dice_pareto(plot_cfs_3d, target_class_name, target_class, dataframe_3d, instance_3d, counterfactuals_3d)