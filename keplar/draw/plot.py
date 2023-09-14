import os
import matplotlib
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import json
import numpy as np
from glob import glob
from matplotlib.pyplot import FixedLocator

matplotlib.rc('pdf', fonttype=42)
sns.set(font_scale=1.25, style="whitegrid")

sns.set_style("whitegrid")

dataset = "pmlb"
feather = f"{dataset}_results.feather"

show_algos = [
    # "AdaBoost", 
    "AIFeynman",
    # "MLP",
    "Linear",
    # "XGB",
    "gplearn",
    "DSR",
    "EQL",
    "ddsr"
]


def save(name='tmp', h=None):
    stype = "pdf"
    name = name.strip().replace(' ', '-').replace('%', 'pct')
    if h == None:
        h = plt.gcf()
    h.tight_layout()
    print('saving', name + f'.{stype}')
    plt.savefig(name + f'.{stype}', bbox_inches='tight')
    # plt.savefig(name + '.png')


def plot_comparison(df_compare, x='model_size', y='algorithm', row=None, col=None, scale=None, xlim=[], **kwargs):
    # filter
    df_compare = df_compare.loc[df_compare["algorithm"] != "gplearn"]

    plt.figure()
    order = df_compare.groupby(y)[x].median().sort_values(ascending=False if x == "r2_test" else True).index
    if scale == 'log' and len(xlim) > 0 and xlim[0] == 0:
        df_compare.loc[:, x] += 1
        xlim[0] = 1
        xnew = '1 + ' + x
        df_compare = df_compare.rename(columns={x: xnew})
        x = xnew

    sns.catplot(data=df_compare,
                kind='box',
                #                 color='w',
                orient="v",
                y=x,
                x=y,
                order=order,
                fliersize=0,
                #                 notch=True,
                #  height=6.5, 
                #  aspect=0.6,
                row=row,
                col=col,
                palette='flare_r',
                **kwargs
                )
    plt.ylabel('')
    plt.xlabel('')
    if len(xlim) > 0:
        plt.xlim(xlim[0], xlim[1])
    plt.ylim(-1, 1)
    if scale:
        plt.gca().set_xscale(scale)

    save(name='_'.join(['boxplot', x + '-by-' + y]))
    if col:
        save(name='_'.join(['boxplot', x + '-by-' + y] + [col]))


def plot_dataset():
    data_dir = "/home/luoyuanzhen/Datasets/"
    with open("../scripts/benchmark.json", "r") as fp:
        benchmark = json.load(fp)
    feynmans = benchmark["feynman"]["datasets"]
    pmlbs = benchmark["pmlb"]["datasets"]

    frames = []
    for feynman in feynmans:
        df = np.loadtxt(os.path.join(data_dir, "feynman/train", feynman + ".txt"))
        frames.append(dict(
            name=feynman,
            nsamples=10000,
            nfeatures=df.shape[1] - 1,
            npoints=10000 * (df.shape[1] - 1),
            Group="Physics"
        ))

    for pmlb in pmlbs:
        df = np.loadtxt(os.path.join(data_dir, "origin-pmlb", pmlb + ".txt"))
        frames.append(dict(
            name=feynman,
            nsamples=df.shape[0],
            nfeatures=df.shape[1] - 1,
            npoints=df.shape[0] * (df.shape[1] - 1),
            Group="Real"
        ))

    df = pd.DataFrame.from_records(frames)
    sns.despine(left=True, bottom=True)
    ## PMLB dataset sizes
    g = sns.scatterplot(
        data=df,
        x='nfeatures',
        y='nsamples',
        hue='Group',
        alpha=0.7,
        s=100,
    )
    ax = plt.gca()
    plt.legend(loc='upper right')
    ax.set_xscale('log')
    # ax.set_yscale('log')
    plt.xlabel('No. of Features')
    plt.ylabel('No. of Samples')
    plt.savefig('benchmark.pdf', dpi=400, bbox_inches='tight')


# plot_dataset()
# exit()


df_plot = pd.read_feather(feather)
# print(df_plot["algorithm"].unique())
# exit()

# if dataset == "feynman":
#     x = "r2_test"
#     plot_comparison(df_plot, x=x, scale='log' if x == "model_size" else None)
#     exit()

rename_algos = {
    "dsr": "DSR",
    "udsr": "uDSR",
    "eql": "EQL"
}
our_impl = ["DSR", "uDSR", "ddsr", "DDSR-NN(ours)", "EQL", "gplearn", "mtaylor","taylor"]
print(df_plot.keys())
# remove excluded
df_plot = df_plot.loc[df_plot['algorithm'] != "DSR"]
# rename
df_plot['*algorithm*'] = df_plot['algorithm'].apply(lambda x: rename_algos[x] if x in rename_algos else x)
# add implementation
df_plot['*algorithm*'] = df_plot['*algorithm*'].apply(lambda x: x if x in our_impl else "*" + x)
# different options
x_vars = [
    #         'rmse_test',
    #         'log_mse_test',
    #         'r2_test_norm',
    'r2_test',
    #         'r2_test_rank',
    'model_size',
    #         'model_size_rank',
    # 'training time (s)',
]
eql = df_plot.loc[df_plot["*algorithm*"] == "EQL"]
print(eql.model_size.tolist())
# fix eql
df_plot.loc[df_plot["*algorithm*"] == "EQL", "model_size"] = eql["model_size"].apply(lambda x: 100 if x == 0 else x)
# df_plot.dropna(axis=0, how="any", inplace=True)
df_plot.loc[df_plot['*algorithm*'] == "EQL", "r2_test"] = -.1
# remove inf or nan
df_plot["r2_test"] = df_plot["r2_test"].apply(lambda x: -1 if np.isnan(x) or np.isinf(x) or x < -1 else x)
# # df_plot["r2_test"] = df_plot["r2_test"].apply(lambda x: 0.9 if x >= 1 else x)
# # df_plot["r2_test"] = df_plot["r2_test"].apply(lambda x: 0 if x < 0 else x)
df_plot["model_size"] = df_plot["model_size"].apply(lambda x: 100 if np.isnan(x) or np.isinf(x) else x)

order = df_plot.groupby('*algorithm*')[x_vars[0]].mean().sort_values(
    ascending='r2' not in x_vars[0] or 'rank' in x_vars[0]).index

x_vars = ["model_size"]
f, ax = plt.subplots(figsize=(7, 6))
if "size" in x_vars[0]:
    ax.set_xscale("log")


def box_plot():
    # eql feynman-1 seed=2 r2
    # eql feynman-1 seed=3 r2
    # dsr feynman-1 ssee=1 r2
    # sns.boxplot(x="r2_test", y="*algorithm*", data=df_plot, order=order, width=0.8, palette="vlag")
    sns.boxplot(x=x_vars[0], y="*algorithm*", data=df_plot, order=order, width=.8, palette="vlag")
    sns.stripplot(x=x_vars[0], y="*algorithm*", data=df_plot, order=order,
                  size=3, color=".3", linewidth=0)
    if "r2_test" in x_vars[0]:
        ax.set_xlim([-.1, 1.02])
    if "size" in x_vars[0]:
        ax.set_xticks([1, 100, 1e4, 1e6], ["$10^1$", "$10^2$", "$10^3$", "$10^4$"])


box_plot()
ax.yaxis.grid(True)

plt.ylabel("")
plt.xlabel("")
# g = sns.PairGrid(df_plot, 
#                  x_vars=x_vars,
#                  y_vars=['*algorithm*'],
#                  height=6.5, 
#                  aspect=0.6,
# #                  hue='symbolic_dataset'
#                 )

# g.map(
#     sns.boxplot,
#     orient="h",
# )
# sns.boxplot(x="r2_test", y="*algorithm*", data=df_plot, width=.6, palette="vlag", ax=g.axes.flat[0])
# sns.boxplot(x="model_size", y="*algorithm*", data=df_plot, width=.6, palette="vlag", ax=g.axes.flat[1])
# g.boxplot(x="r2_test", y="*algorithm*", data=df_plot, width=.6, palette="vlag")
# g.map_diag(sns.boxplot, width=.6, palette="vlag")
# g.map_offdiag(sns.boxplot, width=.6, palette="vlag")

# Draw a dot plot 
# g.map(sns.pointplot,
#     #   kind="box",
#     #   size=10,
#       orient="h",
#     #   jitter=False,
#       order=order,
#     # #   size=4,
#     #   width=.6
#       palette="flare_r",
#     #     cut=0,
#     #     bw=0.5,
#         # scale="count",
#       linewidth=1,
#     #   markeredgecolor='w',
#     #   errorbar="ci",
#     #   join=False,
#     #   estimator=np.mean,
#     #   n_boot=1000,
#     #   ci=95,
#       clip_on=False
#     )
# # Use semantically meaningful titles for the columns
# titles = [x.replace('_',' ').title().replace('(S)','(s)').replace('R2','$R^2$') for x in x_vars]
# g.axes.flat[0].set_ylabel('')
# g.axes.flat[0].set_xlim(0.39, 1.02)
# g.axes.flat[0].set_xticks([0.4, 0.6, 0.8, 1.0], ["0", "0.6", "0.8", "1.0"])
# for ax, title in zip(g.axes.flat, titles):

#     # Set a different title for each axes
#     ax.set(title=title)
#     ax.set_xlabel('')

#     if any([n in title.lower() for n in ['size','time']]):
#         ax.set_xscale('log')

#     # if title == '$R^2$ Test':
#     #     ax.set_xlim([.5,1])

#     # Make the grid horizontal instead of vertical
#     ax.yaxis.grid(True)
# g.axes.flat[1].set_xticks([1, 100, 10000], ["10", "100", "1000"])
sns.despine(left=True, bottom=True)

save(name='_'.join([f'{dataset}-pairgrid-pointplot'] + x_vars))

# # show_df = []
# # for algo in show_algos:
# #     algo_df = df.loc[df['algorithm'] == algo]
# #     algo_df.drop(columns=['level_0'], inplace=True)
# #     # print(algo_df.keys())
# #     show_df.append(algo_df)
# # show_df = pd.concat(show_df).reset_index()
# x = "r2_test"
# xlim = [-0.25, 1]
# plot_comparison(df, x=x, xlim=xlim, scale='log' if x == "model_size" else None)
