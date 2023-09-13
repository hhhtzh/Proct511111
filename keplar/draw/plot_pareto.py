import pandas as pd
import numpy as np
import matplotlib
import matplotlib.pyplot as plt
import seaborn as sns
import pdb

matplotlib.rc('pdf', fonttype=42)
sns.set(font_scale=1, style="whitegrid")

dataset = "feynman"
feather = f"{dataset}_results.feather"


def check_dominance(p1, p2):
    flag1 = 0
    flag2 = 0

    for o1, o2 in zip(p1, p2):
        if o1 < o2:
            flag1 = 1
        elif o1 > o2:
            flag2 = 1

    if flag1 == 1 and flag2 == 0:
        return 1
    elif flag1 == 0 and flag2 == 1:
        return -1
    else:
        return 0


def front(obj1, obj2):
    """return indices from x and y that are on the Pareto front."""
    rank = []
    assert (len(obj1) == len(obj2))
    n_inds = len(obj1)
    front = []

    for i in np.arange(n_inds):
        p = (obj1[i], obj2[i])
        dcount = 0
        dom = []
        for j in np.arange(n_inds):
            q = (obj1[j], obj2[j])
            compare = check_dominance(p, q)
            if compare == 1:
                dom.append(j)
            #                 print(p,'dominates',q)
            elif compare == -1:
                dcount = dcount + 1
        #                 print(p,'dominated by',q)

        if dcount == 0:
            #             print(p,'is on the front')
            front.append(i)

    #     f_obj1 = [obj1[f] for f in front]
    f_obj2 = [obj2[f] for f in front]
    #     s1 = np.argsort(np.array(f_obj1))
    s2 = np.argsort(np.array(f_obj2))
    #     front = [front[s] for s in s1]
    front = [front[s] for s in s2]

    return front


def save(name='tmp', h=None):
    name = name.strip().replace(' ', '-').replace('%', 'pct')
    if h == None:
        h = plt.gcf()
    h.tight_layout()
    print('saving', name + '.pdf')
    plt.savefig(name + '.pdf', bbox_inches='tight')


def bootstrap(val, n=1000, fn=np.mean):
    val_samples = []
    for i in range(n):
        sample = np.random.randint(0, len(val) - 1, size=len(val))
        val_samples.append(fn(val[sample]))
    m = np.mean(val_samples)
    sd = np.std(val_samples)
    ci_upper = np.quantile(val_samples, 0.95)
    ci_lower = np.quantile(val_samples, 0.05)
    return m, sd, ci_upper, ci_lower


labelsize = 18
plt.figure(figsize=(7, 7))
df_plot = pd.read_feather(feather)
print(df_plot.keys())

rename_algos = {
    "dsr": "DSR",
    "udsr": "uDSR"
}
our_impl = ["DSR", "uDSR", "ddsr", "DDSR-NN(ours)", "EQL", "gplearn","mtaylor"]
print(df_plot.keys())

df_plot = df_plot.loc[df_plot["algorithm"] != "eql"].reset_index(drop=True)
df_plot.loc[df_plot["algorithm"] == 'EQL', "model_size"] = float("inf")
print("df_plot", df_plot.keys())

df_results2 = df_plot.merge(df_plot.groupby('dataset')['algorithm'].nunique().reset_index(),
                            on='dataset', suffixes=('', '_count'))

# rankings per trial per dataset
for col in [c for c in df_results2.columns if c.endswith('test') or c.endswith('size')]:
    ascending = 'r2' not in col
    df_results2[col + '_rank_per_trial'] = df_results2.groupby(['dataset', 'random_state'])[col].apply(
        lambda x:
        round(x, 3).rank(
            ascending=ascending))
print("df_result2", df_results2.keys())

df_sum = df_results2.groupby(['algorithm', 'dataset'], as_index=False).median()
df_sum['rmse_test'] = df_sum['mse_test'].apply(np.sqrt)
df_sum['log_mse_test'] = df_sum['mse_test'].apply(lambda x: np.log(1 + x))
df_results = df_results2

print("df_sum", df_sum.keys())

# rankings and normalized scores per dataset
for col in [c for c in df_sum.columns if c.endswith('test') or c.endswith('size')]:
    ascending = 'r2' not in col
    df_sum[col + '_rank'] = df_sum.groupby(['dataset'])[col].apply(lambda x:
                                                                   round(x, 3).rank(ascending=ascending)
                                                                   )
    df_sum[col + '_norm'] = df_sum.groupby('dataset')[col].apply(lambda x: (x - x.min()) / (x.max() - x.min()))

print(df_sum.keys())

# remove excluded
df_sum = df_sum.loc[df_sum['algorithm'] != "DSR"]
# rename
df_sum['*algorithm*'] = df_sum['algorithm'].apply(lambda x: rename_algos[x] if x in rename_algos else x)
# add implementation
print(df_sum['*algorithm*'])
df_sum['*algorithm*'] = df_sum['*algorithm*'].apply(lambda x: x if x in our_impl else "*" + x)

data = df_sum.copy()  # .loc[df_sum.algorithm.isin(symbolic_algs)]

xcol = 'r2_test_rank'
# xcol  = 'r2_test'
ycol = 'model_size_rank'
# ycol = 'model_size'
palette = 'viridis'
# outline pareto front
pareto_data = data.groupby('*algorithm*').median()

objs = pareto_data[[xcol, ycol]].values
# reverse R2 (objs are minimized)
# objs[:,0] = -objs[:,0]
levels = 6  # 9
styles = ['-', '-.', '--', ':', ':', ':']
PFs = []
pareto_ranks = -np.ones(len(pareto_data))
for el in range(levels):
    #     pdb.set_trace()
    PF = front(objs[:, 0], objs[:, 1])
    if len(PF) > 0:
        print('PF:', PF)
        pareto_ranks[PF] = el
    objs[PF, :] = np.inf
    PFs.append(PF)
i = 0
pareto_data.loc[:, 'pareto_rank'] = pareto_ranks
# print(pareto_data.loc[:, 'pareto_rank'])
for pfset in PFs:
    xset, yset = [], []

    for pf in pfset:
        xset.append(pareto_data[xcol].values[pf])
        yset.append(pareto_data[ycol].values[pf])
    linestyle = styles[i]
    plt.plot(xset, yset, styles[i] + 'k', alpha=0.5, zorder=1)
    #     plt.gca().set_zorder(10)
    i += 1

cmap = sns.color_palette(palette=palette,
                         n_colors=pareto_data.pareto_rank.nunique(),
                         desat=None,
                         as_cmap=False)
# cmap = sns.color_palette(palette=palette,
#                          n_colors=4,
#                          desat=None,
#                          as_cmap=False)

print(pareto_data.pareto_rank.nunique())

ax = sns.scatterplot(
    ax=plt.gca(),
    #     ax = g.ax_joint,
    #     data = data.groupby('*algorithm*').median(),
    data=pareto_data,
    x=xcol,
    y=ycol,
    #     style='*algorithm*',
    #     style='pareto_rank',
    hue='pareto_rank',
    s=250,
    #     palette=palette,
    #     edgecolor='k'
    legend=False,
    palette=cmap
)
ax.set_zorder(2)
xoff = .5
yoff = 0.3
# xoff, yoff = 0, 0
for idx, row in pareto_data.iterrows():
    x = row[xcol] - xoff
    y = row[ycol] - yoff
    ha = 'right'
    # if idx in ['*KernelRidge', '*MRGP', "*FEAT", "*Linear", "*FFX", "*AdaBoost", "*BSR", "*ITEA", "gplearn"]:
    #     x = row[xcol] + 3
    # elif idx == "*AIFeynman":
    #     y = row[ycol] + 0.5
    # elif idx == "uDSR":
    #     x += 0.3
    #     y -= 0.3
    # elif idx == "DSR":
    #     x += 2.8
    #     y += 0.5
    # elif idx == "DDSR-NN(ours)":
    #     y -= 0.5
    #     ha = 'left'
    # elif idx == "*ITEA":
    #     x = row[xcol] + 2
    #     y = row[ycol] + 0.5
    # elif idx == '*GP-GOMEA':
    #     x = row[xcol] + 6
    # elif idx == "gplearn":
    #     x = row[xcol] + 2
    #     y = row[ycol] + 0.5

    # elif idx in ["*EPLEX"]:
    #     y = row[ycol] + 0.3
    # elif idx == "uDSR":
    #     y = row[ycol] - 1
    # elif idx == "*AIFeynman":
    #     y = row[ycol] - 1
    #     x = row[xcol] + 2
    # elif idx == "DDSR-NN(ours)":
    #     y = row[ycol] + 1.5

    if idx in ['*MLP', '*MRGP']:
        x = row[xcol] + xoff
        ha = 'left'
    elif idx == "*BSR":
        x = row[xcol] + 1.5
    elif idx == '*AFP_FE':
        x = row[xcol] + 1.5
        y = row[ycol] + 1
    elif idx == "*ITEA":
        x = row[xcol] + 2
    elif idx == '*Operon':
        x = row[xcol] + 1
        y -= yoff
    elif idx in ['gplearn', '*FEAT']:
        #         x -= xoff
        #         x=row[xcol]+xoff
        y = row[ycol] + 1
    #         ha='left'
    elif idx == "uDSR":
        x = row[xcol] + 2.5
        y = row[ycol] - 0.3
        ha = "right"
    elif idx == "DSR":
        y = row[ycol] + 1

    plt.text(s=idx,
             x=x,
             y=y,
             ha=ha,
             va='top',
             bbox=dict(facecolor='w', edgecolor='b', boxstyle='round', alpha=1)
             )

# confidence intervals
i = 0
for alg, dg in data.groupby('*algorithm*'):
    x = dg[xcol].median()
    y = dg[ycol].median()
    _, sdx, ciux, cilx = bootstrap(dg[xcol].values, fn=np.median, n=1000)
    _, sdy, ciuy, cily = bootstrap(dg[ycol].values, fn=np.median, n=1000)
    # print(len(cmap))
    # print(int(pareto_data.loc[alg, 'pareto_rank']))

    # plt.plot(
    #     [cilx, ciux],
    #     [y, y],
    #     alpha=0.5,
    #     color=cmap[int(pareto_data.loc[alg, 'pareto_rank'])]
    #     #              color='b'
    # )
    # plt.plot(
    #     [x, x],
    #     [cily, ciuy],
    #     alpha=0.5,
    #     color=cmap[int(pareto_data.loc[alg, 'pareto_rank'])]
    #     #              color='b'
    # )
    plt.plot(
        [cilx, ciux],
        [y, y],
        alpha=0.5,
        color=cmap[0]
        #              color='b'
    )
    plt.plot(
        [x, x],
        [cily, ciuy],
        alpha=0.5,
        color=cmap[0]
        #              color='b'
    )
    i += 1
ax.set_aspect(1.0)
ax.set_xlim(left=0)
ax.set_ylim(bottom=0)
ticksize = 16
ticks = [0, 5, 10, 15, 20]
if dataset == "pmlb":
    ticks.extend([25])

plt.xticks(ticks, fontsize=ticksize)
plt.yticks(ticks, fontsize=ticksize)
# ax.set_yscale('log')
plt.xlabel(xcol.replace('_', ' ').replace('r2', '$R^2$').title(), fontsize=labelsize)
plt.ylabel(ycol.replace('_', ' ').title(), fontsize=labelsize)
sns.despine(left=True, bottom=True)
save(name=f"{dataset}_pareto_plot_" + xcol + '_' + ycol)
