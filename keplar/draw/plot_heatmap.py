import os
import matplotlib
from matplotlib import pyplot as plt
import seaborn as sns
import pandas as pd
import numpy as np
from scipy.stats import wilcoxon
import itertools as it

alpha = 0.05


def save(name='tmp', h=None):
    name = name.strip().replace(' ', '-').replace('%', 'pct')
    if h == None:
        h = plt.gcf()
    h.tight_layout()
    print('saving', name + '.pdf')
    plt.savefig(name + '.pdf', bbox_inches='tight')


def pairwise_pval(df, metric, alg1, alg2):
    df = df.loc[df.algorithm.isin([alg1, alg2]), :].copy()
    x = df.loc[df.algorithm == alg1, metric].values
    y = df.loc[df.algorithm == alg2, metric].values
    if np.nanmedian(y) == 0:
        eff_size = 1
    else:
        eff_size = np.abs(np.nanmedian(x) / np.nanmedian(y))
    if metric.endswith('norm'):
        rmetric = metric.replace('norm', 'rank')
    else:
        rmetric = metric + '_rank'
    if rmetric not in df.columns:
        rmetric = metric

    x_rank = df.loc[df.algorithm == alg1, rmetric].values
    y_rank = df.loc[df.algorithm == alg2, rmetric].values
    #     pdb.set_trace()
    if len(x) != len(y):
        # print(alg1, len(x))
        # print(alg2, len(y))
        # print(metric)

        print(alg1, alg2)
        min_len = min(len(x), len(y))
        x = x[:min_len]
        y = y[:min_len]
        if min_len == 0:
            print(alg1, len(x))
            print(alg2, len(y))
    assert len(x) == len(y)
    #     w, p = mannwhitneyu(x, y)
    if all(y == 0) and all(x == 0):
        return 1, 1

    w, p = wilcoxon(x, y)
    return p, eff_size


#     return pstr, eff_size_str

def bin_pval(x, c_alpha):
    for stars, level in zip([4, 3, 2, 1], [1e-3, 1e-2, 1e-1, 1]):
        if x < level * c_alpha:
            return stars  # level #*c_alpha
    return 0


def signif(pval, alpha, eff):
    pstr = '{:1.2g}'.format(pval)
    eff_size_str = '{:1.1f}X'.format(eff)
    if pval == '-': return pval
    if float(pval) < alpha:
        return 'textbf{' + pstr + '}', 'textbf{' + eff_size_str + '}'
    #         return pval+'*'
    else:
        return pstr, eff_size_str


def get_pval_df(df, metric, all_algs):
    df = df.copy()
    n = 0
    pvals = []
    # for alg1, alg2 in it.combinations(all_algs, 2):
    for i in range(len(all_algs)):
        for j in range(len(all_algs)):
            if j == i:
                continue
            alg1 = all_algs[i]
            alg2 = all_algs[j]
            pval, eff_size = pairwise_pval(df, metric, alg1, alg2)
            pvals.append(dict(
                alg1=alg1,
                alg2=alg2,
                eff_size=eff_size,
                pval=pval
            ))
            n += 1
    c_alpha = alpha / n

    print('n:', n, 'c_alpha:', c_alpha)
    df_pvals = pd.DataFrame.from_records(pvals)
    # df_pvals['pval_thresh'] = pd.cut(x=df_pvals['pval'], bins = [0, c_alpha, 1])
    df_pvals['pval_thresh'] = df_pvals['pval'].apply(lambda x: bin_pval(x, c_alpha))

    # significance
    df_pvals.loc[:, 'pval_bold'] = df_pvals.apply(lambda x: signif(x['pval'],
                                                                   c_alpha,
                                                                   x['eff_size'])[0],
                                                  axis=1
                                                  )
    df_pvals.loc[:, 'eff_size_bold'] = df_pvals.apply(lambda x: signif(x['pval'],
                                                                       c_alpha,
                                                                       x['eff_size'])[1],
                                                      axis=1
                                                      )
    return df_pvals, c_alpha


def pval_heatmap(df, metric, problem, algs):
    df = df.copy()
    n = 0
    pvals = []

    df_pvals, c_alpha = get_pval_df(df, metric, algs)

    #                                                  pd.cut(x=df_pvals['pval'], bins = [0, c_alpha, 1])
    tbl = df_pvals.set_index(['alg1', 'alg2'])['pval_thresh'].unstack().transpose()  # .fillna('-')

    mask = np.zeros_like(tbl, dtype=np.bool_)
    mask[np.triu_indices_from(mask, k=1)] = True

    h = plt.figure(figsize=(10, 10))

    cmap = sns.color_palette('flare', n_colors=5)
    cmap[0] = [.9, .9, .9]
    ax = sns.heatmap(tbl,
                     linewidth=0.25,
                     mask=mask,
                     square=True,
                     cbar_kws=dict(
                         ticks=[0.4, 1.2, 2.0, 2.8, 3.6],
                         shrink=0.6,
                     ),
                     cmap=cmap,

                     )
    cax = h.axes[-1]
    cbar_labels = [
        'no significance',
        '$p<\\alpha$',
        '$p<$1e-1$\cdot \\alpha$',
        '$p<$1e-2$\cdot \\alpha$',
        '$p<$1e-3$\cdot \\alpha$',
    ]
    cax.set_yticklabels(cbar_labels)
    nice_metric = metric.replace('%', 'pct').replace('_', ' ').replace('R2', '$$R^2$$').title()
    plt.title(('Wilcoxon signed-rank test, '
               + nice_metric
               + ', $\\alpha =$ {:1.1e}').format(c_alpha)
              )
    plt.xlabel('')
    plt.ylabel('')
    savename = ('Pairwise comparison of '
                + nice_metric
                + ' on '
                + problem).replace(' ', '_')
    save(savename, h)


rename_algos = {
    "dsr": "DSR",
    "udsr": "uDSR",
    "eql": "EQL"
}
our_impl = ["DSR", "uDSR", "ddsr", "DDSR-NN(ours)", "EQL", "gplearn","mtaylor","taylor"]
df_sum = pd.read_feather("feynman_results.feather")
# remove
df_sum = df_sum.loc[df_sum['algorithm'] != "DSR"]
# rename
df_sum['algorithm'] = df_sum['algorithm'].apply(lambda x: rename_algos[x] if x in rename_algos else x)
# add implementation
df_sum['algorithm'] = df_sum['algorithm'].apply(lambda x: x if x in our_impl else "*" + x)

algs = df_sum["algorithm"].unique()
print(algs)
# cols = df_sum.columns
# for col in [c for c in cols if 'r2_test' in c]:
#     df_sum.loc[:,col] = df_sum[col].fillna(len(algs)+1)
# for col in [c for c in cols if 'solution_rate' in c]:
#     df_sum.loc[:,col] = df_sum[col].fillna(0.0)


for metric in ['r2_test', 'model_size']:
    name = 'symbolic problems'
    pval_heatmap(df_sum, metric, name, algs)
