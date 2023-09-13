import pandas as pd
import os
import numpy as np
import matplotlib.pyplot as plt


# def draw_box_diagram(fitnessGPTaylor, fitnessGP, fitnessFFX, fitnessLinearSR, fitnessKernelRidge,
#                      fitnessRandomForestRegressor,
#                      fitnessSVM, fitnessXGBoost, fitnessGSGP, fitnessBSR, dataName, isAll: bool):
#     # 设置中文和负号正常显示
#     plt.rcParams['axes.unicode_minus'] = False
#     plt.rcParams['pdf.fonttype'] = 42
#     plt.rcParams['ps.fonttype'] = 42
#     # 设置图形的显示风格
#     # plt.style.use('ggplot')
#     if isAll == True:
#         data = {
#             'TaylorGP': fitnessGPTaylor,
#             'GPLearn': fitnessGP,
#             'FFX': fitnessFFX,
#             'Linear\nRegression': fitnessLinearSR,
#             'KernelRidge': fitnessKernelRidge,
#             'RandomForest\nRegressor': fitnessRandomForestRegressor,
#             'SVM': fitnessSVM,
#             'XGBoost': fitnessXGBoost,
#             'GSGP': fitnessGSGP,
#             'BSR': fitnessBSR
#         }
#     else:
#         data = {
#             'TaylorGP': fitnessGPTaylor,
#             # 'GPLearn': fitnessGP,
#             # 'FFX': fitnessFFX,
#             # 'Linear\nRegression': fitnessLinearSR,
#             # 'KernelRidge': fitnessKernelRidge,
#             # 'RandomForest\nRegressor': fitnessRandomForestRegressor,
#             # 'SVM': fitnessSVM,
#             'XGBoost': fitnessXGBoost,
#             'GSGP': fitnessGSGP,
#             'BSR': fitnessBSR
#         }
#     df = pd.DataFrame(data)
#
#     fig, axes = plt.subplots()
#     # fig, axes = plt.subplots(nrows=1, ncols=2, figsize=(9, 4))
#     bplot1 = axes.boxplot(df,  # 描点上色
#                           # notch=True,  # 箱体中位数凹陷
#                           vert=True,  # 设置箱体垂直
#                           patch_artist=True,  # 允许填充颜色
#                           showfliers=False,  # 不显示异常值
#                           widths=0.8,  # 箱体宽度
#                           medianprops={'color': 'black'})  # 设置中位数线的属性，线的类型和颜色
#     '''
#     bplot2 = axes[1].boxplot(df,
#                              notch=True,#箱体中位数凹陷
#                              vert=True,
#                              patch_artist=True,
#                              showfliers=False,
#                              widths=0.7,
#                              medianprops = {'color':'black'})
#     '''
#
#     # 颜色填充
#     # colors = ['red','lightcoral','gold','yellow','lime','darkseagreen','cyan','deepskyblue'  ]
#     colors = ['red', 'lightcoral', 'gold', 'yellow', 'lime', 'darkseagreen', 'cyan', 'deepskyblue', 'darkorchid',
#               'hotpink']
#     # for bplot in (bplot1):
#     for patch, color in zip(bplot1['boxes'], colors):
#         patch.set_facecolor(color)
#     plt.grid(linestyle="--", alpha=0.3)
#     plt.ylabel("Score(Normalized $R^{2}$)")
#     if isAll == True:
#         plt.xticks(np.arange(11), (
#             '', 'TaylorGP', 'GPLearn', 'FFX', 'Linear\nRegression', 'KernelRidge', 'RandomForest\nRegressor', 'SVM',
#             'XGBoost', 'GSGP', 'BSR'),
#                    fontsize=7, rotation=30)
#     else:
#         plt.xticks(np.arange(5), (
#             '', 'TaylorGP', 'XGBoost', 'GSGP', 'BSR'),
#                    fontsize=7, rotation=30)
#     # plt.xticks(np.arange(5), (
#     #     '', 'TaylorGP', 'XGBoost', 'GSGP', 'BSR'),
#     #            fontsize=7, rotation=30)
#     if isAll == True:
#         figPath = os.path.join('IMG_COLOR', 'LOG')
#     else:
#         figPath = os.path.join('IMG_COLOR', 'LOG_FOUR')
#     plt.savefig(os.path.join(figPath, dataName + '.pdf'), dpi=720, bbox_inches='tight')
#     plt.close()


def draw_box_diagram_vla_2_1(fitnessBingo, fitnessGplearn, fitnessBingoCPP,
                      fitnessKeplarBingoCPP, fitnessKeplarBingo, fitnessOperon,
                      fitnessKeplarMBingo, fitnessKeplarMOperon, fitnessGpBingo,fitnessTaylorBingo,fitnessuDSR,dataName, isAll: bool):
    # 设置中文和负号正常显示
    plt.rcParams['axes.unicode_minus'] = False
    plt.rcParams['pdf.fonttype'] = 42
    plt.rcParams['ps.fonttype'] = 42

    # 设置图形的显示风格
    # plt.style.use('ggplot')
    if isAll == True:
        data = {
            'BingoCPP': fitnessBingoCPP,
            'Operon': fitnessOperon,
            'GpBingo': fitnessGpBingo,
            'Bingo': fitnessBingo,
            'KeplarBingo': fitnessKeplarBingo,
            'KeplarBingoCPP': fitnessKeplarBingoCPP,
            'KeplarMBingo':fitnessKeplarMBingo,
            'KeplarMOperon':fitnessKeplarMOperon,
            'Gplearn': fitnessGplearn,
            'TaylorBingo':fitnessTaylorBingo,
            'uDSR':fitnessuDSR,

        }
    else:
        data = {
            'BingoCPP': fitnessBingoCPP,
            'Operon': fitnessOperon,
            'KeplarBingoCPP': fitnessKeplarBingoCPP
        }
    df = pd.DataFrame(data)
    print(df)
    fig, axes = plt.subplots()
    # fig, axes = plt.subplots(nrows=1, ncols=2, figsize=(9, 4))
    bplot1 = axes.boxplot(df,  # 描点上色
                          # notch=True,  # 箱体中位数凹陷
                          vert=True,  # 设置箱体垂直
                          patch_artist=True,  # 允许填充颜色
                          showfliers=False,  # 不显示异常值
                          widths=0.8,  # 箱体宽度
                          medianprops={'color': 'black'})  # 设置中位数线的属性，线的类型和颜色
    '''
    bplot2 = axes[1].boxplot(df,
                             notch=True,#箱体中位数凹陷
                             vert=True,
                             patch_artist=True,
                             showfliers=False,
                             widths=0.7,
                             medianprops = {'color':'black'})    
    '''

    # 颜色填充
    # colors = ['red','lightcoral','gold','yellow','lime','darkseagreen','cyan','deepskyblue'  ]
    colors = ['red', 'lightcoral', 'gold', 'yellow', 'lime', 'darkseagreen', 'cyan', 'deepskyblue', 'darkorchid',
              'hotpink']
    # for bplot in (bplot1):
    for patch, color in zip(bplot1['boxes'], colors):
        patch.set_facecolor(color)
    plt.grid(linestyle="--", alpha=0.3)
    plt.ylabel("Score( $RMSE$)")
    plt.title("population_size=128\ngeneration=1000")
    if isAll == True:
        plt.xticks(np.arange(12), (
             '','BingoCPP',
            'Operon',
            'GpBingo',
            'Bingo',
            'KeplarBingo',
            'KeplarBingoCPP',
            'KeplarMBingo',
            'KeplarMOperon',
            'Gplearn','TaylorBingo','uDSR'),fontsize=7, rotation=30)
    else:
        plt.xticks(np.arange(5), (
            '', 'MTaylorGp',
            'GPLearn',
            'Operon',
            'UDSR'),
                   fontsize=7, rotation=30)
    # plt.xticks(np.arange(5), (
    #     '', 'TaylorGP', 'XGBoost', 'GSGP', 'BSR'),
    #            fontsize=7, rotation=30)
    if isAll == True:
        figPath = os.path.join('IMG_COLOR', 'LOG')
    else:
        figPath = os.path.join('IMG_COLOR', 'LOG_FOUR')
    plt.savefig(os.path.join(figPath, dataName + '.pdf'), dpi=720, bbox_inches='tight')
    plt.close()


def draw_time_box_diagram_vla_5(fitnessBingo, fitnessGplearn, fitnessBingoCPP,
                      fitnessKeplarBingoCPP, fitnessKeplarBingo, fitnessOperon,
                      fitnessKeplarMBingo, fitnessKeplarMOperon, fitnessGpBingo,fitnessTaylorBingo,fitnessuDSR,dataName, isAll: bool):
    # 设置中文和负号正常显示
    plt.rcParams['axes.unicode_minus'] = False
    plt.rcParams['pdf.fonttype'] = 42
    plt.rcParams['ps.fonttype'] = 42

    # 设置图形的显示风格
    # plt.style.use('ggplot')
    if isAll == True:
        data = {
            'BingoCPP': fitnessBingoCPP,
            'Operon': fitnessOperon,
            'GpBingo': fitnessGpBingo,
            'Bingo': fitnessBingo,
            'KeplarBingo': fitnessKeplarBingo,
            'KeplarBingoCPP': fitnessKeplarBingoCPP,
            'KeplarMBingo':fitnessKeplarMBingo,
            'KeplarMOperon': fitnessKeplarMOperon,
            'Gplearn':fitnessGplearn,
            'TaylorBingo': fitnessTaylorBingo,
            'uDSR': fitnessuDSR,

        }
    else:
        data = {
            'BingoCPP': fitnessBingoCPP,
            'Operon': fitnessOperon,
            'KeplarBingoCPP': fitnessKeplarBingoCPP
        }
    df = pd.DataFrame(data)
    fig, axes = plt.subplots()
    # fig, axes = plt.subplots(nrows=1, ncols=2, figsize=(9, 4))
    bplot1 = axes.boxplot(df,  # 描点上色
                          # notch=True,  # 箱体中位数凹陷
                          vert=True,  # 设置箱体垂直
                          patch_artist=True,  # 允许填充颜色
                          showfliers=False,  # 不显示异常值
                          widths=0.8,  # 箱体宽度
                          medianprops={'color': 'black'})  # 设置中位数线的属性，线的类型和颜色
    '''
    bplot2 = axes[1].boxplot(df,
                             notch=True,#箱体中位数凹陷
                             vert=True,
                             patch_artist=True,
                             showfliers=False,
                             widths=0.7,
                             medianprops = {'color':'black'})    
    '''

    # 颜色填充
    # colors = ['red','lightcoral','gold','yellow','lime','darkseagreen','cyan','deepskyblue'  ]
    colors = ['red', 'lightcoral', 'gold', 'yellow', 'lime', 'darkseagreen', 'cyan', 'deepskyblue', 'darkorchid',
              'hotpink']
    # for bplot in (bplot1):
    for patch, color in zip(bplot1['boxes'], colors):
        patch.set_facecolor(color)
    plt.grid(linestyle="--", alpha=0.3)
    plt.ylabel("time( $s$)")
    plt.title("population_size=128\ngeneration=1000")
    if isAll == True:
        plt.xticks(np.arange(12), (
             '','BingoCPP',
            'Operon',
            'GpBingo',
            'Bingo',
            'KeplarBingo',
            'KeplarBingoCPP',
            'KeplarMBingo',
            'KeplarMOperon',
            'Gplearn','TaylorBingo','uDSR'),fontsize=7, rotation=30)
    else:
        plt.xticks(np.arange(5), (
            '', 'MTaylorGp',
            'GPLearn',
            'Operon',
            'UDSR'),
                   fontsize=7, rotation=30)
    # plt.xticks(np.arange(5), (
    #     '', 'TaylorGP', 'XGBoost', 'GSGP', 'BSR'),
    #            fontsize=7, rotation=30)
    if isAll == True:
        figPath = os.path.join('IMG_COLOR', 'LOG')
    else:
        figPath = os.path.join('IMG_COLOR', 'LOG_FOUR')
    plt.savefig(os.path.join(figPath, dataName + 'time'+'.pdf'), dpi=720, bbox_inches='tight')
    plt.close()


# def draw_box_diagram_inside(fitnessGPTaylor, fitnessGP, fitnessFFX, fitnessLinearSR, fitnessKernelRidge,
#                             fitnessRandomForestRegressor,
#                             fitnessSVM, fitnessXGBoost, fitnessGSGP, fitnessBSR, dataName):
#     # 设置中文和负号正常显示
#     plt.rcParams['axes.unicode_minus'] = False
#     plt.rcParams['pdf.fonttype'] = 42
#     plt.rcParams['ps.fonttype'] = 42
#     # 设置图形的显示风格
#     # plt.style.use('ggplot')
#     data = {
#         'TaylorGP': fitnessGPTaylor,
#         'GPLearn': fitnessGP,
#         'FFX': fitnessFFX,
#         'Linear\nRegression': fitnessLinearSR,
#         'KernelRidge': fitnessKernelRidge,
#         'RandomForest\nRegressor': fitnessRandomForestRegressor,
#         'SVM': fitnessSVM,
#         'XGBoost': fitnessXGBoost,
#         'GSGP': fitnessGSGP,
#         'BSR': fitnessBSR
#     }
#     df = pd.DataFrame(data)
#
#     fig, axes = plt.subplots()
#     # fig, axes = plt.subplots(nrows=1, ncols=2, figsize=(9, 4))
#     bplot1 = axes.boxplot(df,  # 描点上色
#                           notch=True,  # 箱体中位数凹陷
#                           vert=True,  # 设置箱体垂直
#                           patch_artist=True,  # 允许填充颜色
#                           showfliers=False,  # 不显示异常值
#                           widths=0.8,  # 箱体宽度
#                           medianprops={'color': 'black'})  # 设置中位数线的属性，线的类型和颜色
#     '''
#     bplot2 = axes[1].boxplot(df,
#                              notch=True,#箱体中位数凹陷
#                              vert=True,
#                              patch_artist=True,
#                              showfliers=False,
#                              widths=0.7,
#                              medianprops = {'color':'black'})
#     '''
#
#     # 颜色填充
#     # colors = ['red','lightcoral','gold','yellow','lime','darkseagreen','cyan','deepskyblue'  ]
#     colors = ['red', 'lightcoral', 'gold', 'yellow', 'lime', 'darkseagreen', 'cyan', 'deepskyblue', 'darkorchid',
#               'hotpink']
#     # for bplot in (bplot1):
#     for patch, color in zip(bplot1['boxes'], colors):
#         patch.set_facecolor(color)
#     plt.grid(linestyle="--", alpha=0.3)
#     plt.ylabel("Score(Normalized $R^{2}$)")
#     plt.xticks(np.arange(11), (
#         '', 'TaylorGP', 'GPLearn', 'FFX', 'Linear\nRegression', 'KernelRidge', 'RandomForest\nRegressor', 'SVM',
#         'XGBoost', 'GSGP', 'BSR'),
#                fontsize=7, rotation=30)
#     # plt.xticks(np.arange(5), (
#     #     '', 'TaylorGP', 'XGBoost', 'GSGP', 'BSR'),
#     #            fontsize=7, rotation=30)
#
#     figPath = os.path.join('IMG_COLOR', 'inside')
#     plt.savefig(os.path.join(figPath, dataName + '.pdf'), dpi=720, bbox_inches='tight')
#     plt.close()


def draw_box_diagram_contact(fitnessGPTaylor, fitnessGP, fitnessFFX, fitnessLinearSR, fitnessKernelRidge,
                             fitnessRandomForestRegressor,
                             fitnessSVM, fitnessXGBoost, fitnessGSGP, fitnessBSR, isAll: bool):
    # 设置中文和负号正常显示
    plt.rcParams['axes.unicode_minus'] = False
    plt.rcParams['pdf.fonttype'] = 42
    plt.rcParams['ps.fonttype'] = 42

    # 设置图形的显示风格
    # plt.style.use('ggplot')
    if isAll == True:
        data = {
            'TaylorGP': fitnessGPTaylor,
            'GPLearn': fitnessGP,
            'FFX': fitnessFFX,
            'Linear\nRegression': fitnessLinearSR,
            'KernelRidge': fitnessKernelRidge,
            'RandomForest\nRegressor': fitnessRandomForestRegressor,
            'SVM': fitnessSVM,
            'XGBoost': fitnessXGBoost,
            'GSGP': fitnessGSGP,
            'BSR': fitnessBSR
        }
    else:
        data = {
            'TaylorGP': fitnessGPTaylor,
            # 'GPLearn': fitnessGP,
            # 'FFX': fitnessFFX,
            # 'Linear\nRegression': fitnessLinearSR,
            # 'KernelRidge': fitnessKernelRidge,
            # 'RandomForest\nRegressor': fitnessRandomForestRegressor,
            # 'SVM': fitnessSVM,
            'XGBoost': fitnessXGBoost,
            'GSGP': fitnessGSGP,
            'BSR': fitnessBSR
        }
    df = pd.DataFrame(data)

    fig, axes = plt.subplots()
    # fig, axes = plt.subplots(nrows=1, ncols=2, figsize=(9, 4))
    bplot1 = axes.boxplot(df,  # 描点上色
                          # notch=True,  # 箱体中位数凹陷
                          vert=True,  # 设置箱体垂直
                          patch_artist=True,  # 允许填充颜色
                          showfliers=False,  # 不显示异常值
                          widths=0.8,  # 箱体宽度
                          medianprops={'color': 'black'})  # 设置中位数线的属性，线的类型和颜色
    '''
    bplot2 = axes[1].boxplot(df,
                             notch=True,#箱体中位数凹陷
                             vert=True,
                             patch_artist=True,
                             showfliers=False,
                             widths=0.7,
                             medianprops = {'color':'black'})    
    '''

    # 颜色填充
    # colors = ['red','lightcoral','gold','yellow','lime','darkseagreen','cyan','deepskyblue'  ]
    colors = ['red', 'lightcoral', 'gold', 'yellow', 'lime', 'darkseagreen', 'cyan', 'deepskyblue', 'darkorchid',
              'hotpink']
    # for bplot in (bplot1):
    for patch, color in zip(bplot1['boxes'], colors):
        patch.set_facecolor(color)
    plt.grid(linestyle="--", alpha=0.3)
    plt.ylabel("Score(Normalized $R^{2}$)")
    if isAll == True:
        plt.xticks(np.arange(11), (
            '', 'TaylorGP', 'GPLearn', 'FFX', 'Linear\nRegression', 'KernelRidge', 'RandomForest\nRegressor', 'SVM',
            'XGBoost', 'GSGP', 'BSR'),
                   fontsize=7, rotation=30)
    else:
        plt.xticks(np.arange(5), (
            '', 'TaylorGP', 'XGBoost', 'GSGP', 'BSR'),
                   fontsize=7, rotation=30)
    # plt.xticks(np.arange(5), (
    #     '', 'TaylorGP', 'XGBoost', 'GSGP', 'BSR'),
    #            fontsize=7, rotation=30)

    if isAll == True:
        figPath = os.path.join('IMG_COLOR', 'LOG', 'contact')
    else:
        figPath = os.path.join('IMG_COLOR', 'LOG_FOUR', 'contact')
    plt.savefig(figPath + '.pdf', dpi=720, bbox_inches='tight')
    plt.close()


# def draw_box_diagram_contact_inside(fitnessGPTaylor, fitnessGP, fitnessFFX, fitnessLinearSR, fitnessKernelRidge,
#                                     fitnessRandomForestRegressor,
#                                     fitnessSVM, fitnessXGBoost, fitnessGSGP, fitnessBSR):
#     # 设置中文和负号正常显示
#     plt.rcParams['axes.unicode_minus'] = False
#     plt.rcParams['pdf.fonttype'] = 42
#     plt.rcParams['ps.fonttype'] = 42
#
#
#     # 设置图形的显示风格
#     # plt.style.use('ggplot')
#     data = {
#         'TaylorGP': fitnessGPTaylor,
#         'GPLearn': fitnessGP,
#         'FFX': fitnessFFX,
#         'Linear\nRegression': fitnessLinearSR,
#         'KernelRidge': fitnessKernelRidge,
#         'RandomForest\nRegressor': fitnessRandomForestRegressor,
#         'SVM': fitnessSVM,
#         'XGBoost': fitnessXGBoost,
#         'GSGP': fitnessGSGP,
#         'BSR': fitnessBSR
#     }
#     df = pd.DataFrame(data)
#
#     fig, axes = plt.subplots()
#     # fig, axes = plt.subplots(nrows=1, ncols=2, figsize=(9, 4))
#     bplot1 = axes.boxplot(df,  # 描点上色
#                           notch=True,  # 箱体中位数凹陷
#                           vert=True,  # 设置箱体垂直
#                           patch_artist=True,  # 允许填充颜色
#                           showfliers=False,  # 不显示异常值
#                           widths=0.8,  # 箱体宽度
#                           medianprops={'color': 'black'})  # 设置中位数线的属性，线的类型和颜色
#     '''
#     bplot2 = axes[1].boxplot(df,
#                              notch=True,#箱体中位数凹陷
#                              vert=True,
#                              patch_artist=True,
#                              showfliers=False,
#                              widths=0.7,
#                              medianprops = {'color':'black'})
#     '''
#
#     # 颜色填充
#     # colors = ['red','lightcoral','gold','yellow','lime','darkseagreen','cyan','deepskyblue'  ]
#     colors = ['red', 'lightcoral', 'gold', 'yellow', 'lime', 'darkseagreen', 'cyan', 'deepskyblue', 'darkorchid',
#               'hotpink']
#     # for bplot in (bplot1):
#     for patch, color in zip(bplot1['boxes'], colors):
#         patch.set_facecolor(color)
#     plt.grid(linestyle="--", alpha=0.3)
#     plt.ylabel("Score(Normalized $R^{2}$)")
#     plt.xticks(np.arange(11), (
#         '', 'TaylorGP', 'GPLearn', 'FFX', 'Linear\nRegression', 'KernelRidge', 'RandomForest\nRegressor', 'SVM',
#         'XGBoost', 'GSGP', 'BSR'),
#                fontsize=7, rotation=30)
#     # plt.xticks(np.arange(5), (
#     #     '', 'TaylorGP', 'XGBoost', 'GSGP', 'BSR'),
#     #            fontsize=7, rotation=30)
#
#     figPath = os.path.join('IMG_COLOR', 'inside', 'contact')
#     plt.savefig(figPath + '.pdf', dpi=720, bbox_inches='tight')
#     plt.close()

def nestPics(fitnessGPTaylor, fitnessGP, fitnessFFX, fitnessLinearSR, fitnessKernelRidge,
             fitnessRandomForestRegressor,
             fitnessSVM, fitnessXGBoost, fitnessGSGP, fitnessBSR, dataName='contact', isLog=True):
    if isLog == True:
        fitnessGPTaylor, fitnessGP, fitnessFFX, fitnessLinearSR, fitnessKernelRidge, fitnessRandomForestRegressor, fitnessSVM, fitnessXGBoost, fitnessGSGP, fitnessBSR = newLog(
            fitnessGPTaylor, fitnessGP, fitnessFFX, fitnessLinearSR, fitnessKernelRidge, fitnessRandomForestRegressor,
            fitnessSVM, fitnessXGBoost, fitnessGSGP, fitnessBSR)
    plt.rcParams['axes.unicode_minus'] = False
    plt.rcParams['pdf.fonttype'] = 42
    plt.rcParams['ps.fonttype'] = 42
    # 设置图形的显示风格
    # plt.style.use('ggplot')
    data = {
        'TaylorGP': fitnessGPTaylor,
        'GPLearn': fitnessGP,
        'FFX': fitnessFFX,
        'Linear\nRegression': fitnessLinearSR,
        'KernelRidge': fitnessKernelRidge,
        'RandomForest\nRegressor': fitnessRandomForestRegressor,
        'SVM': fitnessSVM,
        'XGBoost': fitnessXGBoost,
        'GSGP': fitnessGSGP,
        'BSR': fitnessBSR
    }

    df = pd.DataFrame(data)

    fig = plt.figure()
    axes = fig.add_axes([0.1, 0.15, 0.8, 0.8])
    axes.set_ylabel("Score(Normalized $R^{2}$)")
    bplot1 = axes.boxplot(df,  # 描点上色
                          # notch=True,  # 箱体中位数凹陷
                          vert=True,  # 设置箱体垂直
                          patch_artist=True,  # 允许填充颜色
                          showfliers=False,  # 不显示异常值
                          widths=0.8,  # 箱体宽度
                          medianprops={'color': 'black'})  # 设置中位数线的属性，线的类型和颜色

    # 颜色填充
    colors = ['red', 'lightcoral', 'gold', 'yellow', 'lime', 'darkseagreen', 'cyan', 'deepskyblue', 'darkorchid',
              'hotpink']
    for patch, color in zip(bplot1['boxes'], colors):
        patch.set_facecolor(color)
    plt.xticks(np.arange(11), (
        '', 'TaylorGP', 'GPLearn', 'FFX', 'Linear\nRegression', 'KernelRidge', 'RandomForest\nRegressor', 'SVM',
        'XGBoost', 'GSGP', 'BSR'),
               fontsize=7, rotation=30)
    data_son = {
        'TaylorGP': fitnessGPTaylor,
        # 'GPLearn': fitnessGP,
        # 'FFX': fitnessFFX,
        # 'Linear\nRegression': fitnessLinearSR,
        # 'KernelRidge': fitnessKernelRidge,
        # 'RandomForest\nRegressor': fitnessRandomForestRegressor,
        # 'SVM': fitnessSVM,
        'XGBoost': fitnessXGBoost,
        'GSGP': fitnessGSGP,
        'BSR': fitnessBSR
    }
    df_son = pd.DataFrame(data_son)

    axes_son = fig.add_axes([0.57, 0.25, 0.30, 0.3])
    bplot2 = axes_son.boxplot(df_son,  # 描点上色
                              # notch=True,  # 箱体中位数凹陷
                              vert=True,  # 设置箱体垂直
                              patch_artist=True,  # 允许填充颜色
                              showfliers=False,  # 不显示异常值
                              widths=0.8,  # 箱体宽度
                              medianprops={'color': 'black'})  # 设置中位数线的

    # 颜色填充
    colors_son = ['red', 'deepskyblue', 'darkorchid', 'hotpink']
    for patch, color in zip(bplot2['boxes'], colors_son):
        patch.set_facecolor(color)

    plt.grid(linestyle="--", alpha=0.3)
    # plt.ylabel("Score(Normalized $R^{2}$)")
    plt.xticks(np.arange(5), (
        '', 'TaylorGP', 'XGBoost', 'GSGP', 'BSR'),
               fontsize=7, rotation=30)

    figPath = os.path.join('IMG_COLOR', 'LOG')
    plt.savefig(os.path.join(figPath, dataName + '.pdf'), dpi=720, bbox_inches='tight')
    plt.close()
    # plt.show()


def addLog(pdSeries):
    return [np.log(ele + 1) if ele > -1 else 0 for ele in pdSeries.values]
    # return pdSeries.values


if __name__ == '__main__':
    GECCO = pd.read_csv('result_select/result_GECCO.csv')
    # draw_box_diagram(addLog(GECCO['TaylorGP']), addLog(GECCO['GPLearn']), addLog(GECCO['FFX']),
    #                  addLog(GECCO['LinearRegression']), addLog(GECCO['KernelRidge']), addLog(GECCO['RF']),
    #                  addLog(GECCO['SVM']), addLog(GECCO['XGBoost']), addLog(GECCO['GSGP']), addLog(GECCO['BSR']),
    #                  "GECCO", isAll=True)
    # draw_box_diagram(addLog(GECCO['TaylorGP']), addLog(GECCO['GPLearn']), addLog(GECCO['FFX']),
    #                  addLog(GECCO['LinearRegression']), addLog(GECCO['KernelRidge']), addLog(GECCO['RF']),
    #                  addLog(GECCO['SVM']), addLog(GECCO['XGBoost']), addLog(GECCO['GSGP']), addLog(GECCO['BSR']),
    #                  "GECCO", isAll=False)

    # draw_box_diagram_inside(addLog(GECCO['TaylorGP']), addLog(GECCO['GPLearn']), addLog(GECCO['FFX']),
    #                         addLog(GECCO['LinearRegression']), addLog(GECCO['KernelRidge']), addLog(GECCO['RF']),
    #                         addLog(GECCO['SVM']), addLog(GECCO['XGBoost']), addLog(GECCO['GSGP']), addLog(GECCO['BSR']),
    #                         "GECCO")

    AIFeynman = pd.read_csv('result_select/result_AIFeynman.csv')
    # draw_box_diagram(addLog(AIFeynman['TaylorGP']), addLog(AIFeynman['GPLearn']), addLog(AIFeynman['FFX']),
    #                  addLog(AIFeynman['LinearRegression']), addLog(AIFeynman['KernelRidge']),
    #                  addLog(AIFeynman['RF']),
    #                  addLog(AIFeynman['SVM']), addLog(AIFeynman['XGBoost']), addLog(AIFeynman['GSGP']),
    #                  addLog(AIFeynman['BSR']),
    #                  "AIFeynman", isAll=True)
    # draw_box_diagram(addLog(AIFeynman['TaylorGP']), addLog(AIFeynman['GPLearn']), addLog(AIFeynman['FFX']),
    #                  addLog(AIFeynman['LinearRegression']), addLog(AIFeynman['KernelRidge']),
    #                  addLog(AIFeynman['RF']),
    #                  addLog(AIFeynman['SVM']), addLog(AIFeynman['XGBoost']), addLog(AIFeynman['GSGP']),
    #                  addLog(AIFeynman['BSR']),
    #                  "AIFeynman", isAll=False)
    # draw_box_diagram_inside(addLog(AIFeynman['TaylorGP']), addLog(AIFeynman['GPLearn']), addLog(AIFeynman['FFX']),
    #                         addLog(AIFeynman['LinearRegression']), addLog(AIFeynman['KernelRidge']),
    #                         addLog(AIFeynman['RF']),
    #                         addLog(AIFeynman['SVM']), addLog(AIFeynman['XGBoost']), addLog(AIFeynman['GSGP']),
    #                         addLog(AIFeynman['BSR']),
    #                         "AIFeynman")

    ML = pd.read_csv('result_select/result_ML.csv')
    # draw_box_diagram(addLog(ML['TaylorGP']), addLog(ML['GPLearn']), addLog(ML['FFX']),
    #                  addLog(ML['LinearRegression']), addLog(ML['KernelRidge']),
    #                  addLog(ML['RF']),
    #                  addLog(ML['SVM']), addLog(ML['XGBoost']), addLog(ML['GSGP']), addLog(ML['BSR']),
    #                  "ML", isAll=True)
    # draw_box_diagram(addLog(ML['TaylorGP']), addLog(ML['GPLearn']), addLog(ML['FFX']),
    #                  addLog(ML['LinearRegression']), addLog(ML['KernelRidge']),
    #                  addLog(ML['RF']),
    #                  addLog(ML['SVM']), addLog(ML['XGBoost']), addLog(ML['GSGP']), addLog(ML['BSR']),
    #                  "ML", isAll=False)
    # draw_box_diagram_inside(addLog(ML['TaylorGP']), addLog(ML['GPLearn']), addLog(ML['FFX']),
    #                         addLog(ML['LinearRegression']), addLog(ML['KernelRidge']),
    #                         addLog(ML['RF']),
    #                         addLog(ML['SVM']), addLog(ML['XGBoost']), addLog(ML['GSGP']), addLog(ML['BSR']),
    #                         "ML")

    # draw_box_diagram_contact(
    #     addLog(pd.concat([AIFeynman['TaylorGP'], GECCO['TaylorGP'], ML['TaylorGP']], axis=0, ignore_index=True)),
    #     addLog(pd.concat([AIFeynman['GPLearn'], GECCO['GPLearn'], ML['GPLearn']], axis=0, ignore_index=True)),
    #     addLog(pd.concat([AIFeynman['FFX'], GECCO['FFX'], ML['FFX']], axis=0, ignore_index=True)), addLog(
    #         pd.concat([AIFeynman['LinearRegression'], GECCO['LinearRegression'], ML['LinearRegression']], axis=0,
    #                   ignore_index=True)), addLog(
    #         pd.concat([AIFeynman['KernelRidge'], GECCO['KernelRidge'], ML['KernelRidge']], axis=0, ignore_index=True)),
    #     addLog(pd.concat([AIFeynman['RF'], GECCO['RF'], ML['RF']], axis=0, ignore_index=True)),
    #     addLog(pd.concat([AIFeynman['SVM'], GECCO['SVM'], ML['SVM']], axis=0, ignore_index=True)),
    #     addLog(pd.concat([AIFeynman['XGBoost'], GECCO['XGBoost'], ML['XGBoost']], axis=0, ignore_index=True)),
    #     addLog(pd.concat([AIFeynman['GSGP'], GECCO['GSGP'], ML['GSGP']], axis=0, ignore_index=True)),
    #     addLog(pd.concat([AIFeynman['BSR'], GECCO['BSR'], ML['BSR']], axis=0, ignore_index=True)), isAll=True)
    # draw_box_diagram_contact(
    #     addLog(pd.concat([AIFeynman['TaylorGP'], GECCO['TaylorGP'], ML['TaylorGP']], axis=0, ignore_index=True)),
    #     addLog(pd.concat([AIFeynman['GPLearn'], GECCO['GPLearn'], ML['GPLearn']], axis=0, ignore_index=True)),
    #     addLog(pd.concat([AIFeynman['FFX'], GECCO['FFX'], ML['FFX']], axis=0, ignore_index=True)), addLog(
    #         pd.concat([AIFeynman['LinearRegression'], GECCO['LinearRegression'], ML['LinearRegression']], axis=0,
    #                   ignore_index=True)), addLog(
    #         pd.concat([AIFeynman['KernelRidge'], GECCO['KernelRidge'], ML['KernelRidge']], axis=0, ignore_index=True)),
    #     addLog(pd.concat([AIFeynman['RF'], GECCO['RF'], ML['RF']], axis=0, ignore_index=True)),
    #     addLog(pd.concat([AIFeynman['SVM'], GECCO['SVM'], ML['SVM']], axis=0, ignore_index=True)),
    #     addLog(pd.concat([AIFeynman['XGBoost'], GECCO['XGBoost'], ML['XGBoost']], axis=0, ignore_index=True)),
    #     addLog(pd.concat([AIFeynman['GSGP'], GECCO['GSGP'], ML['GSGP']], axis=0, ignore_index=True)),
    #     addLog(pd.concat([AIFeynman['BSR'], GECCO['BSR'], ML['BSR']], axis=0, ignore_index=True)), isAll=False)

    # draw_box_diagram_contact_inside(
    #     addLog(pd.concat([AIFeynman['TaylorGP'], GECCO['TaylorGP'], ML['TaylorGP']], axis=0, ignore_index=True)),
    #     addLog(pd.concat([AIFeynman['GPLearn'], GECCO['GPLearn'], ML['GPLearn']], axis=0, ignore_index=True)),
    #     addLog(pd.concat([AIFeynman['FFX'], GECCO['FFX'], ML['FFX']], axis=0, ignore_index=True)), addLog(
    #         pd.concat([AIFeynman['LinearRegression'], GECCO['LinearRegression'], ML['LinearRegression']], axis=0,
    #                   ignore_index=True)), addLog(
    #         pd.concat([AIFeynman['KernelRidge'], GECCO['KernelRidge'], ML['KernelRidge']], axis=0, ignore_index=True)),
    #     addLog(pd.concat([AIFeynman['RF'], GECCO['RF'], ML['RF']], axis=0, ignore_index=True)),
    #     addLog(pd.concat([AIFeynman['SVM'], GECCO['SVM'], ML['SVM']], axis=0, ignore_index=True)),
    #     addLog(pd.concat([AIFeynman['XGBoost'], GECCO['XGBoost'], ML['XGBoost']], axis=0, ignore_index=True)),
    #     addLog(pd.concat([AIFeynman['GSGP'], GECCO['GSGP'], ML['GSGP']], axis=0, ignore_index=True)),
    #     addLog(pd.concat([AIFeynman['BSR'], GECCO['BSR'], ML['BSR']], axis=0, ignore_index=True)))

    nestPics(
        addLog(pd.concat([AIFeynman['TaylorGP'], GECCO['TaylorGP'], ML['TaylorGP']], axis=0, ignore_index=True)),
        addLog(pd.concat([AIFeynman['GPLearn'], GECCO['GPLearn'], ML['GPLearn']], axis=0, ignore_index=True)),
        addLog(pd.concat([AIFeynman['FFX'], GECCO['FFX'], ML['FFX']], axis=0, ignore_index=True)), addLog(
            pd.concat([AIFeynman['LinearRegression'], GECCO['LinearRegression'], ML['LinearRegression']], axis=0,
                      ignore_index=True)), addLog(
            pd.concat([AIFeynman['KernelRidge'], GECCO['KernelRidge'], ML['KernelRidge']], axis=0, ignore_index=True)),
        addLog(pd.concat([AIFeynman['RF'], GECCO['RF'], ML['RF']], axis=0, ignore_index=True)),
        addLog(pd.concat([AIFeynman['SVM'], GECCO['SVM'], ML['SVM']], axis=0, ignore_index=True)),
        addLog(pd.concat([AIFeynman['XGBoost'], GECCO['XGBoost'], ML['XGBoost']], axis=0, ignore_index=True)),
        addLog(pd.concat([AIFeynman['GSGP'], GECCO['GSGP'], ML['GSGP']], axis=0, ignore_index=True)),
        addLog(pd.concat([AIFeynman['BSR'], GECCO['BSR'], ML['BSR']], axis=0, ignore_index=True)))
