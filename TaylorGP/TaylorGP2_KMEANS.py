import argparse
import sys
from sympy import symbols

import numpy as np
import warnings
import time

from TaylorGP.src.taylorGP._global import _init, set_value
from TaylorGP.src.taylorGP.subRegionCalculator import subRegionCalculator

warnings.filterwarnings('ignore')
_init()
x0, x1, x2, x3, x4, x5, x6, x7, x8, x9, x10, x11, x12, x13, x14, x15, x16, x17, x18, x19, x20, x21, x22, x23, x24, x25, x26, x27, x28, x29 = symbols(
    "x0,x1,x2,x3,x4,x5,x6,x7,x8,x9,x10,x11,x12,x13,x14,x15,x16,x17,x18,x19,x20,x21,x22,x23,x24,x25,x26,x27,x28,x29 ")

set_value('_x',
          [x0, x1, x2, x3, x4, x5, x6, x7, x8, x9, x10, x11, x12, x13, x14, x15, x16, x17, x18, x19, x20, x21, x22, x23,
           x24, x25, x26, x27, x28, x29])


def WriteFile_X_Y_pred(SRC, fileNum, cluster, repeatNum):
    # Y_predfile = "D:\PYcharm_program\dynamicalSystem\TaylorGP2_1\Y_pred\TaylorGP2_KMeans_Greedy_" + str(fileNum) + ".out"
    # Y_predfile = "D:\PYcharm_program\dynamicalSystem\TaylorGP2_1\Y_pred\TaylorGP2_KMeans_UCB_" + str(fileNum) + ".out"
    # Y_predfile = "/home/hebaihe/STORAGE/taylorSR/result/TaylorGP2_1/Y_pred/TaylorGP2_KMeans_Greedy_" + str(fileNum) + ".out"
    # Y_predfile = "/home/hebaihe/STORAGE/taylorSR/result/TaylorGP2_1/Y_pred/TaylorGP2_KMeans_NoMAB_" + str(fileNum) + ".out"
    Y_predfile = "Y_pred/TaylorGP2_KMeans_UCB_" + str(fileNum) + ".out"
    # Y_predfile = "/home/hebaihe/taylorSR/result/TaylorGP2_1/Y_pred/TaylorGP2_KMeans_NoMAB_" + str(fileNum) + ".out"
    # Y_predfile = "/home/hebaihe/taylorSR/result/TaylorGP2_1/Y_pred/TaylorGP2_KMeans_UCB_" + str(fileNum) + ".out"
    f = open(Y_predfile, "a+")
    f.write("repeatNum: " + str(repeatNum) + " cluster: " + str(cluster) + '\n')
    f.write("bestLassoFitness: ")
    f.write(str(SRC.bestLassoFitness) + '\n')
    f.write("globalBestLassoCoef: ")
    f.write(str(SRC.globalBestLassoCoef) + '\n')

    for X_Y_pred in SRC.globalBest_X_Y_pred:
        try:
            if isinstance(X_Y_pred[0], list) == False: X_Y_pred[0] = X_Y_pred[0].tolist()
            if isinstance(X_Y_pred[1], list) == False: X_Y_pred[1] = X_Y_pred[1].tolist()
        except AttributeError:
            print("Y_pred is None")
        f.write("subRegion_X: ")
        f.write(str(X_Y_pred[0]) + '\n')
        f.write("subRegion_Y_pred: ")
        f.write(str(X_Y_pred[1]) + '\n')
        # print(X_Y_pred[0])
        # print(X_Y_pred[1])
    f.close()


def Cal_fitness_Coef(dataSets, originalTaylorGPGen, totalGeneration, clusters, repeatNum, Pop, fileNum, np_x,
                     SR_method="gplearn"):
    SRC = subRegionCalculator(dataSets, originalTaylorGPGen, mabPolicy="UCB", lbd=1)
    countAvailableParameters = SRC.CalCountofAvailableParameters(clusters=clusters, np_x=np_x)
    mabLoopNum = max(totalGeneration // originalTaylorGPGen // countAvailableParameters, 1)
    for cluster in clusters:
        SRC.PreDbscan(-1, clusterMethod="KMEANS", n_clusters_=cluster, data_x=np_x)
        SRC.firstMabFlag = True
        set_value('FIRST_EVOLUTION_FLAG', True)  # 进行每轮数据集演化前执行
        print("mabLoopNum:"+str(mabLoopNum))
        for tryNum in range(mabLoopNum):
            SRC.CalTops(repeatNum, Pop, SR_method=SR_method)
            SRC.SubRegionPruning()
            SRC.SparseRegression()
            if SRC.bestLassoFitness < 1e-5:
                print("SelectedNumofEachSubRegion", [Num for Num in SRC.abRockNum])
                print("FinalFitness", SRC.bestLassoFitness, "SelectedGlobalBestLassoCoef:", SRC.globalBestLassoCoef,
                      sep=" ")
                WriteFile_X_Y_pred(SRC, fileNum, cluster, repeatNum)
                return SRC.bestLassoFitness, SRC.globalBestLassoCoef
        print("TempFitness", SRC.bestLassoFitness, "SelectedGlobalBestLassoCoef:", SRC.globalBestLassoCoef, sep=" ")
    print("SelectedNumofEachSubRegion", [Num for Num in SRC.abRockNum])
    print("FinalFitness", SRC.bestLassoFitness, "SelectedGlobalBestLassoCoef:", SRC.globalBestLassoCoef, sep=" ")

    WriteFile_X_Y_pred(SRC, fileNum, cluster, repeatNum)
    if isinstance(SRC.globalBestLassoCoef,list):
        best_ind = SRC.tops[SRC.globalBestLassoCoef[0]]
    elif isinstance(SRC.globalBestLassoCoef,int):
        best_ind=SRC.tops[SRC.globalBestLassoCoef]
    else:
        raise ValueError("SRC.globalBestLassoCoef的类型有问题:"+f"{type(SRC.globalBestLassoCoef)}")
    print(best_ind)
    print("best_ind" + str(best_ind[1][0]))

    return SRC.bestLassoFitness, SRC.globalBestLassoCoef,str(best_ind[1][0])


def TaylorGP2Master(fileNum):  # 核心代码
    """
    核心代码:总共演化100次，摇臂10次，剪枝10次，稀疏回归10次，每次选中臂可以进化100/10次
    Args:
        dataSets:输入数据集

    Returns:tops,#存储m个聚类的前k个个体，先测试一个
    """
    # fileName = "D:\PYcharm_program\Test_everything\AllDataSets\F" + str(fileNum) + ".tsv"
    fileName = "/home/hebaihe/STORAGE/taylorSR/data/AllDataSets/F" + str(args.fileNum) + ".tsv"
    # fileName = "/home/hebaihe/taylorSR/TaylorGP2_1/AllDataSets/F" + str(args.fileNum) + ".tsv"
    dataSets = np.loadtxt(fileName, dtype=np.float, skiprows=1)
    print("fileNum", args.fileNum, "shape:", dataSets.shape[1] - 1)

    average_fitness = 0
    repeat = 2
    totalGeneration = 500
    originalTaylorGPGen = 10
    Pop = 1000  # 种群规模
    if dataSets.shape[1] - 1 == 1:
        clusters = [1]
    elif dataSets.shape[1] - 1 == 2:
        clusters = [2, 4]
    else:
        clusters = [1, 2, 4, 8, 16]
    time_start1 = time.time()
    for repeatNum in range(repeat):
        time_start2 = time.time()
        bestLassoFitness, globalBestLassoCoef = Cal_fitness_Coef(dataSets, originalTaylorGPGen, totalGeneration,
                                                                 clusters, repeatNum, Pop, fileNum)
        average_fitness += bestLassoFitness
        time_end2 = time.time()
        print('current_time_cost', (time_end2 - time_start2) / 3600, 'hour')
    time_end1 = time.time()
    print('average_time_cost', (time_end1 - time_start1) / 3600 / repeat, 'hour')
    print('average_fitness = ', average_fitness / repeat)


if __name__ == '__main__':
    sys.setrecursionlimit(300)
    argparser = argparse.ArgumentParser()
    argparser.add_argument('--fileNum', default=30, type=int)
    args = argparser.parse_args()

    TaylorGP2Master(args.fileNum)

    """
    for fileNum in [79,80,81]:
        
        fileName = "D:\PYcharm_program\Test_everything\AllDataSets\F" + str(fileNum) + ".tsv"
        dataSets = np.loadtxt(fileName,dtype=np.float,skiprows=1)
        TaylorGP2Master(dataSets)  
    """
