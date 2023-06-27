from gplearn.genetic import SymbolicRegressor
from taylorGP.calTaylor import Metrics,Metrics2 # ,cal_Taylor_features
from taylorGP._program import print_program
from taylorGP._global import _init,set_value

import numpy as np
from sklearn.metrics import mean_squared_error  # 均方误差
import time
import sys
import argparse
from sympy import *
_init()
set_value('TUIHUA_FLAG',False)
def CalTaylorFeatures(f_taylor,_x,X,Y,Pop,repeatNum,eq_write):
    print('进入CalTaylorFeatures')
    metric = Metrics2(f_taylor,_x,X,Y)
    if metric.judge_Low_polynomial():
        return metric.low_mse, metric.f_low_taylor
    if X.shape[1]>1:
        if metric.judge_additi_separability():
            print('具有加法可分性')
            print('===========================开始左递归============================')
            low_mse1, f_add1 = CalTaylorFeatures(metric.f_left_taylor, metric._x_left, metric.X_left, metric.Y_left,Pop//2,repeatNum,eq_write)
            print('===========================开始右递归============================')
            low_mse2, f_add2 = CalTaylorFeatures(metric.f_right_taylor, metric._x_right, metric.X_right, metric.Y_right,Pop//2,repeatNum,eq_write)

            f_add = sympify(str(f_add1)+ '+' + str(f_add2))
            # y_pred = metric._calY(f_taylor,_x,metric._X)
            try:
                y_pred_add = metric._calY(f_add,_x,metric._X)
                # mse = mean_squared_error(y_pred,y_pred_add)
                mse = mean_squared_error(Y,y_pred_add)
                if mse < metric.low_mse:
                    return mse,f_add
                else:
                    return metric.low_mse,metric.f_low_taylor
            except BaseException:#演化计算生成的公式加了保护，而sympy无保护，除0会报错
                return metric.low_mse, metric.f_low_taylor
        elif metric.judge_multi_separability():
            print('具有乘法可分性')
            print('===========================开始左递归============================')
            low_mse1, f_multi1 = CalTaylorFeatures(metric.f_left_taylor, metric._x_left, metric.X_left, metric.Y_left,Pop//2,repeatNum,eq_write)
            print('===========================开始右递归============================')
            low_mse2, f_multi2 = CalTaylorFeatures(metric.f_right_taylor, metric._x_right, metric.X_right, metric.Y_right,Pop//2,repeatNum,eq_write)

            f_multi = sympify( '(' + str(f_multi1) + ')*(' + str(f_multi2)+')')
            # f_multi = sympify( '(' + str(f_multi1) + ')*(' + str(f_multi2)+')/'+str(metric.f_mid))
            # y_pred = metric._calY(f_taylor,_x,metric._X)
            try:
                y_pred_multi = metric._calY(f_multi,_x,metric._X)
                # mse = mean_squared_error(y_pred,y_pred_add)
                mse = mean_squared_error(Y,y_pred_multi)
                if mse < metric.low_mse:
                    return mse,f_multi
                else:
                    return metric.low_mse,metric.f_low_taylor
            except BaseException:#演化计算生成的公式加了保护，而sympy无保护，除0会报错
                return metric.low_mse, metric.f_low_taylor

    qualified_list = []
    qualified_list.extend([metric.judge_Bound(),metric.f_low_taylor,metric.low_mse,metric.bias,metric.judge_parity(),metric.judge_monotonicity()])
    return Taylor_Based_SR(_x,X,metric.change_Y(Y),qualified_list,eq_write,Pop,repeatNum,metric.judge_Low_polynomial())


def Taylor_Based_SR(_x,X,Y,qualified_list,eq_write,Pop,repeatNum,low_polynomial):
    f_low_taylor = qualified_list[-5]
    f_low_taylor_mse = qualified_list[-4]
    if low_polynomial == False:  # 比较低阶多项式和GP 输出适应度好的个体
        function_set = ['add', 'sub', 'mul', 'div', 'sin', 'cos', 'log', 'exp', 'sqrt']
        est_gp = SymbolicRegressor(population_size=Pop, init_depth=(2, 5),
                                   generations=12, stopping_criteria=1e-5, function_set=function_set,
                                   p_crossover=0.7, p_subtree_mutation=0.,
                                   p_hoist_mutation=0., p_point_mutation=0.2,
                                   max_samples=1.0, verbose=1,
                                   # metric='rmse',
                                   parsimony_coefficient=0.1,
                                   n_jobs=1,  # 用的不恰当反而效率低
                                   const_range=(-1, 1),
                                   random_state=repeatNum, low_memory=True)
        print(qualified_list)
        est_gp.fit(X, Y, qualified_list, eq_write)
        if est_gp._program.raw_fitness_ > f_low_taylor_mse:
            print(f_low_taylor, f_low_taylor_mse, sep='\n')
            return f_low_taylor_mse,f_low_taylor
            # average_fitness += f_low_taylor_mse
        else:
            return est_gp._program.raw_fitness_,print_program(est_gp._program.get_expression(), qualified_list, X,_x)
            # print_program(est_gp._program, qualified_list, X)
            # print(est_gp._program, est_gp._program.raw_fitness_, sep='\n')
            # average_fitness += est_gp._program.raw_fitness_
    else:
        return f_low_taylor_mse, f_low_taylor
        # print(f_low_taylor, f_low_taylor_mse, sep='\n')
        # average_fitness += f_low_taylor_mse
def cal_gplearn_master(fileNum):
    x0, x1, x2, x3, x4, x5, x6, x7, x8, x9, x10, x11, x12, x13, x14, x15, x16, x17, x18, x19 ,x20, x21, x22, x23, x24, x25, x26, x27, x28, x29 = symbols("x0,x1,x2,x3,x4,x5,x6,x7,x8,x9,x10,x11,x12,x13,x14,x15,x16,x17,x18,x19,x20,x21,x22,x23,x24,x25,x26,x27,x28,x29 ")
    _x = [x0, x1, x2, x3, x4, x5, x6, x7, x8, x9, x10, x11, x12, x13, x14, x15, x16, x17, x18, x19 ,x20, x21, x22, x23, x24, x25, x26, x27, x28, x29 ]
    eqName = r'eq_' + str(fileNum) + '.csv'
    # eqName = r'result/eq_' + str(fileNum) + '.csv'
    eq_write = open(eqName, "w+")  # 重新写
    eq_write.write('Gen|Top1|Length1|Fitness1|Top2|Length2|Fitness2|Top3|Length3|Fitness3|Top4|Length4|Fitness4|Top5|Length5|Fitness5|Top6|Length6|Fitness6|Top7|Length7|Fitness7|Top8|Length8|Fitness8|Top9|Length9|Fitness9|Top10|Length10|Fitness10\n')
    print('fileNum=',fileNum)
    average_fitness = 0
    repeat = 1
    time_start1 = time.time()
    for repeatNum in range(repeat):
        time_start2 = time.time()
        X_Y = np.loadtxt(r"D:\PYcharm_program\Test_everything\Bench_0.15\BenchMark_" + str(fileNum) + ".tsv",dtype=np.float,skiprows=1)
        # hsplit函数可以水平分隔数组，该函数有两个参数，第 1 个参数表示待分隔的数组， 第 2 个参数表示要将数组水平分隔成几个小数组
        X,Y = np.split(X_Y, (-1,), axis=1)
        loopNum = 0
        Metric = []
        while True:  # 进行5次低阶多项式的判别--不是则继续
            metric = Metrics(varNum=X.shape[1], fileNum=fileNum)
            loopNum += 1
            Metric.append(metric)
            if loopNum == 2 and X.shape[1] <= 2:
                break
            elif loopNum == 5 and (X.shape[1]>2 and X.shape[1]<=3):
                break
            elif loopNum == 4 and (X.shape[1]>3 and X.shape[1]<=4):
                break
            elif loopNum == 3 and (X.shape[1]>4 and X.shape[1]<=5):
                break
            elif loopNum == 2 and (X.shape[1]>5 and X.shape[1]<=6):
                break
            elif loopNum == 1 and (X.shape[1]>6):
                break
        Metric.sort(key=lambda x: x.mse)  # 按低阶多项式拟合误差对20个metric排序
        metric = Metric[0]
        print('排序后选中多项式_and_低阶多项式MSE:', metric.mse, metric.low_mse)
        eq_write.write(str(-1) + '|' + str(metric.f_low_taylor) + '|' + '10' + '|' + str(metric.low_mse) + '|'+'\n')
        if metric.mse < 0.1:
            metric.nihe_flag = True
        else:#拟合失败后Taylor特征的判别以数据集为准
            metric.bias = 0.
            print('拟合失败')
        Pop = 1000
        end_fitness,program = None,None
        #下面分成有可分性的递归和非递归处理方式---对应需要Metric和Metric2两个类分别处理,Metric中的_x都是默认的
        if metric.judge_Low_polynomial():
            end_fitness, program = metric.low_mse,metric.f_low_taylor
        elif metric.nihe_flag and (metric.judge_additi_separability() or metric.judge_multi_separability() ):
            end_fitness,program = CalTaylorFeatures(metric.f_taylor,_x[:X.shape[1]],X,Y,Pop,repeatNum,eq_write)
            # nihe_flag, low_polynomial, qualified_list, Y = cal_Taylor_features(varNum=X.shape[1], fileNum=fileNum, Y=Y)
        else:#针对多变量不能拟合：仅根据数据集获取边界以及单调性指导SR--放在Taylor特征判别处理

            qualified_list = []
            qualified_list.extend(
                [metric.judge_Bound(),
                 metric.f_low_taylor,
                 metric.low_mse,
                 metric.bias,
                 metric.judge_parity(),
                 metric.judge_monotonicity()])
            print(qualified_list)
            end_fitness,program = Taylor_Based_SR(_x,X,metric.change_Y(Y),qualified_list,eq_write,Pop,repeatNum,metric.low_mse<1e-5)
        print('fitness_and_program',end_fitness,program,sep=' ')
        average_fitness += end_fitness
        time_end2 = time.time()
        print('current_time_cost', (time_end2 - time_start2) / 3600, 'hour')

    eq_write.close()
    time_end1 = time.time()
    print('average_time_cost', (time_end1 - time_start1) / 3600 / repeat, 'hour')
    print('average_fitness = ',average_fitness/repeat)
if __name__ == '__main__':
    # for i in range(28,29):
    # for i in range(34,38):
    # for i in range(38,41):
    # for i in range(41,44):
    # for i in range(44,45):
    sys.setrecursionlimit(300)  # 设置求导最大递归深度，防止程序卡主或崩溃
    argparser = argparse.ArgumentParser()
    argparser.add_argument('--fileNum', default=9, type=int)
    args = argparser.parse_args()
    cal_gplearn_master(args.fileNum)
    '''
    numList = []
    for ii in range(19):
        p = multiprocessing.Process(target=cal_gplearn_master, args=(ii,))
        numList.append(p)  # 将子进程对象添加在列表中
        p.start()  # 启动进程
    for i in numList:
        i.join()  # 每一个子进程执行结束后，开始下一个循环，待子进程全部执行完毕，再执行主进程
    print("Process end.")    
    '''



