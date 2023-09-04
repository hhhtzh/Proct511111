import time

from sympy import *
import numpy as np
from sklearn.metrics import mean_squared_error
import timeout_decorator
import copy
import itertools
import math
# from itertools import product
CountACC = 0.0


def Global():
    global CountACC


x, y, z, v, w, a, b, c, d = symbols("x,y,z,v,w,a,b,c,d")
x0, x1, x2, x3, x4, x5, x6, x7, x8, x9, x10, x11, x12, x13, x14, x15, x16, x17, x18, x19, x20, x21, x22, x23, x24, x25, x26, x27, x28, x29,x30,x31,x32,x33,x34,x35,x36,x37,x38,x39,x40,x41,x42,x43,x44,x45,x46,x47,x48,x49 = symbols(
    "x0,x1,x2,x3,x4,x5,x6,x7,x8,x9,x10,x11,x12,x13,x14,x15,x16,x17,x18,x19,x20,x21,x22,x23,x24,x25,x26,x27,x28,x29,x30,x31,x32,x33,x34,x35 ,x36,x37,x38,x39,x40,x41,x42,x43,x44,x45,x46,x47,x48,x49")


class Point:
    name = 'Select k+1 Points to calculate Taylor Series'

    def __init__(self, in1=0, in2=0, in3=0, in4=0, in5=0, target=0, expansionPoint=2., varNum=1):
        self.varNum = varNum
        self.in1 = in1
        self.in2 = in2
        self.in3 = in3
        self.in4 = in4
        self.in5 = in5
        self.target = target
        self.expansionPoint = expansionPoint

    def __lt__(self, other):
        if self.varNum == 1:
            return abs(self.in1 - self.expansionPoint) < abs(other.in1 - self.expansionPoint)
        elif self.varNum == 2:
            return (self.in1 - self.expansionPoint) ** 2 + (self.in2 - self.expansionPoint) ** 2 < (
                    other.in1 - self.expansionPoint) ** 2 + (other.in2 - self.expansionPoint) ** 2
        elif self.varNum == 3:
            return (self.in1 - self.expansionPoint) ** 2 + (self.in2 - self.expansionPoint) ** 2 + (
                    self.in3 - self.expansionPoint) ** 2 < (other.in1 - self.expansionPoint) ** 2 + (
                           other.in2 - self.expansionPoint) ** 2 + (other.in3 - self.expansionPoint) ** 2
        elif self.varNum == 4:
            return (self.in1 - self.expansionPoint) ** 2 + (self.in2 - self.expansionPoint) ** 2 + (
                    self.in3 - self.expansionPoint) ** 2 + (self.in4 - self.expansionPoint) ** 2 < (
                           other.in1 - self.expansionPoint) ** 2 + (other.in2 - self.expansionPoint) ** 2 + (
                           other.in3 - self.expansionPoint) ** 2 + (other.in4 - self.expansionPoint) ** 2
        elif self.varNum == 5:
            return (self.in1 - self.expansionPoint) ** 2 + (self.in2 - self.expansionPoint) ** 2 + (
                    self.in3 - self.expansionPoint) ** 2 + (self.in4 - self.expansionPoint) ** 2 + (
                           self.in5 - self.expansionPoint) ** 2 < (other.in1 - self.expansionPoint) ** 2 + (
                           other.in2 - self.expansionPoint) ** 2 + (other.in3 - self.expansionPoint) ** 2 + (
                           other.in4 - self.expansionPoint) ** 2 + (other.in5 - self.expansionPoint) ** 2


class Metrics:
    name = 'Good calculator'

    def __init__(self, fileName=0,dataSet =None, model=None, f=None, classNum=8, varNum=1,linalg= "solve"):
        self.model = model
        self.f_taylor = 0
        self.f_low_taylor = 0
        self.fileName = fileName
        self.dataSet = dataSet
        self.classNum = classNum
        self.x, self.y, self.z, self.v, self.w = symbols("x,y,z,v,w")
        self.x0, self.x1, self.x2, self.x3, self.x4 = symbols("x0,x1,x2,x3,x4")
        _x = [x0, x1, x2, x3, x4, x5, x6, x7, x8, x9]
        self._x = _x[:varNum]
        self.count = 0
        self.mediumcount = 0
        self.supercount = 0
        self.count1 = 0
        self.mediumcount1 = 0
        self.supercount1 = 0
        self.count2 = 0
        self.mediumcount2 = 0
        self.supercount2 = 0
        self.tempVector = np.zeros((1, 6, 126))
        self.varNum = varNum
        self.di_jian_flag = False
        self.parity_flag = False
        self.bias = 0.
        self.nmse = float("inf")
        self.low_nmse = float("inf")
        self.mse_log = float("inf")
        self.Y_log = None
        self.b_log = None
        self.Taylor_log = None
        self.f_taylor_log = 0
        self.linalg = linalg
        self.A = None
        self.midpoint = None
        self.Y_left, self.Y_right = None, None
        self.X_left, self.X_right = None, None
        X_Y = dataSet
        self.expantionPoint = copy.deepcopy(X_Y[0])
        self.mmm = X_Y.shape[0] - 1
        np.random.shuffle(X_Y)
        change = True
        for i in range(self.mmm):
            if (X_Y[i] == self.expantionPoint).all():
                X_Y[[i, -1], :] = X_Y[[-1, i], :]
                break

        X, Y = np.split(X_Y, (-1,), axis=1)
        self.X = copy.deepcopy(X)
        _X = []
        len = X.shape[1]
        for i in range(len):
            X, temp = np.split(X, (-1,), axis=1)
            temp = temp.reshape(-1)
            _X.extend([temp])
        _X.reverse()
        self._X, self.X_Y, self.Y = _X, X_Y, Y.reshape(-1)
        self.f0_log, self.Y_log = np.log(X_Y[0][-1]), np.log(abs(self.Y))
        self.b, self.b_log = (self.Y - self.expantionPoint[-1])[:-1], (self.Y_log - self.f0_log)[:-1]
        self.nihe_flag = False
        self._mid_left, self._mid_right = 0, 0
        self._x_left, self._x_right = 0, 0

        self.f_taylor, self.k, self.nmse = self._getTaylorPolyBest(varNum=varNum)
        self.low_nmse = self.nmse
        self.f_low_taylor = self.f_taylor
    
    def _getTaylorPolyBest(self, varNum):
        if varNum == 1 or varNum == 2:
            f_taylor ,k ,nmse= self._CalTaylorNmse(varNum,10)
        elif varNum == 3 or varNum == 4:
            f_taylor ,k ,nmse= self._CalTaylorNmse(varNum,8)
        elif varNum == 5 or varNum == 6:
            f_taylor ,k ,nmse= self._CalTaylorNmse(varNum,6)
        elif varNum == 7 or varNum == 8:
            f_taylor ,k ,nmse= self._CalTaylorNmse(varNum,4)
        elif varNum < 12:
            f_taylor ,k ,nmse= self._CalTaylorNmse(varNum,3)
        elif varNum < 20:
            f_taylor ,k ,nmse= self._CalTaylorNmse(varNum,2)
        else:    
            mytaylor_num , f ,k= self._getTDatas(varNum,1)

        return f_taylor ,k ,nmse
        

    
    def _CalTaylorNmse(self, varNum ,k):
        
        for i in range(1,k):
            mytaylor_num , f ,k= self._getTDatas(varNum,i)
            f_taylor = sympify(f)
            f_taylor = f_taylor.expand()
            test_y_pred = self._calY(f_taylor)
            test_nmse = mean_squared_error(self.Y, test_y_pred)
            if test_nmse < self.nmse:
                nmse = test_nmse
                f_taylor_all = f_taylor
                k1 = k
            print('GET Taylor New  NMSE expanded to order k，k=', k1, 'nmse=', nmse)
        return f_taylor_all ,k1 ,nmse

        
    def _getTDatas(self, varNum, k):
        # the datasets shape is mmm
        mmm = self.X.shape[0] - 1  
        combinations = itertools.product(range(k), repeat=varNum)
        count=0
        # 遍历每个组合
        for combo in combinations:
        # 计算组合中所有数字的总和
            total = sum(combo)
            # 如果总和小于等于 n，增加计数器
            if total <= k:
                count += 1
        print("count=",count)
        while count > mmm:
            k -= 1
            combinations = itertools.product(range(k), repeat=varNum)
            count=0
            # 遍历每个组合
            for combo in combinations:
            # 计算组合中所有数字的总和
                total = sum(combo)
                # 如果总和小于等于 n，增加计数器
                if total <= k:
                    count += 1
            print("count=",count)
        x_var=[]
        for i in range(varNum):
            x = self._X[i][:-1] - self.expantionPoint[i]
            x0 = np.resize(x, (count,)) 
            # x0 = x.reshape((count,))
            x_var.append(x0)
        # self.b = self.b.reshape(count,1)
        b0 = np.resize(self.b, (count,)) 
        # Get the (x0-a0)^0 (x0-a0)^1 (x0-a0)^2 (x0-a0)^3 .... (x0-a0)^n , such like this
        var_array =[]
        for i in range(varNum):
            sum_now=[]
            for j in range(k):
                sum_now.append(x_var[i]**j)
            var_array.append(sum_now)
        # Get the flag (or index) such like (0,0,0,0,0,0) (0,0,0,0,0,1) (0,0,0,0,0,2) (0,0,0,0,0,3) .... (0,0,0,0,0,n) 
        flag_array=[]
        for i in range(varNum):
            sum_now=[]
            z=-1
            for j in range(k):
                z+=1
                sum_now.append(z)
            flag_array.append(sum_now)
        
        # USE itertools to get the combinations and flag
        combinations = list(itertools.product(*var_array))
        combinations_flag = list(itertools.product(*flag_array))
        # Get the combination such like num*(x0-a0)^0*(x1-a1)^0*(x2-a2)^0*(x3-a3)^0 .... *(xn-an)^0 
        product_array = []
        for combination in combinations:
            product = 1
            for num in combination:
                product *= num
            product_array.append(product)

        fl = []
        Number_record = []
        #Get the combination's index sum 
        for combination in combinations_flag:
            product = 0
            for num in combination:
                product += num
            # if the sum is less than k, we need to record the combination and the sum
            if product <=k:
                Number_record.append(combination)
            fl.append(product)
        
        # print("fl=",fl)

        # Get the A matrix
        A = None
        Number =0
        for i in range(len(fl)):
            if fl[i]<=k:
                Number+=1
                if A is None:
                    A = (product_array[i]).reshape(count, 1)
                else:
                    A = np.hstack((A, ((product_array[i]).reshape(count, 1))))

        length = Number
        print("length=",length)
        Taylor = 0
        # Taylor = np.linalg.lstsq(A, b0, rcond=None)[0]
        Taylor = np.linalg.solve(A, b0)

        Taylor = np.insert(Taylor, 0, self.expantionPoint[-1])  

        # transform the Taylor to the string of sympy ,and return the string f
        f=str(Taylor[0]) 
        for i,combination in enumerate(Number_record):
            u = i+1
            product =0
            for z in combination: #z is each num in combination
                product+=z
            if product<=k:
                # print("combination= ",combination,"i= ",i)
                lenth =len(combination)
                if(Taylor[u]>0):
                    f += '+'
                else:
                    pass
                f += str(Taylor[u]) 
                for j,num in enumerate(combination):
                    # print("j= ,num=",j,num)
                    f += '*' + '(x' + str(j) + '-' + str(self.expantionPoint[j]) + ')**' + str(num) 

        return Taylor.tolist()[:length],f,k


    def judge_Low_polynomial(self, lowLine=7, varNum=1):
        if self.low_nmse > 1e-5:
            return False
        return True

    
    def _calY(self, f, _x=None, X=None):
        y_pred = []
        len1, len2 = 0, 0
        if _x is None:
            _x = [x0, x1, x2, x3, x4, x5, x6, x7, x8, x9, x10, x11, x12, x13, x14, x15, x16, x17, x18, x19, x20, x21,
                  x22, x23, x24, x25, x26, x27, x28, x29,x30,x31,x32,x33,x34,x35,x36,x37,x38,x39,x40,x41,x42,x43,x44,x45,x46,x47,x48,x49]
        if X is None:
            X = self._X
            len2 = self.varNum
        else:
            len2 = len(X)
        len1 = X[0].shape[0]
        for i in range(len1):
            _sub = {}
            for j in range(len2):
                _sub.update({_x[j]: X[j][i]})
            y_pred.append(f.evalf(subs=_sub))
        return y_pred

    @timeout_decorator.timeout(10, use_signals=False)
    def cal_critical_point(self, fx, x):
        if self.varNum == 2:
            return solve([fx[0], fx[1]], [x[0], x[1]])

    def judge_Bound(self):
        if self.nihe_flag == False:
            y_bound, var_bound = [], []
            _X = copy.deepcopy(self._X)
            for i in range(len(_X)):
                _X[i].sort()
                var_bound.extend([_X[i][0], _X[i][-1]])
            _Y = self.Y.reshape(-1)
            _Y.sort()
            y_bound.extend([_Y[0], _Y[-1]])
            return [y_bound, var_bound]
        else:
            y_bound, var_bound = [], []
            _X = copy.deepcopy(self._X)
            for i in range(len(_X)):
                _X[i].sort()
                var_bound.extend([_X[i][0], _X[i][-1]])
            Y = copy.deepcopy(self.Y)
            Y.sort()
            y_bound.extend([Y[0], Y[-1]])

            f_diff = []
            for i in range(len(self._X)):
                f_diff.append(sympify(diff(self.f_taylor, self._x[i])))
            '''
            try:
                critical_point = self.cal_critical_point(f_diff, self._x[:len(self._X)])
            except BaseException:
                critical_point = None
            if critical_point is not None:
                for c in critical_point:
                    if 'I' not in str(c) and not any(
                            [c[0] < var_bound[[i][0]] and c[1] > var_bound[i][1] for i in range(len(c))]):
                        _sub = {}
                        for i in range(len(c)):
                            _sub.update({self._x[i]: c[i]})
                        y_bound.append(self.f_taylor.evalf(subs=_sub))
                        print('Critical Point', c)            
            '''

            y_bound.sort()
            return [[y_bound[0], y_bound[-1]], var_bound]

    def judge_monotonicity(self, Num=1):
        Increase, Decrease = False, False
        X, Y = copy.deepcopy(self.X), copy.deepcopy(self.Y)
        Y_index = np.argsort(Y, axis=0)
        Y_index = Y_index.reshape(-1)
        for i in range(1, Y_index.shape[0]):
            Increase_flag = not any([(X[Y_index[i]][j] < X[Y_index[i - 1]][j]) for j in range(X.shape[1])])
            if Increase_flag:
                Increase = True
            Decrease_flag = not any([(X[Y_index[i]][j] > X[Y_index[i - 1]][j]) for j in range(X.shape[1])])
            if Decrease_flag:
                Decrease = True
        if Increase == True and Decrease == False:
            if Num == 1:
                print('Monotone increasing function！！！')
            else:
                print('concave function')
            return 1
        elif Increase == False and Decrease == True:
            if Num == 1:
                print('Monotone decreasing function！！！')
            else:
                print('convex function')
            return 2
        if Num == 1:
            print('Non increasing and non decreasing function！！！')
        else:
            print(' concavity and convexity')
        return -1

    def judge_program_monotonicity(self, Num=1):  # 适合任意一维和多维的情况
        Increase, Decrease = False, False
        f_ = diff(self.f_taylor, self.x, Num)  # 求f的Num阶导数
        for x_ in self.X0:
            if f_.evalf(subs={x: x_}) >= 0:
                Increase = True
            if f_.evalf(subs={x: x_}) <= 0:
                Decrease = True
        if Increase == True and Decrease == False:
            if Num == 1:
                print('Monotone increasing function！！！')
            else:
                print('concave function')
            return 1
        elif Increase == False and Decrease == True:
            if Num == 1:
                self.di_jian_flag = True
                print('Monotone decreasing function！！！')
            else:
                print('convex function')
            return 2
        if Num == 1:
            print('Non increasing non decreasing function！！！')
        else:
            print('no concavity and convexity')
        return -1

    def judge_concavityConvexity(self):
        return self.judge_monotonicity(Num=2)

    def cal_power_expr(self, expr):
        expr = expr.split('*')
        j = 0
        for i in range(len(expr)):
            if expr[j] == '':
                expr.pop(j)
            else:
                j += 1
        count = 0
        for i in range(1, len(expr) - 1):
            if 'x' in expr[i] and expr[i + 1].isdigit():
                count += int(expr[i + 1])
            elif 'x' in expr[i] and 'x' in expr[i + 1]:
                count += 1
        if 'x' in expr[-1]:
            count += 1
        return count

    def judge_parity(self):
        '''
                odd function：1
                even function：2
        '''

        if self.nihe_flag:
            if self.bias != 0:
                f_taylor = str(self.f_taylor).split()[:-2]
            else:
                f_taylor = str(self.f_taylor).split()
            Y = self.Y - self.bias
            f_odd, f_even = '', ''
            if self.cal_power_expr(f_taylor[0]) % 2 == 1:
                f_odd += f_taylor[0]
            else:
                f_even += f_taylor[0]
            for i in range(2, len(f_taylor), 2):
                if self.cal_power_expr(f_taylor[i - 1] + f_taylor[i]) % 2 == 1:
                    f_odd += f_taylor[i - 1] + f_taylor[i]
                else:
                    f_even += f_taylor[i - 1] + f_taylor[i]
            f_odd, f_even = sympify(f_odd), sympify(f_even)
            Jishu, Oushu, nmse_odd, nmse_even = False, False, 0, 0
            y_pred = self._calY(f_odd)
            nmse_odd = mean_squared_error(Y, y_pred)
            y_pred = self._calY(f_even)
            nmse_even = mean_squared_error(Y, y_pred)
            print('NMSE of parity function', nmse_odd, nmse_even, sep='\n')
            if nmse_odd < 0.01:
                Jishu = True
            if nmse_even < 0.01:
                Oushu = True
            if Jishu == True and Oushu == False:
                print('odd function！！！')
                self.parity_flag = True
                return 1
            elif Jishu == False and Oushu == True:
                self.parity_flag = True
                print('even function！！！')
                return 2
        print('non odd non even function！！！')
        return -1

    def judge_program_parity(self):
        Jishu, Oushu = False, False
        f = self.f_taylor
        for x_ in self.X0:
            if abs(f.evalf(subs={x: -1 * x_}) + f.evalf(subs={x: x_})) < 0.001:
                Jishu = True
            elif abs(f.evalf(subs={x: -1 * x_}) - f.evalf(subs={x: x_})) < 0.001:
                Oushu = True
            else:
                print('non odd non even function！！！')
                return -1
        if Jishu == True and Oushu == False:
            print('odd function！！！')
            return 1
        elif Jishu == False and Oushu == True:
            print('even function！！！')
            return 2

    def _cal_add_separability(self, multi_flag=False, expantionpoint=None):
        f_taylor = str(copy.deepcopy(self.f_taylor))
        for i in range(len(self._x_right)):
            f_taylor = f_taylor.replace(str(self._x_right[i]), str(self._mid_right[i])) + '-(' + str(self.bias) + ')'
        self.f_left_taylor = f_taylor
        f_taylor = str(copy.deepcopy(self.f_taylor))
        for i in range(len(self._x_left)):
            f_taylor = f_taylor.replace(str(self._x_left[i]), str(self._mid_left[i]))
        self.f_right_taylor = f_taylor

        self.f_left_taylor = sympify(self.f_left_taylor)
        self.f_right_taylor = sympify(self.f_right_taylor)
        Y_left = self._calY(self.f_left_taylor, self._x_left, self._X_left)
        Y_right = self._calY(self.f_right_taylor, self._x_right, self._X_right)
        if multi_flag:
            f_taylor = str(copy.deepcopy(self.f_taylor))
            for i in range(len(self._x)):
                f_taylor = f_taylor.replace(str(self._x[i]), str(expantionpoint[i]))
            self.f_mid = eval(f_taylor)
        len_Y = len(Y_left)
        Y_left, Y_right = np.array(Y_left), np.array(Y_right)
        self.Y_left, self.Y_right = Y_left.reshape(len_Y, 1), Y_right.reshape(len_Y, 1)
        return None

    def judge_additi_separability(self, f_taylor=None, taylor_log_flag=False):
        for i in range(len(self._x) - 1):
            _x = copy.deepcopy(self._x)
            _x_left = _x.pop(i)
            _x_right = _x
            if f_taylor == None:
                f_taylor = self.f_taylor
            f_taylor_split = str(f_taylor).split()
            f_temp = []
            for subF in f_taylor_split:
                if any([str(_x_left) in subF and str(_x_right[j]) in subF for j in range(len(_x_right))]):
                    f_temp = f_temp[:-1]
                else:
                    f_temp.append(subF)

            f = ''
            for temp in f_temp:
                f += temp
            f = sympify(f)
            y_pred = self._calY(f, self._x, self._X)
            if taylor_log_flag:
                nmse = mean_squared_error(self.Y_log, y_pred)
            else:
                nmse = mean_squared_error(self.Y, y_pred)
            if nmse > 0.001:
                continue
            else:
                print('addition separable')
                return True

    def judge_multi_separability(self):
        '''multiplicative separability discrimination'''
        _expantionpoint = copy.deepcopy(self.expantionPoint)
        if self.expantionPoint[-1] == 0:
            _expantionpoint = _expantionpoint + 0.2
        for i in range(len(self._x) - 1):
            f_taylor = self.f_taylor
            _x = copy.deepcopy(self._x)
            self._x_left = [_x[i]]
            _x.pop(i)
            self._x_right = _x
            expantionpoint = copy.deepcopy(_expantionpoint).tolist()
            self._mid_left = [expantionpoint[i]]
            expantionpoint.pop(i)
            self._mid_right = expantionpoint[:-1]
            _X = copy.deepcopy(self._X)
            self._X_left = [_X.pop(i)]
            try:
                a = _X[0].shape[0]
            except BaseException:
                _X = [_X]
            self._X_right = _X
            self._cal_add_separability(multi_flag=True, expantionpoint=expantionpoint)
            nmse = mean_squared_error(self.Y, self.Y_left * self.Y_right / self.f_mid)
            if nmse < 0.001:
                print('multiplication separable')
                print('multi_nmse = ', nmse)
                return True
            else:
                try:
                    return self.judge_additi_separability(f_taylor=self.f_taylor_log, taylor_log_flag=True)
                except BaseException:
                    return False

        return False

    def change_Y(self, Y):
        if Y is None:
            return None
        if self.parity_flag:
            if abs(self.bias) > 1e-5:
                Y -= self.bias
        if self.di_jian_flag:
            return Y * (-1)
        else:
            return Y


class Metrics2(Metrics):

    def __init__(self, f_taylor, _x, X, Y):
        self.f_taylor = f_taylor
        self.f_low_taylor = None
        self.x0, self.x1, self.x2, self.x3, self.x4 = symbols("x0,x1,x2,x3,x4")
        self._x = _x
        self.bias, self.low_nmse = 0., 0.
        self.varNum = X.shape[1]
        self.Y_left, self.Y_right, self.Y_right_temp = None, None, None
        self.X_left, self.X_right = None, None
        self.midpoint = None
        self.parity_flag = False
        self.di_jian_flag = False
        self.expantionPoint = np.append(copy.deepcopy(X[0]), Y[0][0])
        self.X = copy.deepcopy(X)
        _X = []
        len = X.shape[1]
        for i in range(len):
            X, temp = np.split(X, (-1,), axis=1)
            temp = temp.reshape(-1)
            _X.extend([temp])
        _X.reverse()
        self._X, self.Y = _X, Y.reshape(-1)
        self.b = (self.Y - self.expantionPoint[-1])[:-1]
        y_pred = self._calY(f_taylor, self._x, self._X)
        self.nihe_flag = False
        if mean_squared_error(self.Y, y_pred) < 0.01:
            self.nihe_flag = True
        self._mid_left, self._mid_right = 0, 0
        self._x_left, self._x_right = 0, 0

    def judge_Low_polynomial(self):
        f_taylor = str(self.f_taylor).split()
        try:
            self.bias = float(f_taylor[-2] + f_taylor[-1])
        except BaseException:
            self.bias = 0.
        f_low_taylor = ''
        if self.cal_power_expr(f_taylor[0]) <= 4:
            f_low_taylor += f_taylor[0]
        for i in range(2, len(f_taylor), 2):
            if self.cal_power_expr(f_taylor[i - 1] + f_taylor[i]) <= 4:
                f_low_taylor += f_taylor[i - 1] + f_taylor[i]
        self.f_low_taylor = sympify(f_low_taylor)
        print(f_low_taylor)
        y_pred_low = self._calY(self.f_low_taylor, self._x, self._X)
        self.low_nmse = mean_squared_error(self.Y, y_pred_low)
        if self.low_nmse < 1e-5:
            return True
        else:
            return False

    def judge_Bound(self):
        y_bound, var_bound = [], []
        _X = copy.deepcopy(self._X)
        for i in range(len(_X)):
            _X[i].sort()
            var_bound.extend([_X[i][0], _X[i][-1]])
        _Y = self.Y.reshape(-1)
        _Y.sort()
        y_bound.extend([_Y[0], _Y[-1]])
        return [y_bound, var_bound]

    def change_XToX(self, _X):
        len1 = len(_X)
        len2 = len(_X[0])
        X = np.array(_X[0])
        X = X.reshape(len(_X[0]), 1)
        for i in range(1, len1):
            temp = np.array(_X[i]).reshape(len2, 1)
            X = np.concatenate((X, temp), axis=1)
        return X

    def _cal_add_separability(self, multi_flag=False, expantionpoint=None):
        f_taylor = str(copy.deepcopy(self.f_taylor))
        for i in range(len(self._x_right)):
            f_taylor = f_taylor.replace(str(self._x_right[i]), str(self._mid_right[i])) + '-(' + str(self.bias) + ')'
        self.f_left_taylor = f_taylor
        f_taylor = str(copy.deepcopy(self.f_taylor))
        for i in range(len(self._x_left)):
            f_taylor = f_taylor.replace(str(self._x_left[i]), str(self._mid_left[i]))
        self.f_right_taylor = f_taylor

        self.f_left_taylor = sympify(self.f_left_taylor)
        self.f_right_taylor = sympify(self.f_right_taylor)
        Y_left = self._calY(self.f_left_taylor, self._x_left, self._X_left)
        Y_right = self._calY(self.f_right_taylor, self._x_right, self._X_right)
        self.X_left = self.change_XToX(self._X_left)
        self.X_right = self.change_XToX(self._X_right)
        if multi_flag:
            f_taylor = str(copy.deepcopy(self.f_taylor))
            for i in range(len(self._x)):
                f_taylor = f_taylor.replace(str(self._x[i]), str(expantionpoint[i]))
            self.f_mid = eval(f_taylor)
        len_Y = len(Y_left)
        Y_left, Y_right = np.array(Y_left), np.array(Y_right)
        Y = self.Y.reshape(len_Y, 1)
        self.Y_left = Y_left.reshape(len_Y, 1)
        try:
            if multi_flag:
                self.Y_right = Y_right.reshape(len_Y, 1)
                self.Y_right_temp = Y / self.Y_left
            else:
                self.Y_right = Y - self.Y_left
        except BaseException:
            self.Y_right_temp = Y_right.reshape(len_Y, 1)
        return None

    def judge_additi_separability(self, f_taylor=None, taylor_log_flag=False):
        '''additive separability discrimination'''
        for i in range(len(self._x) - 1):
            _x = copy.deepcopy(self._x)
            _x_left = _x.pop(i)
            _x_right = _x
            if f_taylor == None:
                f_taylor = self.f_taylor
            f_taylor_split = str(f_taylor).split()
            f_temp = []
            for subF in f_taylor_split:
                if any([str(_x_left) in subF and str(_x_right[j]) in subF for j in range(len(_x_right))]):
                    f_temp = f_temp[:-1]
                else:
                    f_temp.append(subF)
            f = ''
            for temp in f_temp:
                f += temp
            f = sympify(f)
            y_pred = self._calY(f, self._x, self._X)
            if taylor_log_flag:
                nmse = mean_squared_error(self.Y_log, y_pred)
            else:
                nmse = mean_squared_error(self.Y, y_pred)
            if nmse > 0.001:
                continue
            else:
                _x = copy.deepcopy(self._x)
                self._x_left = [_x[i]]
                _x.pop(i)
                self._x_right = _x
                expantionpoint = copy.deepcopy(self.expantionPoint).tolist()
                self._mid_left = [expantionpoint[i]]
                expantionpoint.pop(i)
                self._mid_right = expantionpoint[:-1]
                _X = copy.deepcopy(self._X)
                self._X_left = [_X.pop(i)]
                try:
                    a = _X[0].shape[0]
                except BaseException:
                    _X = [_X]
                self._X_right = _X
                self._cal_add_separability()
                print('addition separable')
                return True

    def judge_multi_separability(self):
        '''multiplicative separability discrimination'''
        _expantionpoint = copy.deepcopy(self.expantionPoint)
        if self.expantionPoint[-1] == 0:
            _expantionpoint = _expantionpoint + 0.2
        for i in range(len(self._x) - 1):
            f_taylor = self.f_taylor
            _x = copy.deepcopy(self._x)
            self._x_left = [_x[i]]
            _x.pop(i)
            self._x_right = _x
            expantionpoint = copy.deepcopy(_expantionpoint).tolist()
            self._mid_left = [expantionpoint[i]]
            expantionpoint.pop(i)
            self._mid_right = expantionpoint[:-1]
            _X = copy.deepcopy(self._X)
            self._X_left = [_X.pop(i)]
            try:
                a = _X[0].shape[0]
            except BaseException:
                _X = [_X]
            self._X_right = _X
            self._cal_add_separability(multi_flag=True, expantionpoint=expantionpoint)
            nmse = mean_squared_error(self.Y, self.Y_left * self.Y_right / self.f_mid)
            if nmse < 0.001:
                print('multiplication separable')
                print('multi_nmse = ', nmse)
                self.Y_right = self.Y_right_temp
                return True
            else:
                try:
                    return self.judge_additi_separability(f_taylor=self.f_taylor_log, taylor_log_flag=True)
                except BaseException:
                    return False

        return False

    def judge_parity(self):
        '''
        return：non odd non even function：-1
                odd function：1
                even function：2
        '''
        if self.nihe_flag:
            if self.bias != 0:
                f_taylor = str(self.f_taylor).split()[:-2]
            else:
                f_taylor = str(self.f_taylor).split()
            Y = self.Y - self.bias
            f_odd, f_even = '', ''
            if self.cal_power_expr(f_taylor[0]) % 2 == 1:
                f_odd += f_taylor[0]
            else:
                f_even += f_taylor[0]
            for i in range(2, len(f_taylor), 2):
                if self.cal_power_expr(f_taylor[i - 1] + f_taylor[i]) % 2 == 1:
                    f_odd += f_taylor[i - 1] + f_taylor[i]
                else:
                    f_even += f_taylor[i - 1] + f_taylor[i]
            f_odd, f_even = sympify(f_odd), sympify(f_even)
            Jishu, Oushu, nmse_odd, nmse_even = False, False, 0, 0
            y_pred = self._calY(f_odd, self._x, self._X)
            nmse_odd = mean_squared_error(Y, y_pred)
            y_pred = self._calY(f_even, self._x, self._X)
            nmse_even = mean_squared_error(Y, y_pred)
            print('NMSE of parity function', nmse_odd, nmse_even, sep='\n')
            if nmse_odd < 0.001:
                Jishu = True
            if nmse_even < 0.001:
                Oushu = True
            if Jishu == True and Oushu == False:
                print('Odd function！！！')
                self.parity_flag = True
                return 1
            elif Jishu == False and Oushu == True:
                self.parity_flag = True
                print('even function！！！')
                return 2
        print('non odd non even function！！！')
        return -1


def cal_Taylor_features(varNum, dataSet, Y=None):
    '''qualified_list = [low_high_target_bound, low_high_var_bound,bias,partity,monity]'''
    qualified_list = []
    low_polynomial = False

    Metric = []
    for linalg in ["solve","ls"]:
        loopNum = 0
        while True:
            metric = Metrics(varNum=varNum, dataSet=dataSet,linalg = linalg)
            loopNum += 1
            Metric.append(metric)
            if loopNum == 3:
                break
    Metric.sort(key=lambda x: x.nmse)
    metric = Metric[0]
    print('NMSE of polynomial and lower order polynomial after sorting', metric.nmse, metric.low_nmse)
    if metric.nmse < 0.1:
        metric.nihe_flag = True
    else:
        print('Fitting failed')
    if metric.judge_Low_polynomial():
        print('The result is a low order polynomial')
        low_polynomial = True

    '''
    add_seperatity = metric.judge_additi_separability()
    multi_seperatity = metric.judge_multi_separability()

    qualified_list.extend(metric.judge_Bound()) 
    # qualified_list.extend([1,1,1,1])
    qualified_list.append(metric.f_low_taylor)
    qualified_list.append(metric.low_nmse) 
    qualified_list.append(metric.bias)  
    qualified_list.append(metric.judge_parity())
    qualified_list.append(metric.judge_monotonicity())
    # qualified_list.append(metric.di_jian_flag)
    print('qualified_list = ',qualified_list)
    # X,Y = metric.X, metric.change_Y(Y)
    return metric.nihe_flag,low_polynomial,qualified_list,metric.change_Y(Y)     
    '''


if __name__ == '__main__':
    Global()
    # fileName = "D:\PYcharm_program\Test_everything\Bench_0.15\BenchMark_23.tsv"
    # fileName = "example.tsv"
    for fileNum in range(24,72):
        print("fileNum",fileNum)
        fileName = "D:\PYcharm_program\Test_everything\Feynman\F"+str(fileNum)+ ".tsv"
        X_Y = np.loadtxt(fileName, dtype=np.float, skiprows=1)
        # X_Y_temp = copy.deepcopy(X_Y[1:])
        # for Sampling_proportion in [0.4]:
        #     index_1 = np.random.choice(X_Y_temp.shape[0], int(X_Y_temp.shape[0]*Sampling_proportion), replace=False)#随机不重复采样
        #     print("抽样比例：百分之"+str(Sampling_proportion))
        #     X_Y_temp1 = X_Y_temp[index_1]
        #     X_Y = np.insert(X_Y_temp1,0,X_Y[0],axis=0)
        cal_Taylor_features(varNum=X_Y.shape[1]-1,dataSet=X_Y )


