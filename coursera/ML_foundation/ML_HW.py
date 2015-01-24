#!/usr/bin/env python
import logging
import random
import numpy as np
import time
import argparse
import math

# http://docs.scipy.org/doc/numpy/reference/generated/numpy.matrix.html
# http://docs.scipy.org/doc/numpy/reference/routines.linalg.html
# http://page.mi.fu-berlin.de/rojas/neural/index.html.html#forword
def HW1Q15_17(N, ratio):
    L = GetList('ntumlone_hw1_hw1_15_train.dat')
    total_sum = 0
    for i in range(0,N):
        if N > 1:
            random.seed(i)
            random.shuffle(L)
        wbest, nchange = PLA(L, [1,0,0,0,0], ratio, -1)
        total_sum += nchange
    logging.info("N = %i, total sum = %i, average N of corrections = %i" %(N, total_sum, total_sum/N))

def HW1Q18_20(N, nit, bPLA):
    Ltrain = GetList('ntumlone_hw1_hw1_18_train.dat')
    total_sum = 0
    wbest = [1,0,0,0,0]
    for i in range(0,N):
        if N > 1:
            random.seed(i)
            random.shuffle(Ltrain)
        if bPLA:
            wbest, nchange = PLA(Ltrain, [1,0,0,0,0], 1.0, nit)
        else:
            train_errors, wbest = POCKET(Ltrain, w0=wbest, nit=nit)
        Ltest = GetList('ntumlone_hw1_hw1_18_test.dat')
        total_sum += matrixSum(Ltest, wbest, -1)
    logging.info("N = %i, total sum = %i, average N of mistakes= %i" %(N, total_sum, total_sum/N))

def HW2Q17_18():
    counter = 1
    sum_min_ein = 0.0
    sum_eout = 0.0
    Nit = 5000
    Nit = 1
    for counter in range (0, Nit):
        random.seed(counter)
        dataL = []
        yL = []
        Ndata = 20
        for i in range (0, Ndata):
            dataL.append(random.uniform(-1.0, 1.0))
            noise_const = 1
            if random.uniform(0, 10) <= 2:
                noise_const = -1
            yL.append(noise_const*math.copysign(1.0, dataL[i]))
        min_ein = CalEinEout(dataL, yL)
        sum_eout += (0.5+0.3*min_ein[2]*(abs(min_ein[1])-1))
        sum_min_ein += float(min_ein[0])
    print "Average Ein = %.2f, Eout = %.2f" %(sum_min_ein/Nit, sum_eout/Nit)

def HW2Q19_20():
    Ltrain = GetDataMap('ntumlone_hw2_hw2_train.dat')
    Ltest = GetDataMap('ntumlone_hw2_hw2_test.dat')
    N1D = len(Ltrain[0])
    best_of_best = [N1D, -1.0, 1.0, 0] 
    sum_min_ein = 0.0
    sum_eout = 0.0
    for i in range(0, len(Ltrain)-1):
        min_ein = CalEinEout(Ltrain[i], Ltrain[-1])
        if min_ein[0] < best_of_best[0]:
            best_of_best = list(min_ein) + [i]
    Eout = [0, 0]
    for i in range(0, len(Ltest[0])):
        DecisionStump(best_of_best[1], Ltest[best_of_best[-1]][i], Eout, Ltest[-1][i])
    print "Best Ein = %.2f, Best Eout = %.2f" %(best_of_best[0], min(float(Eout[0])/len(Ltest[0]), float(Eout[1])/len(Ltest[0])))

def HW3Q6_10():
    def E(u,v):
        return (math.exp(u)+math.exp(2*v)+math.exp(u*v)+u*u-2*u*v+2*v*v-3*u-2*v)
    def partial_deriv_u(u, v):
        return (math.exp(u)+v*math.exp(u*v)+2*(u-v)-3)
    def partial_deriv_v(u, v):
        return (2*math.exp(2*v)+u*math.exp(u*v)-2*u+4*v-2)
    def partial_deriv_uu(u, v):
        return (math.exp(u)+v*v*math.exp(u*v)+2)
    def partial_deriv_vv(u, v):
        return (4*math.exp(2*u)+u*u*math.exp(u*v)+4)
    def partial_deriv_uv(u, v):
        return (u*v*math.exp(u*v)-2)
    def Q6():
        eta = 0.01
        uv = [0,0]
        for i in range (1,6):
            x = partial_deriv_u(uv[0], uv[1])
            y = partial_deriv_v(uv[0], uv[1])
            uv = [uv[0]-eta*x, uv[1]-eta*y]
        print E(uv[0], uv[1])
    def Q7(u, v):
        print partial_deriv_uu(0,0)/2, partial_deriv_vv(0,0)/2, partial_deriv_uv(0,0), partial_deriv_u(0,0), partial_deriv_v(0,0), E(u,v), 
    def Q10():
        uv = [0,0]
        for i in range (1,6):
            x = partial_deriv_u(uv[0], uv[1])/partial_deriv_uu(uv[0], uv[1])
            y = partial_deriv_v(uv[0], uv[1])/partial_deriv_vv(uv[0], uv[1])
            uv = [uv[0]-x, uv[1]-y]
        print E(uv[0], uv[1])
    Q10()
def HW3Q13_15():
    sumEin = 0
    sumEout = 0
    Ntest = 100
    W = [np.array([-1, -0.05, 0.08, 0.13, 1.5, 15]),
        np.array([-1, -0.05, 0.08, 0.13, 15, 1.5]),
        np.array([-1, -1.5, 0.08, 0.13, 0.05, 0.05]),
        np.array([-1, -0.05, 0.08, 0.13, 1.5, 1.5])]
    for i in range(0, Ntest):
        X = []
        XFV = []
        Y = []
        counterL = [0]*len(W)
        def genData(i):
            random.seed(i)
            noise = 1.0
            if random.uniform(0, 10) <= 1:
                noise = -1.0
            return random.uniform(-1, 1), random.uniform(-1, 1), noise
        def f(x1, x2):
            return math.copysign(1.0, x1*x1 + x2*x2 - 0.6)
        def feature_vec(x1, x2):
            return [1, x1, x2, x1*x2, x1*x1, x2*x2]
        for i in range(0, 1000):
            x1, x2, noise = genData(i)
            result_13 = f(x1, x2)
            # ========== Q14
            xfv = np.array(feature_vec(x1, x2)).T
            for i in range(0, len(W)):
                result_14 = math.copysign(1.0, np.dot(xfv, W[i]))
                if result_14 != result_13:
                    counterL[i] += 1
            # ========== Q14
            X.append([x1, x2])
            Y.append(noise*result_13)
        Ein_sqr, Ein_class, w = LinReg(X, Y)
        sumEin += Ein_class
        # ========== Q15
        wdelta = W[counterL.index(min(counterL))]
        Xtest = []
        Ytest = []
        for i in range(0, 1000):
            x1, x2, noise = genData(i+Ntest)
            Xtest.append(feature_vec(x1, x2))
            Ytest.append(noise*f(x1, x2))
        Ein_sqr, Ein_class, w = LinReg(Xtest, Ytest, w=wdelta)
        sumEout += Ein_class
        # ========== Q15
    print sumEin/Ntest, wdelta, sumEout/Ntest

def HW3Q18_20():
    Xtrain, Ytrain = GetDataMap('ntumlone-hw3-hw3_train.dat')
    Xtest, Ytest = GetDataMap('ntumlone-hw3-hw3_test.dat')
    #w = LogReg(np.array(Xtrain).T, Ytrain, eta=0.001, T=2000, SGD=True)
    w = LogReg(np.array(Xtrain).T, Ytrain, eta=0.01, T=2000)
    print classerr(np.dot(np.array(Xtest).T, w), Ytest)

def SplitValidadationData(X, Y, n0, n):
    Xval = list(X[n0:n])
    Xtrain = list(X)
    del Xtrain[n0:n]
    Yval = list(Y[n0:n])
    Ytrain = list(Y)
    del Ytrain[n0:n]
    return np.array(Xval), np.array(Yval), np.array(Xtrain), np.array(Ytrain)

def HW4Q19_20():
    _Xtrain, _Ytrain = GetDataMap('ntumlone-hw4-hw4_train.dat')
    Xtest, Ytest = GetDataMap('ntumlone-hw4-hw4_test.dat')
    minDic = {'in': [1, 1, 1, 0, 1], 'out': [1, 1, 1, 0, 1], 'val': [1, 1, 1, 0, 1]}
    Nit = len(_Xtrain)/40
    for loglambda in range(-10, 3):
        lambda_value = math.pow(10, loglambda)
        SumEval = 0
        for i in range(0, Nit):
            Xtrain_val, Ytrain_val, Xtrain, Ytrain = SplitValidadationData(_Xtrain, _Ytrain, i*40, (i+1)*40)
            sqrerr, binerr, wreg = RegularizedLinReg(Xtrain, Ytrain, lambda_value=lambda_value)
            SumEval += classerr(np.dot(Xtrain_val, wreg), Ytrain_val)
        AveEval = SumEval/Nit
        if AveEval <= minDic['val'][-1]:
            minDic['val'][-1] = AveEval
            minDic['val'][3]  = loglambda
    minlambda = math.pow(10, minDic['val'][3])
    sqrerr, binerr, wreg = RegularizedLinReg(np.array(_Xtrain), np.array(_Ytrain), lambda_value=minlambda)
    print binerr, classerr(np.dot(Xtest, wreg), Ytest), minDic['val']

def LogReg(X, Y, **kwargs):
    w = [0]*len(X[0])
    eta = 0.1
    T = 500
    bSGD = False
    for key, value in kwargs.iteritems():
        if key == 'w0':
            w = value
        if key == 'eta':
            eta = value
        if key == 'SGD':
            bSGD = value
        if key == 'T':
            T = value
    def theta_func(s):
        return (1/(1+math.exp(0-s)))
    def cal_s(yn, w, xn):
        return yn*np.dot(np.array(w).T, xn)
    print "Logistic Regression: Initial w = ", w, "eta = ", eta
    counter = 0
    N = len(Y)
    Nx = len(X[0])
    for it in range(0, T):
        _Xvec = [0]*Nx
        if bSGD:
            n = it%1000
            _Xvec = theta_func(cal_s(Y[n], w, X[n]))*(-np.array(Y[n]*X[n]))
        else:
            for n in range(0, N):
                _Xvec = np.array(_Xvec) - theta_func(0-cal_s(Y[n], w, X[n]))*Y[n]*(np.array(X[n]))
            _Xvec = _Xvec/N
        w = np.array(w) - (eta)*_Xvec
        if it%100 == 0:
            print "Logistic Regression, %i: wnext = " %it, w
    return w

def VecSum(V):
    _V = np.array(V)
    return math.sqrt(np.dot(_V, _V))

def NormalizeList(L, N):
    newL=[]
    for item in L:
        newL.append(item/N)
    return newL

def RegularizedLinReg(X, Y, **kwargs):
    lambda_value = 10
    for key, value in kwargs.iteritems():
        if key == 'lambda_value':
             lambda_value = value
    Ndim = len(X[0])
    wreg = np.dot(\
            np.linalg.inv(np.dot(X.T, X) + lambda_value*np.identity(Ndim))\
            ,  np.dot(X.T, Y))
    result_sqr = np.dot(X, wreg) - Y
    Ein=np.dot(result_sqr, result_sqr)/len(Y)
    return Ein, classerr(np.dot(X, wreg), Y), wreg

def LinReg(X, Y, **kwargs):
    Xpsuedo_inv = np.linalg.pinv(X)
    w = np.dot(Xpsuedo_inv, Y)
    for key, value in kwargs.iteritems():
        if key == 'w':
            w = value
    result_sqr = np.dot(X, w) - Y
    Ein=np.dot(result_sqr, result_sqr)/len(Y)
    return Ein, classerr(np.dot(X, w), Y), w

def classerr(Xw, Y):
    sumE = 0
    def err(x):
        return (-0.5)*x+0.5
    for i in range(0, len(Y)):
        sumE += err(math.copysign(1.0, Xw[i]*Y[i]))
    return sumE/len(Y)

def GetList(fin):
    f = open(fin, 'r')
    lines = f.readlines()
    L = []
    for line in lines:
        tmpLine=line.replace("\n","").split("\t")
        L.append([1])
        for item in tmpLine[0].split(" "):
            L[-1].append(float(item))
        L[-1].append(int(tmpLine[-1]))
    return L

def GetDataMap(fin):
    f = open(fin, 'r')
    lines = f.readlines()
    X = []
    Y = []
    for line in lines:
        itemL = line.replace("\n","").split("\t")[0].split(" ")
        X.append([1])
        for i in range (0, len(itemL)-1):
            X[-1].append(float(itemL[i]))
        Y.append(int(itemL[-1]))
    return X, Y #X, Y

def PLA(inputL, w0, ratio, nit):
    logging.debug("Using PLA algorithm with input w0 = ")
    logging.debug(w0)
    counter = 0
    keepLoop = True
    while keepLoop:
        keepLoop = False
        for item in inputL:
            w0, bnext = chooseW(item, w0, ratio)
            if bnext:
                counter += 1
                keepLoop = True
                if nit >= 0 and counter >= nit:
                    return w0, counter
    logging.debug("Weight is adjusted for %i times and the output wbest = " %counter)
    logging.debug(w0)
    return w0, counter

def POCKET(inputL, **kwargs):
    w0 = [1] + [0]*(len(inputL[0])-1)
    nit = 50
    for key, value in kwargs.iteritems():
        if key == 'w0':
            w0 = value
        elif key == 'nit':
            nit = value
    logging.debug("Using POCKET algorithm with input w0 = ")
    logging.debug(w0)
    least_mistakes = len(inputL)
    wbest = list(w0)
    nchange = 0
    while(nchange < nit):
        for item in inputL:
            w0, bnext = chooseW(item, w0, 1.0)
            if bnext:
                _nmistakes = matrixSum(inputL, w0, least_mistakes)
                if _nmistakes < least_mistakes:
                    least_mistakes = _nmistakes
                    wbest = list(w0)
                nchange += 1
                if nchange > nit:
                    break
    logging.debug("Iterate %i times, find least mistakes = %i, best weight = " %(nchange,least_mistakes))
    logging.debug(wbest)
    return least_mistakes, wbest

def chooseW(Xn, Wn, ratio):
    _Yn = np.array([Xn[-1]*ratio]*len(Wn))
    _Xn = np.array(Xn[0:-1])
    _Wn = np.array(Wn)
    result = np.dot(_Wn, _Xn)
    if result*Xn[-1] <= 0:
        logging.debug("Find error: result = %f, yn = %i" %(result, Xn[-1]))
        print _Wn + _Yn*_Xn
        return _Wn + _Yn*_Xn, True
    else:
        return list(Wn), False

def matrixSum(Xn, Wn, threashold):
    Yn = []
    TrimXn = []
    for item in Xn:
        result = 0
        Yn.append(item[-1])
        TrimXn.append(item[0:-1])
    _Yn = np.array(Yn)
    _Xn = np.array(TrimXn)
    _Wn = np.array(Wn)
    inner_product = np.dot(_Xn, _Wn)*Yn
    error_count = 0
    for item in inner_product:
        if threashold > 0 and error_count >= threashold:
            break
        if item < 0:
            error_count += 1
    return error_count

def CalEinEout(dataL, yL):
    min_ein = [len(dataL), 1.0, 1.0] #Assign a big enough initial value
    for item in dataL:
        EIN = [0, 0]
        for i in range(0, len(dataL)):
            DecisionStump(item, dataL[i], EIN, yL[i])
        for j in range(0, len(EIN)):
            EIN[j] = float(EIN[j])/len(dataL)
            if EIN[j] < min_ein[0]:
                min_ein[0] = EIN[j]
                min_ein[1] = item
                min_ein[2] = (-2)*j + 1 #y = -2x + 1, satisfy 0->1.0, 1->-1.0
    return min_ein

def DecisionStump(theta, x, E, fx):
    hx = math.copysign(1.0, (x-theta))
    if fx*hx < 0:
        E[0] += 1 #s = +1
    else:
        E[1] += 1 #s = -1

def timeit(start_time):
    logging.info("Takes total %i seconds" %int(time.time() - start_time))
    return time.time()

def main():
    parser = argparse.ArgumentParser(description='Machine Learning HW')
    parser.add_argument('--debug', action='store_true')
    args = parser.parse_args()
    if args.debug:
        level = logging.DEBUG
    else:
        level = logging.INFO
    logging.basicConfig(level=level, 
        format='%(asctime)s %(levelname)s %(message)s')
    start_time = time.time()
    #HW1Q15_17(1, 1.0) #Q15
    #HW1Q15_17(2000, 1.0) #Q16
    #HW1Q15_17(2000, 0.5) #Q17
    #HW1Q18_20(1, 50, False) #Q18
    #HW1Q18_20(2000, 50, True) #Q19
    #HW1Q18_20(2000, 100, False) #Q20
    #HW2Q17_18()
    #HW2Q19_20()
    #HW3Q6_10()
    #HW3Q13_15()
    #HW3Q18_20()
    HW4Q13_20()
    start_time = timeit(start_time)

if __name__ == "__main__":
    main()
