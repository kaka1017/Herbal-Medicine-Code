# -*- coding: utf-8 -*-
# encoding=utf-8
"""
现在和以前的区别主要是需要引入类的继承进去
    Created on 21:29 2018/11/12 
    @author: Jindong Wang
"""
import numpy as np
import scipy.io
import scipy.linalg
import sklearn.metrics
from sklearn.neighbors import KNeighborsClassifier
import pandas as pd

import numpy as np
import scipy.io
import scipy.linalg
import sklearn.metrics
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder,OneHotEncoder,LabelBinarizer

##### 这一部分是TCA、JDA、DDA的综合实现#########
def kernel(ker, X1, X2, gamma):
    K = None
    if not ker or ker == 'primal':
        K = X1
    elif ker == 'linear':
        if X2 is not None:
            K = sklearn.metrics.pairwise.linear_kernel(
                np.asarray(X1).T, np.asarray(X2).T)
        else:
            K = sklearn.metrics.pairwise.linear_kernel(np.asarray(X1).T)
    elif ker == 'rbf':
        if X2 is not None:
            K = sklearn.metrics.pairwise.rbf_kernel(
                np.asarray(X1).T, np.asarray(X2).T, gamma)
        else:
            K = sklearn.metrics.pairwise.rbf_kernel(
                np.asarray(X1).T, None, gamma)
    return K


class TCA:
    '''
    if you need to change some hyper parameters,you can try these parameters
    Parameters:
    lamba: Regularization weight, which means lamba*|A|^F
    gamma: kernel bandwidth for rbf kernel
    kernel_type: primal(without_kernel_trick),linear,rbf
    '''
    def __init__(self, kernel_type='primal', dim=30, lamb=1, gamma=1):
        '''
        Init func
        :param kernel_type: kernel, values: 'primal' | 'linear' | 'rbf'
        :param dim: dimension after transfer
        :param lamb: lambda value in equation
        :param gamma: kernel bandwidth for rbf kernel
        '''
        self.kernel_type = kernel_type
        self.dim = dim
        self.lamb = lamb
        self.gamma = gamma

    def fit(self, Xs, Xt):
        '''
        Transform Xs and Xt
        :param Xs: ns * n_feature, source feature
        :param Xt: nt * n_feature, target feature
        :return: Xs_new and Xt_new after TCA
        '''
        X = np.hstack((Xs.T, Xt.T))
        # X /= np.linalg.norm(X, axis=0)
        m, n = X.shape
        ns, nt = len(Xs), len(Xt)
        e = np.vstack((1 / ns * np.ones((ns, 1)), -1 / nt * np.ones((nt, 1))))
        M = e * e.T
        M = M / np.linalg.norm(M, 'fro')
        H = np.eye(n) - 1 / n * np.ones((n, n))
        K = kernel(self.kernel_type, X, None, gamma=self.gamma)
        n_eye = m if self.kernel_type == 'primal' else n
        a, b = K @ M @ K.T + self.lamb * np.eye(n_eye), K @ H @ K.T
        w, V = scipy.linalg.eig(a, b)
        ind = np.argsort(w)
        A = V[:, ind[:self.dim]]
        Z = A.T @ K
        Z /= np.linalg.norm(Z, axis=0)

        Xs_new, Xt_new = Z[:, :ns].T, Z[:, ns:].T
        return np.real(Xs_new), np.real(Xt_new)

    def fit_predict(self, Xs, Ys, Xt, Yt):
        '''
        Transform Xs and Xt, then make predictions on target using 1NN
        :param Xs: ns * n_feature, source feature
        :param Ys: ns * 1, source label
        :param Xt: nt * n_feature, target feature
        :param Yt: nt * 1, target label
        :return: Accuracy and predicted_labels on the target domain
        '''
        Xs_new, Xt_new = self.fit(Xs, Xt)
        Xs_new, Xt_new = np.real(Xs_new),np.real(Xt_new)
        clf = KNeighborsClassifier(n_neighbors=1)
        clf.fit(Xs_new, Ys.ravel())
        y_pred = clf.predict(Xt_new)
        acc = sklearn.metrics.accuracy_score(Yt, y_pred)

        return acc, y_pred

    # TCA code is done here. You can ignore fit_new and fit_predict_new.

    def fit_new(self, Xs, Xt, Xt2):
        '''
        Map Xt2 to the latent space created from Xt and Xs
        :param Xs : ns * n_feature, source feature
        :param Xt : nt * n_feature, target feature
        :param Xt2: n_s, n_feature, target feature to be mapped
        :return: Xt2_new, mapped Xt2 with projection created by Xs and Xt
        '''
        # Computing projection matrix A from Xs an Xt
        X = np.hstack((Xs.T, Xt.T))
        X /= np.linalg.norm(X, axis=0)
        m, n = X.shape
        ns, nt = len(Xs), len(Xt)
        e = np.vstack((1 / ns * np.ones((ns, 1)), -1 / nt * np.ones((nt, 1))))
        M = e * e.T
        M = M / np.linalg.norm(M, 'fro')
        H = np.eye(n) - 1 / n * np.ones((n, n))
        K = kernel(self.kernel_type, X, None, gamma=self.gamma)
        n_eye = m if self.kernel_type == 'primal' else n
        a, b = np.linalg.multi_dot(
            [K, M, K.T]) + self.lamb * np.eye(n_eye), np.linalg.multi_dot([K, H, K.T])
        w, V = scipy.linalg.eig(a, b)
        ind = np.argsort(w)
        A = V[:, ind[:self.dim]]

        # Compute kernel with Xt2 as target and X as source
        Xt2 = Xt2.T
        K = kernel(self.kernel_type, X1=Xt2, X2=X, gamma=self.gamma)

        # New target features
        Xt2_new = K @ A

        return Xt2_new

    def fit_predict_new(self, Xt, Xs, Ys, Xt2, Yt2):
        '''
        Transfrom Xt and Xs, get Xs_new
        Transform Xt2 with projection matrix created by Xs and Xt, get Xt2_new
        Make predictions on Xt2_new using classifier trained on Xs_new
        :param Xt: ns * n_feature, target feature
        :param Xs: ns * n_feature, source feature
        :param Ys: ns * 1, source label
        :param Xt2: nt * n_feature, new target feature
        :param Yt2: nt * 1, new target label
        :return: Accuracy and predicted_labels on the target domain
        '''
        Xs_new, _ = self.fit(Xs, Xt)
        Xt2_new = self.fit_new(Xs, Xt, Xt2)
        clf = KNeighborsClassifier(n_neighbors=1)
        clf.fit(Xs_new, Ys.ravel())
        y_pred = clf.predict(Xt2_new)
        acc = sklearn.metrics.accuracy_score(Yt2, y_pred)

        return acc, y_pred

class JDA:
    '''
    if you need to change some hyper parameters,you can try these parameters
    Parameters:
    lamba: Regularization weight, which means lamba*|A|^F
    gamma: kernel bandwidth for rbf kernel
    T:iteration number, because JDA utilized fake 
    '''
    def __init__(self, kernel_type='primal', dim=30, lamb=1, gamma=1, T=10):
        '''
        Init func
        :param kernel_type: kernel, values: 'primal' | 'linear' | 'rbf'
        :param dim: dimension after transfer
        :param lamb: lambda value in equation
        :param gamma: kernel bandwidth for rbf kernel
        :param T: iteration number
        '''
        self.kernel_type = kernel_type
        self.dim = dim
        self.lamb = lamb
        self.gamma = gamma
        self.T = T

    def fit_predict(self, Xs, Ys, Xt, Yt):
        '''
        Transform and Predict using 1NN as JDA paper did
        :param Xs: ns * n_feature, source feature
        :param Ys: ns * 1, source label
        :param Xt: nt * n_feature, target feature
        :param Yt: nt * 1, target label
        :return: acc, y_pred, list_acc
        '''
        list_acc = []
        X = np.hstack((Xs.T, Xt.T))
        X /= np.linalg.norm(X, axis=0)
        m, n = X.shape
        ns, nt = len(Xs), len(Xt)
        e = np.vstack((1 / ns * np.ones((ns, 1)), -1 / nt * np.ones((nt, 1))))
        C = len(np.unique(Ys))
        C_list = np.unique(Ys)
        H = np.eye(n) - 1 / n * np.ones((n, n))

        M = 0
        Y_tar_pseudo = None
        for t in range(self.T):
            print('t = ',t)
            N = 0
            M0 = e * e.T * C
            if Y_tar_pseudo is not None and len(Y_tar_pseudo) == nt:
                for c in range(1, C + 1):
                    e = np.zeros((n, 1))
                    tt = Ys == c
                    print(c)
                    e[np.where(tt == True)] = 1 / len(Ys[np.where(Ys == c)])
                    yy = Y_tar_pseudo == c
                    ind = np.where(yy == True)
                    inds = [item + ns for item in ind]
                    e[tuple(inds)] = -1 / (Y_tar_pseudo == c).sum()
                    e[np.isinf(e)] = 0
                    N = N + np.dot(e, e.T)
            M = M0 + N
            M = M / np.linalg.norm(M, 'fro')
            K = kernel(self.kernel_type, X, None, gamma=self.gamma)
            n_eye = m if self.kernel_type == 'primal' else n
            a, b = np.linalg.multi_dot([K, M, K.T]) + self.lamb * np.eye(n_eye), np.linalg.multi_dot([K, H, K.T])
            w, V = scipy.linalg.eig(a, b)
            ind = np.argsort(w)
            A = V[:, ind[:self.dim]]
            Z = np.dot(A.T, K)
            Z /= np.linalg.norm(Z, axis=0)
            Xs_new, Xt_new = Z[:, :ns].T, Z[:, ns:].T

            clf = KNeighborsClassifier(n_neighbors=1)
            clf.fit(Xs_new, Ys.ravel())
            Y_tar_pseudo = clf.predict(Xt_new)
            acc = sklearn.metrics.accuracy_score(Yt, Y_tar_pseudo)
            list_acc.append(acc)
            print('JDA iteration [{}/{}]: Acc: {:.4f}'.format(t + 1, self.T, acc))
        return acc, Y_tar_pseudo, list_acc

class BDA:
    def __init__(self, kernel_type='primal', dim=30, lamb=1, mu=0.5, gamma=1, T=10, mode='BDA', estimate_mu=False):
        '''
        Init func
        :param kernel_type: kernel, values: 'primal' | 'linear' | 'rbf'
        :param dim: dimension after transfer
        :param lamb: lambda value in equation
        :param mu: mu. Default is -1, if not specificied, it calculates using A-distance
        :param gamma: kernel bandwidth for rbf kernel
        :param T: iteration number
        :param mode: 'BDA' | 'WBDA'
        :param estimate_mu: True | False, if you want to automatically estimate mu instead of manally set it
        '''
        self.kernel_type = kernel_type
        self.dim = dim
        self.lamb = lamb
        self.mu = mu
        self.gamma = gamma
        self.T = T
        self.mode = mode
        self.estimate_mu = estimate_mu

    def fit_predict(self, Xs, Ys, Xt, Yt):
        '''
        Transform and Predict using 1NN as JDA paper did
        :param Xs: ns * n_feature, source feature
        :param Ys: ns * 1, source label
        :param Xt: nt * n_feature, target feature
        :param Yt: nt * 1, target label
        :return: acc, y_pred, list_acc
        '''
        list_acc = []
        X = np.hstack((Xs.T, Xt.T))
        X /= np.linalg.norm(X, axis=0)
        m, n = X.shape
        ns, nt = len(Xs), len(Xt)
        e = np.vstack((1 / ns * np.ones((ns, 1)), -1 / nt * np.ones((nt, 1))))
        C = len(np.unique(Ys))
        H = np.eye(n) - 1 / n * np.ones((n, n))
        mu = self.mu
        M = 0
        Y_tar_pseudo = None
        Xs_new = None
        for t in range(self.T):
            N = 0
            M0 = e * e.T * C
            if Y_tar_pseudo is not None and len(Y_tar_pseudo) == nt:
                for c in range(1, C + 1):
                    e = np.zeros((n, 1))
                    Ns = len(Ys[np.where(Ys == c)])
                    Nt = len(Y_tar_pseudo[np.where(Y_tar_pseudo == c)])

                    if self.mode == 'WBDA':
                        Ps = Ns / len(Ys)
                        Pt = Nt / len(Y_tar_pseudo)
                        alpha = Pt / Ps
                        mu = 1
                    else:
                        alpha = 1

                    tt = Ys == c
                    e[np.where(tt == True)] = 1 / Ns
                    yy = Y_tar_pseudo == c
                    ind = np.where(yy == True)
                    inds = [item + ns for item in ind]
                    e[tuple(inds)] = -alpha / Nt
                    e[np.isinf(e)] = 0
                    N = N + np.dot(e, e.T)

            # In BDA, mu can be set or automatically estimated using A-distance
            # In WBDA, we find that setting mu=1 is enough
            if self.estimate_mu and self.mode == 'BDA':
                if Xs_new is not None:
                    mu = estimate_mu(Xs_new, Ys, Xt_new, Y_tar_pseudo)
                else:
                    mu = 0
            M = (1 - mu) * M0 + mu * N
            M /= np.linalg.norm(M, 'fro')
            K = kernel(self.kernel_type, X, None, gamma=self.gamma)
            n_eye = m if self.kernel_type == 'primal' else n
            a, b = K @ M @ K.T + self.lamb * np.eye(n_eye), K @ H @ K.T
            w, V = scipy.linalg.eig(a, b)
            ind = np.argsort(w)
            A = V[:, ind[:self.dim]]
            Z = A.T @ K
            Z /= np.linalg.norm(Z, axis=0)
            Xs_new, Xt_new = Z[:, :ns].T, Z[:, ns:].T

            clf = sklearn.neighbors.KNeighborsClassifier(n_neighbors=1)
            clf.fit(Xs_new, Ys.ravel())
            Y_tar_pseudo = clf.predict(Xt_new)
            acc = sklearn.metrics.accuracy_score(Yt, Y_tar_pseudo)
            list_acc.append(acc)
            print('{} iteration [{}/{}]: Acc: {:.4f}'.format(self.mode, t + 1, self.T, acc))
        return acc, Y_tar_pseudo, list_acc

class subspace_transfer_learning_base:
    ###这一部分是编写的迁移学习的函数基本类
    def __init__(self,X_S,Y_S,X_T,Y_T):
        self.X_S = X_S #store source domain data
        self.Y_S = Y_S #store source domain label
        self.X_T = X_T #store target domain data
        self.Y_T = Y_T # store target domain label
    def source_domain_scatter(self,data):
        '''source domain scatter matrix'''
        row = data.shape[0]
        H = np.eye(row) - (1 / row) * np.ones((row, row))
        Ss = np.dot(np.dot(data.T,H),data)
        return Ss
    def target_domain_scatter(self,data):
        '''target domain scatter matrix'''
        row = data.shape[0]
        H = np.eye(row) - (1/row)*np.ones((row,row))
        St = np.dot(np.dot(data.T,H),data)
        return St
    def Labelencoder(self,label):
        '''
        onehot encoder
        :param label: label
        :return:
        '''
        label_encoder = LabelEncoder()
        integer_encoded = label_encoder.fit_transform(label)
        onehot_encoder = OneHotEncoder(sparse_output=False)
        integer_encoded = integer_encoded.reshape(len(integer_encoded),1)
        label_bina = LabelBinarizer(neg_label=-1,pos_label=1)
        # onehot_encoded = onehot_encoder.fit_transform(integer_encoded)
        onehot_encoded = label_bina.fit_transform(integer_encoded)
        return onehot_encoded

    def HSIC(self,data,label):
        '''
        Hilbert-Schmidt independence criterion
        :param data: data
        :param label: label
        :return:
        '''
        row = data.shape[0]
        H = np.eye(row) - (1 / row) * np.ones((row, row))
        Y = self.Labelencoder(label)
        Z = np.dot(np.dot(Y.T,H),data)
        hsic = np.dot(Z.T,Z)
        return hsic/(row*row) #这里进行了修改
    def NCCO(self,data,label):
        row = data.shape[0]
        H = np.eye(row) - (1 / row) * np.ones((row, row))
        label_en = LabelBinarizer(neg_label=0,pos_label=1)
        Y = label_en.fit_transform(label)
        Y = Y/sum(Y)
        X1 = H@data
        ncco_matrix = np.linalg.pinv(data)@Y@np.linalg.pinv(Y)@data
        return ncco_matrix #这里进行了修改
    def within_class_scatter(self,data,label):
        '''within class scatter matrix'''
        labelset = set(label)
        dim = data.shape[1]
        row = data.shape[0]
        Sw = np.zeros((dim,dim))

        for i in labelset:
            pos = np.where(label == i)
            X = data[pos]
            possize = np.size(pos)
            mean = np.mean(X,0)
            mean = np.array([mean])
            S = np.dot((X-mean).T,(X-mean))
            # Sw = Sw + (possize/row)*S#陈希错了
            Sw = Sw + (1-possize/row)*S
            # Sw = Sw + S
            # Sw = Sw +  S/(possize*len(labelset))
            # Sw = Sw + S
        return Sw
    def within_class_scatter_normalized(self,data,label):
        '''within class scatter matrix 主要是用来求类内散度矩阵的'''
        labelset = set(label)
        dim = data.shape[1]
        row = data.shape[0]
        Sw = np.zeros((dim,dim))

        for i in labelset:
            pos = np.where(label == i)
            X = data[pos]
            possize = np.size(pos)
            mean = np.mean(X,0)
            mean = np.array([mean])
            vec = X-mean
            vec_norm = np.linalg.norm(vec)
            vec_norm_test = np.linalg.norm(vec,axis=0)
            unit_vec = vec/vec_norm_test
            S = np.dot(unit_vec.T,unit_vec)
            Sw = Sw + S           
        return Sw
    def within_class_scatter_nonweight(self,data,label):
        '''within class scatter matrix'''
        labelset = set(label)
        dim = data.shape[1]
        row = data.shape[0]
        Sw = np.zeros((dim,dim))

        for i in labelset:
            pos = np.where(label == i)
            X = data[pos]
            possize = np.size(pos)
            mean = np.mean(X,0)
            mean = np.array([mean])
            S = np.dot((X-mean).T,(X-mean))
            Sw = Sw + S           
        return Sw
    def between_class_scatter(self,data,label):
        '''between class scatter matrix'''
        labelset = set(label)
        dim = data.shape[1]
        row = data.shape[0]
        Sb = np.zeros((dim,dim))
        total_mean = np.mean(data,0)
        total_mean = np.array([total_mean])

        for i in labelset:
            pos = np.where(label == i)
            X = data[pos]
            possize = np.size(pos)
            mean = np.mean(X,0)
            mean = np.array([mean])
            S = np.dot((mean-total_mean).T,(mean-total_mean))
            # Sb = Sb + (possize / row) * S#陈希错了
            # Sb = Sb + (1-possize/row)*S
            # Sb = Sb + S*(possize/len(labelset))
            Sb = Sb + possize*S
        return Sb
    def between_class_scatter_normalized(self,data,label):
        '''between class scatter matrix_normalized'''
        labelset = set(label)
        dim = data.shape[1]
        row = data.shape[0]
        Sb = np.zeros((dim,dim))
        total_mean = np.mean(data,0)
        total_mean = np.array([total_mean])
        
        for i in labelset:
            pos = np.where(label == i)
            X = data[pos]
            possize = np.size(pos)
            mean = np.mean(X,0)
            mean = np.array([mean])
            vec = mean-total_mean
            vec_norm = np.linalg.norm(vec)
            unit_vec = vec/vec_norm
            S = np.dot((unit_vec).T,(unit_vec))
            Sb = Sb + (possize/row)*S
        return Sb
    # data : N,D
    def LLE_W(self,data, n_dims = 2, n_neighbors = 10):
        '''
        我们需要注意的是LLE中的W矩阵和拉普拉斯中的W矩阵是不一样的，但是这里其实没有用错，
        西南大学他们使用的确实是LLE这个准则
        '''
        def cal_pairwise_dist(x):
        # 以下算法模块调用了LLE模块
        # =============================================================================
        # 输入数据：X是N*D，N为样本量，D为每一个特征上的维度
        # =============================================================================
            N,D = np.shape(x)
            
            dist = np.zeros([N,N])
            
            for i in range(N):
                for j in range(N):
                    dist[i,j] = np.sqrt(np.dot((x[i]-x[j]),(x[i]-x[j]).T))

            #返回任意两个点之间距离
            return dist


        # 获取每个样本点的 n_neighbors个临近点的位置
        def get_n_neighbors(data, n_neighbors = n_neighbors):
        # =============================================================================
        #  这个函数是获取每个点的K个邻居的,返回N*K矩阵，矩阵里面对应的是K个邻居的下标(索引)
        # =============================================================================
            dist = cal_pairwise_dist(data)
            dist[dist < 0] = 0
            N = dist.shape[0] 
            Index = np.argsort(dist,axis=1)[:,1:n_neighbors+1]
            return Index

        N,D = np.shape(data)
        if n_neighbors > D:
            tol = 1e-3
        else:
            tol = 0
        # 获取 n_neighbors个临界点的位置
        Index_NN = get_n_neighbors(data,n_neighbors)
        
        # 计算重构权重
        w = np.zeros([N,n_neighbors])
        ###计算每个点样本点对应的权重
        for i in range(N):
            
            X_k = data[Index_NN[i]]  #K个邻居组成的数据，[k,D]
            X_i = [data[i]]       #单个样本点，[1,D]
            I = np.ones([n_neighbors,1])
            
            Si = np.dot((np.dot(I,X_i)-X_k), (np.dot(I,X_i)-X_k).T)
            
            # 为防止对角线元素过小
            Si = Si+np.eye(n_neighbors)*tol*np.trace(Si)
            
            Si_inv = np.linalg.pinv(Si)
            w[i] = np.dot(I.T,Si_inv)/(np.dot(np.dot(I.T,Si_inv),I))
         
        # 计算 W
        W = np.zeros([N,N])
        for i in range(N):
            W[i,Index_NN[i]] = w[i]
        return W
    def Laplace_Matrix(self,data,n_neighbors):
        '''

        Parameters
        ----------
        data : Data_matrix

        Returns
        -------
        W:weight_matrix
        L:  Laplace_Matrix of Graphs.

        '''
        def cal_pairwise_dist(x):
        # 以下算法模块调用了LLE模块
        # =============================================================================
        # 输入数据：X是N*D，N为样本量，D为每一个特征上的维度
        # =============================================================================
            N,D = np.shape(x)
            
            dist = np.zeros([N,N])
            
            for i in range(N):
                for j in range(N):
                    dist[i,j] = np.sqrt(np.dot((x[i]-x[j]),(x[i]-x[j]).T))

            #返回任意两个点之间距离
            return dist

        def get_n_neighbors(data, n_neighbors):
        # =============================================================================
        #  这个函数是获取每个点的K个邻居的,返回N*K矩阵，矩阵里面对应的是K个邻居的下标(索引)
        # =============================================================================
            dist = cal_pairwise_dist(data)
            dist[dist < 0] = 0
            N = dist.shape[0] 
            Index = np.argsort(dist,axis=1)[:,1:n_neighbors+1]
            return Index
        N,D = np.shape(data)
        Index_NN = get_n_neighbors(data,n_neighbors)
        W = np.zeros([N,N])
        # #下面这一步无非不就是把w装到W里面的过程，明天再写。
        # for i in range(N):
        #     for j in range(n_neighbors):
        #         W[i][Index_NN[j]] = w[i][j]

        '''
        今天我们要做的就是如何去搞定这样的逻辑，直接去大矩阵里面操作,跳出LLE的固定思维
        '''
        for i in range(N):
            for j in range(n_neighbors):
                neighbor_index = Index_NN[i][j]
                point_i = data[i,:]
                point_j = data[neighbor_index,:]
                W[i][neighbor_index] = np.exp(-(np.linalg.norm((point_i-point_j), keepdims=True))**2/1)
        row_sum = np.sum(W,axis=0)
        D = np.diag(row_sum)
        L = D - W 
        return L
       
       
    

class DRCA(subspace_transfer_learning_base):
    def __init__(self,X_S,Y_S,X_T,Y_T):
        super().__init__(X_S,Y_S,X_T,Y_T)
    def func_target_domain_scatter(self,data):
        return super(DRCA, self).target_domain_scatter(data)
    def func_source_domain_scatter(self,data):
        return super(DRCA, self).source_domain_scatter(data)
    def fit_transform(self,Lambda):
        Ms = np.mean(self.X_S,axis=0)
        Mt = np.mean(self.X_T,axis=0)
        dim = np.size(self.X_S,1)
        A = np.zeros((dim,dim))
        Ms = Ms.reshape((1, dim))  # reshape
        Mt = Mt.reshape((1, dim))  # reshape
        tem = np.linalg.pinv(np.dot((Ms-Mt).T,(Ms-Mt)))
        Ss = self.func_source_domain_scatter(self.X_S)  # source domain scatter matrix
        St = self.func_target_domain_scatter(self.X_T)  # target domain scatter matrix
        tem1 = Ss + Lambda*St
        A = A + np.dot(tem,tem1)
        eigenvalue,eigenvector = np.linalg.eig(A)
        eigenvalue = np.real(eigenvalue) # Complex numbers appear during eigen decomposition, which makes it impossible to compare eigenvalues, so take the real part of the eigenvalue and the eigenvector
        Idex = np.argsort(-eigenvalue)
        eigenvector = np.real(eigenvector) # Same as above
        neweigenvector = eigenvector[:,Idex]
        return neweigenvector

        

class DDRCA(subspace_transfer_learning_base):
    '''
    这里的DDRCA里面的几个参数，第一个alpha:用于平衡源域和目标域的能量，beta:类内散度矩阵，delta:类间散度矩阵的考虑
    '''
    def __init__(self,X_S,Y_S,X_T,Y_T):
        super().__init__(X_S,Y_S,X_T,Y_T)
    def func_target_domain_scatter(self,data):
        return super(DDRCA, self).target_domain_scatter(data)
    def func_source_domain_scatter(self,data):
        return super(DDRCA, self).source_domain_scatter(data)
    def func_within_class_scatter_nonweight(self,data,label):
        return super(DDRCA,self).within_class_scatter(data, label)
    def func_between_class_scatter(self,data,label):
        return super(DDRCA,self).between_class_scatter(data,label)
    def fit_transform(self,alpha,lambdas,detla):
        Ms = np.mean(self.X_S,axis=0)
        Mt = np.mean(self.X_T,axis=0)
        dim = np.size(self.X_S, 1)
        Ns = np.size(self.X_S,0)
        Nt = np.size(self.X_T,0)
        M = np.zeros((dim, dim))
        Ss = self.func_source_domain_scatter(self.X_S)
        Sw = self.func_within_class_scatter_nonweight(self.X_S, self.Y_S)
        St = self.func_target_domain_scatter(self.X_T)
        Sb = self.func_between_class_scatter(self.X_S, self.Y_S)
        Ms = Ms.reshape((1, dim))  # reshape
        Mt = Mt.reshape((1, dim))  # reshape
        MDD = np.dot((Ms - Mt).T, (Ms - Mt))  # mean distribution discrepancy minimization
        tem = np.linalg.pinv(MDD)
        tem1 = Ss/Ns + (alpha*St)/Nt - lambdas*Sw + detla*Sb
        M = M + np.dot(tem,tem1)
        eigenvalue, eigenvector = np.linalg.eig(M)
        eigenvalue = np.real(eigenvalue)  # Complex numbers appear during eigen decomposition, which makes it impossible to compare eigenvalues, so take the real part of the eigenvalue and the eigenvector
        Idex = np.argsort(-eigenvalue)
        eigenvector = np.real(eigenvector)# Same as above
        neweigenvector = eigenvector[:, Idex]
        return neweigenvector

class DMDMR(subspace_transfer_learning_base):
    def __init__(self,X_S,Y_S,X_T,Y_T):
        super().__init__(X_S,Y_S,X_T,Y_T)
    def func_target_domain_scatter(self,data):
        return super(DMDMR, self).target_domain_scatter(data)
    def func_source_domain_scatter(self,data):
        return super(DMDMR, self).source_domain_scatter(data)
    def func_hisc(self,data,label):
        return super(DMDMR,self).HSIC(data, label)
    def fit_transform(self,alpha,lambdas,detla):
        Ms = np.mean(self.X_S,axis=0)
        Mt = np.mean(self.X_T,axis=0)
        dim = np.size(self.X_S,1)                #  dimensional
        M = np.zeros((dim,dim))
        Ss = self.func_source_domain_scatter(self.X_S)     # source domain scatter matrix
        St = self.func_target_domain_scatter(self.X_T)     # target domain scatter matrix
        Ms = Ms.reshape((1,dim))
        Mt = Mt.reshape((1,dim))
        MDD = np.dot((Ms-Mt).T,(Ms-Mt))        # mean distribution discrepancy minimization
        hsic = self.func_hisc(self.X_S,self.Y_S)                 # Hilbert-Schmidt independence criterion
        M = M + Ss + alpha*St - lambdas*MDD + detla*hsic # M matrix
        eigenvalue, eigenvector = np.linalg.eig(M)
        index = np.argsort(-eigenvalue)        # max index of eigenvalue
        neweigenvector = eigenvector[:,index]  # reoder
        return neweigenvector


class MCSP(subspace_transfer_learning_base):
    def __init__(self,X_S,Y_S,X_T,Y_T):
        super().__init__(X_S,Y_S,X_T,Y_T)
    def func_target_domain_scatter(self,data):
        return super(MCSP, self).target_domain_scatter(data)
    def func_source_domain_scatter(self,data):
        return super(MCSP, self).source_domain_scatter(data)
    def func_within_class_scatter_nonweight(self,data,label):
        return super(MCSP,self).within_class_scatter(data, label)
    def func_between_class_scatter(self,data,label):
        return super(MCSP,self).between_class_scatter(data,label)
    def func_hisc(self,data,label):
        return super(MCSP,self).HSIC(data, label)
    def fit_transform(self,alpha,lambdas,detla,beta):
        batchS = self.X_S
        batchT = self.X_T
        Ms = np.mean(batchS,axis=0)
        Mt = np.mean(batchT,axis=0)
        Ys = self.Y_S
        dim = np.size(batchS,1)                #  dimensional
        M = np.zeros((dim,dim))
        Ss = self.func_source_domain_scatter(batchS)     # source domain scatter matrix
        Sw = self.func_within_class_scatter_nonweight(batchS,Ys)   # within class scatter matrix
        St = self.func_target_domain_scatter(batchT)     # target domain scatter matrix
        Ms = Ms.reshape((1,dim))               # reshape
        Mt = Mt.reshape((1,dim))               # reshape
        MDD = np.dot((Ms-Mt).T,(Ms-Mt))        # mean distribution discrepancy minimization
        hsic = self.func_hisc(batchS,Ys)                 # Hilbert-Schmidt independence criterion
        M = M + Ss + alpha*St - lambdas*MDD + detla*hsic - beta*Sw # M matrix
        eigenvalue, eigenvector = np.linalg.eig(M)
        eigenvalue = np.real(eigenvalue)  # Complex numbers appear during eigen decomposition, which makes it impossible to compare eigenvalues, so take the real part of the eigenvalue and the eigenvector
        index = np.argsort(-eigenvalue)        # max index of eigenvalue
        eigenvector = np.real(eigenvector)# Same as above
        neweigenvector = eigenvector[:,index]  # reorder
        return neweigenvector

class MCSP_modified(subspace_transfer_learning_base):
    def __init__(self,X_S,Y_S,X_T,Y_T):
        super().__init__(X_S,Y_S,X_T,Y_T)
    def func_target_domain_scatter(self,data):
        return super(MCSP_modified, self).target_domain_scatter(data)
    def func_source_domain_scatter(self,data):
        return super(MCSP_modified, self).source_domain_scatter(data)
    def func_within_class_scatter_nonweight(self,data,label):
        return super(MCSP_modified,self).within_class_scatter(data, label)
    def func_between_class_scatter(self,data,label):
        return super(MCSP_modified,self).between_class_scatter(data,label)
    def func_ncco(self,data,label):
        return super(MCSP_modified,self).NCCO(data, label)
    def fit_transform(self,alpha,lambdas,detla,beta):
        batchS = self.X_S
        batchT = self.X_T
        Ms = np.mean(batchS,axis=0)
        Mt = np.mean(batchT,axis=0)
        Ys = self.Y_S
        dim = np.size(batchS,1)                #  dimensional
        M = np.zeros((dim,dim))
        Ss = self.func_source_domain_scatter(batchS)     # source domain scatter matrix
        Sw = self.func_within_class_scatter_nonweight(batchS,Ys)   # within class scatter matrix
        St = self.func_target_domain_scatter(batchT)     # target domain scatter matrix
        Ms = Ms.reshape((1,dim))               # reshape
        Mt = Mt.reshape((1,dim))               # reshape
        MDD = np.dot((Ms-Mt).T,(Ms-Mt))        # mean distribution discrepancy minimization
        ncco = self.func_ncco(batchS,Ys)                 # Hilbert-Schmidt independence criterion
        M = M + Ss + alpha*St - lambdas*MDD + detla*ncco - beta*Sw # M matrix
        eigenvalue, eigenvector = np.linalg.eig(M)
        eigenvalue = np.real(eigenvalue)  # Complex numbers appear during eigen decomposition, which makes it impossible to compare eigenvalues, so take the real part of the eigenvalue and the eigenvector
        index = np.argsort(-eigenvalue)        # max index of eigenvalue
        eigenvector = np.real(eigenvector)# Same as above
        neweigenvector = eigenvector[:,index]  # reorder
        return neweigenvector

class MCSP3(subspace_transfer_learning_base):
    def __init__(self,X_S,Y_S,X_T,Y_T):
        super().__init__(X_S,Y_S,X_T,Y_T)
    def func_target_domain_scatter(self,data):
        return super(MCSP3, self).target_domain_scatter(data)
    def func_source_domain_scatter(self,data):
        return super(MCSP3, self).source_domain_scatter(data)
    def func_within_class_scatter_nonweight(self,data,label):
        return super(MCSP3,self).within_class_scatter(data, label)
    def func_between_class_scatter(self,data,label):
        return super(MCSP3,self).between_class_scatter(data,label)
    def func_hisc(self,data,label):
        return super(MCSP3,self).HSIC(data, label)
    def fit_transform(self,alpha,beta):
        batchS = self.X_S
        batchT = self.X_T
        Ms = np.mean(batchS,axis=0)
        Mt = np.mean(batchT,axis=0)
        Ys = self.Y_S
        dim = np.size(batchS,1)                #  dimensional
        M = np.zeros((dim,dim))
        Ss = self.func_source_domain_scatter(batchS)     # source domain scatter matrix
        Sw = self.func_within_class_scatter_nonweight(batchS,Ys)   # within class scatter matrix
        St = self.func_target_domain_scatter(batchT)     # target domain scatter matrix
        Ms = Ms.reshape((1,dim))               # reshape
        Mt = Mt.reshape((1,dim))               # reshape
        MDD = np.dot((Ms-Mt).T,(Ms-Mt))        # mean distribution discrepancy minimization
        hsic = self.func_hisc(batchS,Ys)                 # Hilbert-Schmidt independence criterion
        M = M + np.linalg.pinv(MDD)@(Ss + alpha*St- beta*Sw) # M matrix
        eigenvalue, eigenvector = np.linalg.eig(M)
        eigenvalue = np.real(eigenvalue)  # Complex numbers appear during eigen decomposition, which makes it impossible to compare eigenvalues, so take the real part of the eigenvalue and the eigenvector
        index = np.argsort(-eigenvalue)        # max index of eigenvalue
        eigenvector = np.real(eigenvector)# Same as above
        neweigenvector = eigenvector[:,index]  # reorder
        return neweigenvector

class CSDL(subspace_transfer_learning_base):
    def __init__(self,X_S,Y_S,X_T,Y_T):
        super().__init__(X_S,Y_S,X_T,Y_T)
        print("CSDL修正")
    def func_target_domain_scatter(self,data):
        return data.T@data
    def func_source_domain_scatter(self,data):
        return data.T@data
    def func_within_class_scatter_nonweight(self,data,label):
        '''within class scatter matrix'''
        labelset = set(label)
        dim = data.shape[1]
        row = data.shape[0]
        Sw = np.zeros((dim,dim))

        for i in labelset:
            pos = np.where(label == i)
            X = data[pos]
            possize = np.size(pos)
            mean = np.mean(X,0)
            mean = np.array([mean])
            S = np.dot((X-mean).T,(X-mean))
            Sw = Sw + S
        return Sw
    def func_between_class_scatter(self,data,label):
        '''between class scatter matrix'''
        labelset = set(label)
        dim = data.shape[1]
        row = data.shape[0]
        Sb = np.zeros((dim,dim))
        total_mean = np.mean(data,0)
        total_mean = np.array([total_mean])

        for i in labelset:
            pos = np.where(label == i)
            X = data[pos]
            possize = np.size(pos)
            mean = np.mean(X,0)
            mean = np.array([mean])
            S = np.dot((mean-total_mean).T,(mean-total_mean))
            Sb = Sb + possize*S
        return Sb
    def func_LLE_W(self,data,n_dims=2,n_neighbors=10):
        return super(CSDL,self).LLE_W(data,n_dims=n_dims,n_neighbors=n_neighbors)
    def fit_transform(self,lamda,neighbors):
        W = self.func_LLE_W(self.X_S,n_neighbors=neighbors) #计算LLE中的W矩阵
        '''
        一定需要注意的是这里的W矩阵是否具有对称性，为了说明这一点，我们来举一个简单的例子
        现在有3个点,假设K近邻中的参数K取1
        A,B,C，首先我们来生成邻接矩阵,A和B是邻居，那么B和A也一定是吗？显然不是
        因此在这里我们默认认为W矩阵是对称是部队的
        '''
        Ms = np.mean(self.X_S,axis=0)
        Mt = np.mean(self.X_T,axis=0)
        dim = np.size(self.X_S, 1)
        Ns = np.size(self.X_S,0)
        Nt = np.size(self.X_T,0)
        M = np.zeros((dim, dim))
        Sw = self.func_within_class_scatter_nonweight(self.X_S, self.Y_S)
        Sb = self.func_between_class_scatter(self.X_S, self.Y_S)
        Ms = Ms.reshape((1, dim))  # reshape
        Mt = Mt.reshape((1, dim))  # reshape
        MDD = np.dot((Ms - Mt).T, (Ms - Mt))  # mean distribution discrepancy minimization
        LLE_1 = np.dot(np.identity(Ns)-W,(np.identity(Ns)-W).T)
        LLE_2 = np.dot(self.X_S.T,LLE_1)
        LLE_3 = np.dot(LLE_2,self.X_S)
        LLE_4 = Sw+ LLE_3 + MDD #主要是用来计算B矩阵的
        tem = np.linalg.pinv(LLE_4) #计算inv(B)
        tem1 = Sb + np.dot(self.X_S.T,self.X_S) + lamda * np.dot(self.X_T.T,self.X_T) #计算A
        M = M + np.dot(tem,tem1)
        eigenvalue, eigenvector = np.linalg.eig(M)
        eigenvalue = np.real(eigenvalue)  # Complex numbers appear during eigen decomposition, which makes it impossible to compare eigenvalues, so take the real part of the eigenvalue and the eigenvector
        Idex = np.argsort(-eigenvalue)
        eigenvector = np.real(eigenvector)# Same as above
        neweigenvector = eigenvector[:, Idex]
        return neweigenvector

class UDA_CA(subspace_transfer_learning_base):
    '''
    这是刘涛老师提出的方法
    Liu T, Zhu X, Wang Q. Domain Adaptation on Asymmetric Drift Data for an 
    Electronic Nose[J]. IEEE Transactions on Instrumentation and Measurement, 2023.

    '''
    def __init__(self,X_S,Y_S,X_T,Y_T):
        super().__init__(X_S,Y_S,X_T,Y_T)
    def func_Laplace_Matrix(self,data, n_neighbors):
        return super(UDA_CA,self).Laplace_Matrix(data, n_neighbors)
    def fit_transform(self,n_neighbors,alpha,lamba,beta,max_inter):
        '''
        一定需要注意的是这里的W矩阵是否具有对称性，为了说明这一点，我们来举一个简单的例子
        现在有3个点,假设K近邻中的参数K取1
        A,B,C，首先我们来生成邻接矩阵,A和B是邻居，那么B和A也一定是吗？显然不是
        因此在这里我们默认认为W矩阵是对称是部队的
        '''
        # W = W.T #这里不应该取转置符号
        XS = self.X_S.T #D*NS
        XT = self.X_T.T #D*NT
        NS = np.size(XS, 1)
        NT = np.size(XT,1)
        W = np.zeros([NS,NT]) #NS*NT
        LS = self.func_Laplace_Matrix(XS.T, n_neighbors) #source_domain_Laplace,这里因为有个转置在这 NS*NS
        LT = self.func_Laplace_Matrix(XT.T, n_neighbors) #target_domain_Laplace NT*NT
        for i in range(max_inter):
            tem1a = np.dot(XS,W) #D*NT
            tem1 = np.dot(XT,XT.T)+np.dot(tem1a,tem1a.T)
            tem2 = np.dot(np.dot(XT,LT),XT.T) + np.dot(np.dot(XS,LS),XS.T)
            tem3 = np.dot(np.dot(XS,W),XT.T) + np.dot(np.dot(XT,W.T),XS.T)
            formula1 = alpha*tem1 + lamba*tem2 - (1/2*alpha+beta)*tem3
            eigenvalue, eigenvector = np.linalg.eig(formula1)
            eigenvalue = np.real(eigenvalue)  # Complex numbers appear during eigen decomposition, which makes it impossible to compare eigenvalues, so take the real part of the eigenvalue and the eigenvector
            Idex = np.argsort(-eigenvalue)
            eigenvector = np.real(eigenvector)# Same as above
            neweigenvector = eigenvector[:, Idex]
            #上述第一阶段执行完毕，下面重点是计算矩阵B，到底应该如何执行的问题
            B = np.linalg.norm(W,𝑎𝑥𝑖𝑠=1,keepdims=True)
            tem4a = np.dot(XS.T,neweigenvector)
            tem4 = np.linalg.pinv(B + 2*alpha* np.dot(tem4a,tem4a.T))
            tem5 = np.dot(neweigenvector.T,XT)
            W = (2*alpha+beta)*np.dot(np.dot(tem4,tem4a),tem5)
        
        return neweigenvector
    
class DAST(subspace_transfer_learning_base):
    '''
    这是郭坦(重庆邮电大学通信与信息工程学院)和谭院长共同提出的
    Guo T, Tan X, Yang L, et al. Domain adaptive subspace transfer model for sensor drift compensation in biologically inspired electronic nose[J]. Expert Systems with Applications, 2022, 208: 118237.

    '''
# =============================================================================
# 从基本类中继承所需的模块
# =============================================================================
    def __init__(self,X_S,Y_S,X_T,Y_T):
        super().__init__(X_S,Y_S,X_T,Y_T)
    def func_Laplace_Matrix(self,data, n_neighbors):
        return super(DAST,self).Laplace_Matrix(data, n_neighbors)
    def fit_transform(self,neta,mu,n_neighbors,max_inter,alpha,beta,gamma):
        '''
        neta:用于稀疏表示的权重
        mu:求解子问题1中使用的动态参数
        n_neighbors:计算图拉普拉斯矩阵时使用的k近邻数
        max_inter：外层迭代次数
        alpha:保持重构关系
        beta:保持目标域流形结构
        gamma:保持源域流形结构
        
        '''
        def soft_threshold_function(matrix,thread):
            '''
            

            Parameters
            ----------
            matrix : input a matrix,apply soft_thread_function to each element
            of matrix
            thread: threading value of soft-ranged function
            Returns
            -------
            Matrix after transformation

            '''
            for i in range(len(matrix)):
                for j in range(len(matrix[0])):
                    if abs(matrix[i][j])<thread:
                        matrix[i][j] = 0
                    elif matrix[i][j]> thread:
                        matrix[i][j] = matrix[i][j] - thread
                    else:
                        matrix[i][j] = matrix[i][j] + thread
            return matrix
# =============================================================================
# 算法执行迭代求解的子问题1
# =============================================================================
        # W = W.T #这里不应该取转置符号
        XS = self.X_S.T #D*NS
        XT = self.X_T.T #D*NT
        X = np.hstack([XS,XT])
        NS = np.size(XS, 1)
        NT = np.size(XT,1)
        Z = np.ones([NS,NT])
        R = np.zeros([NS,NT])
        Y = np.zeros([NS,NT])
        W = np.zeros([NS,NT]) #NS*NT
        D = np.size(XS, 0)
        d = D #这个参数放在这只是为了测试
        rho = 2
        mu_max = 1000
        P = np.ones([D,d])
        LS = self.func_Laplace_Matrix(XS.T, n_neighbors) #source_domain_Laplace,这里因为有个转置在这 NS*NS
        LT = self.func_Laplace_Matrix(XT.T, n_neighbors) #target_domain_Laplace NT*NT
        for i in range(max_inter):
            #step 1: update Z
            error = 0.1
            while(np.linalg.norm(Z-R,ord=np.Inf)>error):
                Z = 2*neta*np.linalg.pinv( ((P.T@XS).T)@P.T@XS + mu*np.identity(NS) )@(mu*R-Y +2*neta*(P.T@XS).T@P.T@XT)
                R = soft_threshold_function(Z+Y/mu,0.01)
                Y = Y+ mu*(Z-R)
                mu = min(rho*mu,mu_max)
    # =============================================================================
    # 算法执行迭代求解的子问题2
    # =============================================================================
            J = (XT - XS@Z)@( (XT - XS@Z).T )
            tem = neta*J - alpha*(X@X.T) + beta* XT@LT@XT.T +gamma * XS@LS@XS.T
            eigenvalue, eigenvector = np.linalg.eig(tem)
            Idex = np.argsort(-eigenvalue)
            eigenvector = np.real(eigenvector)# Same as above
            P = eigenvector[:, Idex[0:d]] #UPDATE P
        
        
        return P
    
class LDSP(subspace_transfer_learning_base):
    def __init__(self,X_S,Y_S,X_T,Y_T):
        super().__init__(X_S,Y_S,X_T,Y_T)
    def func_target_domain_scatter(self,data):
        return super(LDSP, self).target_domain_scatter(data)
    def func_source_domain_scatter(self,data):
        return super(LDSP, self).source_domain_scatter(data)
    def calcuate_slbc(self):
        X_S = self.X_S
        Ns = self.X_S.shape[0]
        D = self.X_S.shape[1]
        A = np.zeros([Ns,Ns])
        W_lbc = np.zeros([Ns,Ns])
        ## 计算矩阵A ###
        for i in range(Ns):
            for j in range(Ns):
                point_i = X_S[i,:]
                point_j = X_S[j,:]
                A[i][j] = np.exp(-(np.linalg.norm((point_i-point_j), keepdims=True))**2/1)
          ## 计算矩阵A ###
         
          ## 计算矩阵Wlbc ###
        for i in range(Ns):
            for j in range(Ns):
                if self.Y_S[i] == self.Y_S[j]:
                    pos = np.where(self.Y_S == self.Y_S[j])
                    X = self.X_S[pos]
                    nl = np.size(pos)
                    W_lbc[i][j] = A[i][j]*(1/Ns-1/nl) #这里的nl指每一类中的样本量
                else:
                    W_lbc[i][j] = 1/Ns 
         
          ## 计算矩阵Wlbc ###
        slbc =np.zeros([D,D])
        for i in range(Ns):
            for j in range(Ns):
                slbc = slbc + 1/2*W_lbc[i][j]*np.dot((self.X_S[i,:]-self.X_S[j,:]).T.reshape(-1,1),(self.X_S[i,:]-self.X_S[j,:]).reshape(1,-1))
        return slbc
    def calcuate_slwc(self):
        X_S = self.X_S
        Ns = self.X_S.shape[0]
        D = self.X_S.shape[1]
        A = np.zeros([Ns,Ns])
        W_lwc = np.zeros([Ns,Ns])
         ## 计算矩阵A ###
        for i in range(Ns):
            for j in range(Ns):
                point_i = X_S[i,:]
                point_j = X_S[j,:]
                A[i][j] = np.exp(-(np.linalg.norm((point_i-point_j), keepdims=True))**2/1)
           ## 计算矩阵A ###
          
           ## 计算矩阵Wlbc ###
        for i in range(Ns):
            for j in range(Ns):
                if self.Y_S[i] == self.Y_S[j]:
                    pos = np.where(self.Y_S == self.Y_S[j])
                    X = self.X_S[pos]
                    nl = np.size(pos)
                    W_lwc[i][j] = A[i][j]/nl #这里的nl指每一类中的样本量
                else:
                    W_lwc[i][j] = 0
          
           ## 计算矩阵Wlbc ###
        slwc =np.zeros([D,D])
        for i in range(Ns): 
            for j in range(Ns):
                slwc = slwc + 1/2*W_lwc[i][j]*np.dot((self.X_S[i,:]-self.X_S[j,:]).T.reshape(-1,1),(self.X_S[i,:]-self.X_S[j,:]).reshape(1,-1))
        return slwc
    def fit_transform(self,Lambda,k,u,neta):
        Ns = self.X_S.shape[0] #获取源域数据和目标域数据的样本量
        Nt = self.X_T.shape[0]
        Ms = np.mean(self.X_S,axis=0)
        Mt = np.mean(self.X_T,axis=0)
        dim = np.size(self.X_S,1)
        A = np.zeros((dim,dim))
        Ms = Ms.reshape((1, dim))  # reshape
        Mt = Mt.reshape((1, dim))  # reshape
        tem = neta*np.linalg.pinv(np.dot((Ms-Mt).T,(Ms-Mt)))
        # Ss = self.func_source_domain_scatter(self.X_S)  # source domain scatter matrix
        # St = self.func_target_domain_scatter(self.X_T)  # target domain scatter matrix
        Slwc = self.calcuate_slwc()
        Slbc = self.calcuate_slbc()
        tem1 = 1/Ns*self.X_S.T@self.X_S + 1/Nt*Lambda*self.X_T.T@self.X_T-k*Slwc+u*Slbc
        A = A + np.dot(tem,tem1)
        eigenvalue,eigenvector = np.linalg.eig(A)
        eigenvalue = np.real(eigenvalue) # Complex numbers appear during eigen decomposition, which makes it impossible to compare eigenvalues, so take the real part of the eigenvalue and the eigenvector
        Idex = np.argsort(-eigenvalue)
        eigenvector = np.real(eigenvector) # Same as above
        neweigenvector = eigenvector[:,Idex]
        return neweigenvector
class LDSP_constrained(subspace_transfer_learning_base):
    def __init__(self, X_S, Y_S, X_T, Y_T):
        super().__init__(X_S, Y_S, X_T, Y_T)
        self.Slbc = self.calcuate_slbc()
        self.Slwc = self.calcuate_slwc()

    def func_target_domain_scatter(self, data):
        return super(LDSP_constrained, self).target_domain_scatter(data)

    def func_source_domain_scatter(self, data):
        return super(LDSP_constrained, self).source_domain_scatter(data)

    def calcuate_slbc(self):
        X_S = self.X_S
        Ns = self.X_S.shape[0]
        D = self.X_S.shape[1]
        A = np.zeros([Ns, Ns])
        W_lbc = np.zeros([Ns, Ns])
        ## 计算矩阵A ###
        for i in range(Ns):
            for j in range(Ns):
                point_i = X_S[i, :]
                point_j = X_S[j, :]
                A[i][j] = np.exp(-(np.linalg.norm((point_i - point_j), keepdims=True)) ** 2 / 1)
        ## 计算矩阵A ###

        ## 计算矩阵Wlbc ###
        for i in range(Ns):
            for j in range(Ns):
                if self.Y_S[i] == self.Y_S[j]:
                    pos = np.where(self.Y_S == self.Y_S[j])
                    X = self.X_S[pos]
                    nl = np.size(pos)
                    W_lbc[i][j] = A[i][j] * (1 / Ns - 1 / nl)  # 这里的nl指每一类中的样本量
                else:
                    W_lbc[i][j] = 1 / Ns

                    ## 计算矩阵Wlbc ###
        slbc = np.zeros([D, D])
        for i in range(Ns):
            for j in range(Ns):
                slbc = slbc + 1 / 2 * W_lbc[i][j] * np.dot((self.X_S[i, :] - self.X_S[j, :]).T.reshape(-1, 1),
                                                           (self.X_S[i, :] - self.X_S[j, :]).reshape(1, -1))
        return slbc

    def calcuate_slwc(self):
        X_S = self.X_S
        Ns = self.X_S.shape[0]
        D = self.X_S.shape[1]
        A = np.zeros([Ns, Ns])
        W_lwc = np.zeros([Ns, Ns])
        ## 计算矩阵A ###
        for i in range(Ns):
            for j in range(Ns):
                point_i = X_S[i, :]
                point_j = X_S[j, :]
                A[i][j] = np.exp(-(np.linalg.norm((point_i - point_j), keepdims=True)) ** 2 / 1)
        ## 计算矩阵A ###

        ## 计算矩阵Wlbc ###
        for i in range(Ns):
            for j in range(Ns):
                if self.Y_S[i] == self.Y_S[j]:
                    pos = np.where(self.Y_S == self.Y_S[j])
                    X = self.X_S[pos]
                    nl = np.size(pos)
                    W_lwc[i][j] = A[i][j] / nl  # 这里的nl指每一类中的样本量
                else:
                    W_lwc[i][j] = 0

        ## 计算矩阵Wlbc ###
        slwc = np.zeros([D, D])
        for i in range(Ns):
            for j in range(Ns):
                slwc = slwc + 1 / 2 * W_lwc[i][j] * np.dot((self.X_S[i, :] - self.X_S[j, :]).T.reshape(-1, 1),
                                                           (self.X_S[i, :] - self.X_S[j, :]).reshape(1, -1))
        return slwc

    def fit_transform(self, Lambda, k, u, beta):
        Ns = self.X_S.shape[0]  # 获取源域数据和目标域数据的样本量
        Nt = self.X_T.shape[0]
        Ms = np.mean(self.X_S, axis=0)
        Mt = np.mean(self.X_T, axis=0)
        dim = np.size(self.X_S, 1)
        A = np.zeros((dim, dim))
        Ms = Ms.reshape((1, dim))  # reshape
        Mt = Mt.reshape((1, dim))  # reshape
        tem = np.linalg.pinv(np.dot((Ms - Mt).T, (Ms - Mt)))
        # Ss = self.func_source_domain_scatter(self.X_S)  # source domain scatter matrix
        # St = self.func_target_domain_scatter(self.X_T)  # target domain scatter matrix
        Slwc = self.Slwc
        Slbc = self.Slbc
        tem1 = 1 / Ns * self.X_S.T @ self.X_S + 1 / Nt * Lambda * self.X_T.T @ self.X_T - k * Slwc + u * Slbc - beta * tem
        A = A + tem1
        eigenvalue, eigenvector = np.linalg.eig(A)
        eigenvalue = np.real(eigenvalue)  # Complex numbers appear during eigen decomposition, which makes it impossible to compare eigenvalues, so take the real part of the eigenvalue and the eigenvector
        Idex = np.argsort(-eigenvalue)
        eigenvector = np.real(eigenvector)  # Same as above
        neweigenvector = eigenvector[:, Idex]
        return neweigenvector
class My_algorithm1(subspace_transfer_learning_base):
    ####
    #Liu S, Feng L, Qiao H. Scatter balance: An angle-based supervised dimensionality reduction[J]. IEEE transactions on neural networks and learning systems, 2014, 26(2): 277-289.
    #这样处理明显的一个不好的地方可能是因为没有对类内散度矩阵和类间散度矩阵单独进行考量，导致整体的算法识别效果较差
    ####
     def __init__(self,X_S,Y_S,X_T,Y_T):
         super().__init__(X_S,Y_S,X_T,Y_T)
     def func_target_domain_scatter(self,data):
         return super(My_algorithm1, self).target_domain_scatter(data)
     def func_source_domain_scatter(self,data):
         return super(My_algorithm1, self).source_domain_scatter(data)
     def func_within_class_scatter_normalized(self,data,label):
         return super(My_algorithm1,self).within_class_scatter_normalized(data, label)
     def func_between_class_scatter_normalized(self,data,label):
         return super(My_algorithm1,self).between_class_scatter_normalized(data,label)
     def fit_transform(self,alpha,lambdas,beta):
         Ms = np.mean(self.X_S,axis=0)
         Mt = np.mean(self.X_T,axis=0)
         dim = np.size(self.X_S, 1)
         Ns = np.size(self.X_S,0)
         Nt = np.size(self.X_T,0)
         M = np.zeros((dim, dim))
         Ss = self.func_source_domain_scatter(self.X_S)
         Sw_normalized = self.func_within_class_scatter_normalized(self.X_S, self.Y_S)
         St = self.func_target_domain_scatter(self.X_T)
         Sb_normalized = self.func_between_class_scatter_normalized(self.X_S, self.Y_S)
         Ms = Ms.reshape((1, dim))  # reshape
         Mt = Mt.reshape((1, dim))  # reshape
         MDD = np.dot((Ms - Mt).T, (Ms - Mt))  # mean distribution discrepancy minimization
         M = Ss/Ns + (alpha*St)/Nt - lambdas*MDD+beta*(1/dim*np.identity(dim)-Sw_normalized+Sb_normalized)
         eigenvalue, eigenvector = np.linalg.eig(M)
         eigenvalue = np.real(eigenvalue)  # Complex numbers appear during eigen decomposition, which makes it impossible to compare eigenvalues, so take the real part of the eigenvalue and the eigenvector
         Idex = np.argsort(-eigenvalue)
         eigenvector = np.real(eigenvector)# Same as above
         neweigenvector = eigenvector[:, Idex]
         return neweigenvector

class My_algorithm2(subspace_transfer_learning_base):
    ####
    #Graph Embedded Nonparametric Mutual Information For Supervised Dimensionality Reduction
    #这样处理明显的一个不好的地方可能是因为没有对类内散度矩阵和类间散度矩阵单独进行考量，导致整体的算法识别效果较差
    ####
     def __init__(self,X_S,Y_S,X_T,Y_T):
         super().__init__(X_S,Y_S,X_T,Y_T)
     def func_target_domain_scatter(self,data):
         return super(My_algorithm2, self).target_domain_scatter(data)
     def func_source_domain_scatter(self,data):
         return super(My_algorithm2, self).source_domain_scatter(data)
     def func_within_class_scatter_normalized(self,data,label):
         return super(My_algorithm2,self).within_class_scatter_normalized(data, label)
     def func_between_class_scatter_normalized(self,data,label):
         return super(My_algorithm2,self).between_class_scatter_normalized(data,label)
     def func_calculateM(self,data,label):
         labelset = set(label)
         dim = data.shape[1]
         row = data.shape[0]
         Sw = np.zeros((dim,dim))
         C_ALL = 0
         C_IN = 1/(row*row)
         C_BTW = 0 #这里除以了数据量的立方
         for i in labelset:
             pos = np.where(label == i)
             X = data[pos]
             possize = np.size(pos)
             C_ALL = C_ALL + 1/(row**4)*(possize**2)
         for i in labelset:
             pos = np.where(label == i)
             X = data[pos]
             possize = np.size(pos)
         M = np.zeros([row,row]) #initialize the matrix
         tem2 = np.zeros([row,row])
         tem3 = np.zeros([1,row])
         M = M + C_ALL*np.ones([row,1])@(np.ones([row,1]).T)
         for i in labelset:
             var_1c = np.zeros([row,1])
             pos = np.where(label == i)
             var_1c[pos] = 1
             tem2 += var_1c@(var_1c.T)
             C_BTW_c = possize/(row**3)
             tem3 += (C_BTW_c*(var_1c.T))
         tem2 = tem2 /(row**2)
         tem33 = np.ones([row,1])@tem3
         M =M + tem2 - 2* tem33
         return M
     def fit_transform(self,alpha,lambdas,beta):
         Ms = np.mean(self.X_S,axis=0)
         Mt = np.mean(self.X_T,axis=0)
         dim = np.size(self.X_S, 1)
         Ns = np.size(self.X_S,0)
         Nt = np.size(self.X_T,0)
         M = np.zeros((dim, dim))
         Ss = self.func_source_domain_scatter(self.X_S)
         Sw_normalized = self.func_within_class_scatter_normalized(self.X_S, self.Y_S)
         St = self.func_target_domain_scatter(self.X_T)
         Sb_normalized = self.func_between_class_scatter_normalized(self.X_S, self.Y_S)
         Ms = Ms.reshape((1, dim))  # reshape
         Mt = Mt.reshape((1, dim))  # reshape
         MDD = np.dot((Ms - Mt).T, (Ms - Mt))  # mean distribution discrepancy minimization
         M_matrix = self.func_calculateM(self.X_S, self.Y_S)
         M_modified = (M_matrix+M_matrix.T)/2
         M = Ss/Ns + (alpha*St)/Nt - lambdas*MDD+beta*(self.X_S.T)@M_modified@(self.X_S)
         eigenvalue, eigenvector = np.linalg.eig(M)
         eigenvalue = np.real(eigenvalue)  # Complex numbers appear during eigen decomposition, which makes it impossible to compare eigenvalues, so take the real part of the eigenvalue and the eigenvector
         Idex = np.argsort(-eigenvalue)
         eigenvector = np.real(eigenvector)# Same as above
         neweigenvector = eigenvector[:, Idex]
         return neweigenvector
     

class DDRCA_Dinkelbach(subspace_transfer_learning_base):
    '''
    这里的DDRCA里面的几个参数，第一个alpha:用于平衡源域和目标域的能量，beta:类内散度矩阵，delta:类间散度矩阵的考虑
    '''
    def __init__(self,X_S,Y_S,X_T,Y_T):
        super().__init__(X_S,Y_S,X_T,Y_T)
    def func_target_domain_scatter(self,data):
        return super(DDRCA_Dinkelbach, self).target_domain_scatter(data)
    def func_source_domain_scatter(self,data):
        return super(DDRCA_Dinkelbach, self).source_domain_scatter(data)
    def func_within_class_scatter_nonweight(self,data,label):
        return super(DDRCA_Dinkelbach,self).within_class_scatter(data, label)
    def func_between_class_scatter(self,data,label):
        return super(DDRCA_Dinkelbach,self).between_class_scatter(data,label)
    def fit_transform(self,alpha,beta,detla):
        Ms = np.mean(self.X_S,axis=0)
        Mt = np.mean(self.X_T,axis=0)
        dim = np.size(self.X_S, 1)
        Ns = np.size(self.X_S,0)
        Nt = np.size(self.X_T,0)
        M = np.zeros((dim, dim))
        Ss = self.func_source_domain_scatter(self.X_S)
        Sw = self.func_within_class_scatter_nonweight(self.X_S, self.Y_S)
        St = self.func_target_domain_scatter(self.X_T)
        Sb = self.func_between_class_scatter(self.X_S, self.Y_S)
        Ms = Ms.reshape((1, dim))  # reshape
        Mt = Mt.reshape((1, dim))  # reshape
        MDD = np.dot((Ms - Mt).T, (Ms - Mt))  # mean distribution discrepancy minimization
        # mu = np.trace(MDD + beta*Sw) / np.trace(1/Ns*Ss + 1/Nt*alpha*St + detla*Sb) #计算u0,作为迭代的初始值
        P = np.identity(dim) #定义初值
        mu =  np.trace(P.T@(1/Ns*Ss + 1/Nt*alpha*St + detla*Sb)@P)/np.trace(P.T@(MDD + beta*Sw)@P)#更新下一轮的P
        F = 5000 #收敛条件的判断
        error = 1e-5
        #### 计算后续需要使用到的值#######
        while(F>error):
            M = 1/Ns*Ss + 1/Nt*alpha*St + detla*Sb - mu*(MDD + beta*Sw)
            eigenvalue, eigenvector = np.linalg.eig(M)
            eigenvalue = np.real(eigenvalue)  # Complex numbers appear during eigen decomposition, which makes it impossible to compare eigenvalues, so take the real part of the eigenvalue and the eigenvector
            Idex = np.argsort(-eigenvalue)
            if min(eigenvalue)<0:
                M = M -min(eigenvalue)*np.identity(dim)
                eigenvalue, eigenvector = np.linalg.eig(M)
                eigenvalue = np.real(eigenvalue)  # Complex numbers appear during eigen decomposition, which makes it impossible to compare eigenvalues, so take the real part of the eigenvalue and the eigenvector
                Idex = np.argsort(-eigenvalue)
            eigenvector = np.real(eigenvector)# Same as above
            P = eigenvector[:, Idex] #这里其实就是我们迭代过程中用到的P矩阵
            F = np.trace(P.T@(1/Ns*Ss + 1/Nt*alpha*St + detla*Sb - mu*(MDD + beta*Sw))@P) #迭代的收敛条件
            # print('本轮迭代中的函数值',F,'mu=',mu,mu-mu0)
            mu =  np.trace(P.T@(1/Ns*Ss + 1/Nt*alpha*St + detla*Sb)@P)/np.trace(P.T@(MDD + beta*Sw)@P)#更新下一轮的P
        return P

class DDRCA_fake(subspace_transfer_learning_base):
    '''
    这里的DDRCA里面的几个参数，第一个alpha:用于平衡源域和目标域的能量，beta:类内散度矩阵，delta:类间散度矩阵的考虑
    '''
    def __init__(self,X_S,Y_S,X_T,Y_T):
        super().__init__(X_S,Y_S,X_T,Y_T)
    def func_target_domain_scatter(self,data):
        # return super(DDRCA_fake, self).target_domain_scatter(data)
        return data.T@data
    def func_source_domain_scatter(self,data):
        return data.T@data
    def func_within_class_scatter_nonweight(self,data,label):
        '''within class scatter matrix'''
        labelset = set(label)
        dim = data.shape[1]
        row = data.shape[0]
        Sw = np.zeros((dim,dim))

        for i in labelset:
            pos = np.where(label == i)
            X = data[pos]
            possize = np.size(pos)
            mean = np.mean(X,0)
            mean = np.array([mean])
            S = np.dot((X-mean).T,(X-mean))
            Sw = Sw + (1-possize/row)*S
        return Sw
    def func_between_class_scatter(self,data,label):
        '''between class scatter matrix'''
        labelset = set(label)
        dim = data.shape[1]
        row = data.shape[0]
        Sb = np.zeros((dim,dim))
        total_mean = np.mean(data,0)
        total_mean = np.array([total_mean])

        for i in labelset:
            pos = np.where(label == i)
            X = data[pos]
            possize = np.size(pos)
            mean = np.mean(X,0)
            mean = np.array([mean])
            S = np.dot((mean-total_mean).T,(mean-total_mean))
            Sb = Sb + possize/row*S
        return Sb
    def fit_transform(self,alpha,beta,detla,mu):
        Ms = np.mean(self.X_S,axis=0)
        Mt = np.mean(self.X_T,axis=0)
        dim = np.size(self.X_S, 1)
        Ns = np.size(self.X_S,0)
        Nt = np.size(self.X_T,0)
        M = np.zeros((dim, dim))
        Ss = self.func_source_domain_scatter(self.X_S)
        Sw = self.func_within_class_scatter_nonweight(self.X_S, self.Y_S)
        St = self.func_target_domain_scatter(self.X_T)
        Sb = self.func_between_class_scatter(self.X_S, self.Y_S)
        Ms = Ms.reshape((1, dim))  # reshape
        Mt = Mt.reshape((1, dim))  # reshape
        MDD = np.dot((Ms - Mt).T, (Ms - Mt))  # mean distribution discrepancy minimization
        M = 1/Ns*Ss + 1/Nt*alpha*St + detla*Sb - mu*(MDD + beta*Sw)
        eigenvalue, eigenvector = np.linalg.eig(M)
        ## 新增的验证部分
        M = M -min(eigenvalue)*np.identity(Ss.shape[0])
        eigenvalue, eigenvector = np.linalg.eig(M)
        ## 新增的验证部分
        eigenvalue = np.real(eigenvalue)  # Complex numbers appear during eigen decomposition, which makes it impossible to compare eigenvalues, so take the real part of the eigenvalue and the eigenvector
        Idex = np.argsort(-eigenvalue)
        eigenvector = np.real(eigenvector)# Same as above
        neweigenvector = eigenvector[:, Idex]
        return neweigenvector
    
class myLDE(subspace_transfer_learning_base):
    def __init__(self,X_S,Y_S,X_T,Y_T):
        super().__init__(X_S,Y_S,X_T,Y_T)
    def func_target_domain_scatter(self,data):
        return super(myLDE, self).target_domain_scatter(data)
    def func_source_domain_scatter(self,data):
        return super(myLDE, self).source_domain_scatter(data)
    def calculate_L(self,gamma):
        def matrix_concentrate(*args):
            for i in range(1,len(args)):
                if i == 1:
                    a = args[i-1]
                    b = args[i] 
                    lena = len(a)
                    lenb = len(b)
                    leftmaxtric = np.row_stack((a, np.zeros((lenb, lena))))  # 先将矩阵a和一个长为a大小，宽为b大小的零矩阵垂直拼接，得到左矩阵
                    rightmaxtric = np.row_stack((np.zeros((lena, lenb)), b))  # 再将一个长为b大小，宽为a大小的零矩阵和b垂直拼接，得到右矩阵
                    maxtirc = np.hstack((leftmaxtric, rightmaxtric))  # 将左右矩阵水平拼接
                else:
                    a = maxtirc
                    b = args[i]
                    lena = len(a)
                    lenb = len(b)
                    leftmaxtric = np.row_stack((a, np.zeros((lenb, lena))))  # 先将矩阵a和一个长为a大小，宽为b大小的零矩阵垂直拼接，得到左矩阵
                    rightmaxtric = np.row_stack((np.zeros((lena, lenb)), b))  # 再将一个长为b大小，宽为a大小的零矩阵和b垂直拼接，得到右矩阵
                    maxtirc = np.hstack((leftmaxtric, rightmaxtric))  # 将左右矩阵水平拼接
            return maxtirc 
        labelset = np.unique(self.Y_S)
        W_all = list()
        D_all = list()
        X_S = self.X_S
        for i in range(len(labelset)):
            pos = np.where(self.Y_S == labelset[i])
            X_c = X_S[pos,:]#取出属于第c列的元素
            n_c = len(pos[0])
            W_c = np.zeros([n_c,n_c])
            D_l = np.zeros([n_c])
            for l in range(n_c):
                for m in range(n_c):
                    point_l = X_S[l,:]
                    point_m = X_S[m,:]
                    W_c[l,m] = np.exp(-(np.linalg.norm((point_l-point_m)/gamma, keepdims=True))**2/1)
                D_l[l] = sum(W_c[l,:])
            D_matrix = np.diag(D_l)
            W_all.append(W_c)
            D_all.append(D_matrix)
            #### step1:接着拼接D和W即可
        D = matrix_concentrate(D_all[0],D_all[1],D_all[2],D_all[3])
        W = matrix_concentrate(W_all[0],W_all[1],W_all[2],W_all[3])
        L = D - W
        result = self.X_S.T@L@self.X_S
        return result
    def calculate_Q(self,gamma):
        # 一定需要注意，这里的Q矩阵是LDE中独特的定义的存在，需要单独计算
        labelset = np.unique(self.Y_S)
        B = np.zeros([len(labelset),len(labelset)])
        X_S = self.X_S
        temp =  np.empty([self.X_S.shape[1],0])
        for i in range(len(labelset)):
            pos = np.where(self.Y_S == labelset[i])[0]
            X_c = X_S[pos,:]#取出属于第c类的元素
            m_c = np.mean(X_c,axis = 0) #这里的逻辑可能需要再行调整
            temp = np.hstack((temp,m_c.reshape(-1,1)))
        F = temp #得出类中心对应的F矩阵
        E_l = np.zeros([F.shape[1]])
        for l in range(F.shape[1]):
            for m in range(F.shape[1]):
                point_l = F[:,l]
                point_m = F[:,m]
                B[l,m] = np.exp(-(np.linalg.norm((point_l-point_m)/gamma, keepdims=True))**2/1)
            E_l[l] = sum(B[l,:])
        E = np.diag(E_l)
        H = E - B
        result = F@H@(F.T)
        return result
        
            
    def fit_transform(self,Lambda,k,u,gamma):
        Ns = self.X_S.shape[0] #获取源域数据和目标域数据的样本量
        Nt = self.X_T.shape[0]
        Ms = np.mean(self.X_S,axis=0)
        Mt = np.mean(self.X_T,axis=0)
        dim = np.size(self.X_S,1)
        A = np.zeros((dim,dim))
        Ms = Ms.reshape((1, dim))  # reshape
        Mt = Mt.reshape((1, dim))  # reshape
        tem = np.linalg.pinv(np.dot((Ms-Mt).T,(Ms-Mt)))
        # Ss = self.func_source_domain_scatter(self.X_S)  # source domain scatter matrix
        # St = self.func_target_domain_scatter(self.X_T)  # target domain scatter matrix
        L = self.calculate_L(gamma)
        Q = self.calculate_Q(gamma)
        tem1 = 1/Ns*self.X_S.T@self.X_S + 1/Nt*Lambda*self.X_T.T@self.X_T-k*L+u*Q
        A = A + np.dot(tem,tem1)
        eigenvalue,eigenvector = np.linalg.eig(A)
        eigenvalue = np.real(eigenvalue) # Complex numbers appear during eigen decomposition, which makes it impossible to compare eigenvalues, so take the real part of the eigenvalue and the eigenvector
        Idex = np.argsort(-eigenvalue)
        eigenvector = np.real(eigenvector) # Same as above
        neweigenvector = eigenvector[:,Idex]
        return neweigenvector
    
    
class myLDE_constrained(subspace_transfer_learning_base):
    def __init__(self,X_S,Y_S,X_T,Y_T):
        super().__init__(X_S,Y_S,X_T,Y_T)
    def func_target_domain_scatter(self,data):
        return super(myLDE_constrained, self).target_domain_scatter(data)
    def func_source_domain_scatter(self,data):
        return super(myLDE_constrained, self).source_domain_scatter(data)
    def calculate_L(self,gamma):
        def matrix_concentrate(*args):
            for i in range(1,len(args)):
                if i == 1:
                    a = args[i-1]
                    b = args[i] 
                    lena = len(a)
                    lenb = len(b)
                    leftmaxtric = np.row_stack((a, np.zeros((lenb, lena))))  # 先将矩阵a和一个长为a大小，宽为b大小的零矩阵垂直拼接，得到左矩阵
                    rightmaxtric = np.row_stack((np.zeros((lena, lenb)), b))  # 再将一个长为b大小，宽为a大小的零矩阵和b垂直拼接，得到右矩阵
                    maxtirc = np.hstack((leftmaxtric, rightmaxtric))  # 将左右矩阵水平拼接
                else:
                    a = maxtirc
                    b = args[i]
                    lena = len(a)
                    lenb = len(b)
                    leftmaxtric = np.row_stack((a, np.zeros((lenb, lena))))  # 先将矩阵a和一个长为a大小，宽为b大小的零矩阵垂直拼接，得到左矩阵
                    rightmaxtric = np.row_stack((np.zeros((lena, lenb)), b))  # 再将一个长为b大小，宽为a大小的零矩阵和b垂直拼接，得到右矩阵
                    maxtirc = np.hstack((leftmaxtric, rightmaxtric))  # 将左右矩阵水平拼接
            return maxtirc 
        labelset = np.unique(self.Y_S)
        W_all = list()
        D_all = list()
        X_S = self.X_S
        for i in range(len(labelset)):
            pos = np.where(self.Y_S == labelset[i])
            X_c = X_S[pos,:]#取出属于第c列的元素
            n_c = len(pos[0])
            W_c = np.zeros([n_c,n_c])
            D_l = np.zeros([n_c])
            for l in range(n_c):
                for m in range(n_c):
                    point_l = X_S[l,:]
                    point_m = X_S[m,:]
                    W_c[l,m] = np.exp(-(np.linalg.norm((point_l-point_m)/gamma, keepdims=True))**2/1)
                D_l[l] = sum(W_c[l,:])
            D_matrix = np.diag(D_l)
            W_all.append(W_c)
            D_all.append(D_matrix)
            #### step1:接着拼接D和W即可
        D = matrix_concentrate(D_all[0],D_all[1],D_all[2],D_all[3])
        W = matrix_concentrate(W_all[0],W_all[1],W_all[2],W_all[3])
        L = D - W
        result = self.X_S.T@L@self.X_S
        return result
    def calculate_Q(self,gamma):
        # 一定需要注意，这里的Q矩阵是LDE中独特的定义的存在，需要单独计算
        labelset = np.unique(self.Y_S)
        B = np.zeros([len(labelset),len(labelset)])
        X_S = self.X_S
        temp =  np.empty([self.X_S.shape[1],0])
        for i in range(len(labelset)):
            pos = np.where(self.Y_S == labelset[i])[0]
            X_c = X_S[pos,:]#取出属于第c类的元素
            m_c = np.mean(X_c,axis = 0) #这里的逻辑可能需要再行调整
            temp = np.hstack((temp,m_c.reshape(-1,1)))
        F = temp #得出类中心对应的F矩阵
        E_l = np.zeros([F.shape[1]])
        for l in range(F.shape[1]):
            for m in range(F.shape[1]):
                point_l = F[:,l]
                point_m = F[:,m]
                B[l,m] = np.exp(-(np.linalg.norm((point_l-point_m)/gamma, keepdims=True))**2/1)
            E_l[l] = sum(B[l,:])
        E = np.diag(E_l)
        H = E - B
        result = F@H@(F.T)
        return result
        
            
    def fit_transform(self,Lambda,k,u,beta,gamma):
        Ns = self.X_S.shape[0] #获取源域数据和目标域数据的样本量
        Nt = self.X_T.shape[0]
        Ms = np.mean(self.X_S,axis=0)
        Mt = np.mean(self.X_T,axis=0)
        dim = np.size(self.X_S,1)
        A = np.zeros((dim,dim))
        Ms = Ms.reshape((1, dim))  # reshape
        Mt = Mt.reshape((1, dim))  # reshape
        tem = np.linalg.pinv(np.dot((Ms-Mt).T,(Ms-Mt)))
        # Ss = self.func_source_domain_scatter(self.X_S)  # source domain scatter matrix
        # St = self.func_target_domain_scatter(self.X_T)  # target domain scatter matrix
        L = self.calculate_L(gamma)
        Q = self.calculate_Q(gamma)
        tem1 = 1/Ns*self.X_S.T@self.X_S + 1/Nt*Lambda*self.X_T.T@self.X_T-k*L+u*Q-beta*np.dot((Ms-Mt).T,(Ms-Mt))
        A = A + tem1
        eigenvalue,eigenvector = np.linalg.eig(A)
        eigenvalue = np.real(eigenvalue) # Complex numbers appear during eigen decomposition, which makes it impossible to compare eigenvalues, so take the real part of the eigenvalue and the eigenvector
        Idex = np.argsort(-eigenvalue)
        eigenvector = np.real(eigenvector) # Same as above
        neweigenvector = eigenvector[:,Idex]
        return neweigenvector
    
class myLDEr(subspace_transfer_learning_base):
    def __init__(self,X_S,Y_S,X_T,Y_T):
        super().__init__(X_S,Y_S,X_T,Y_T)
    def func_target_domain_scatter(self,data):
        return super(myLDEr, self).target_domain_scatter(data)
    def func_source_domain_scatter(self,data):
        return super(myLDE, self).source_domain_scatter(data)
    def calculate_L(self,gamma):
        def matrix_concentrate(*args):
            for i in range(1,len(args)):
                if i == 1:
                    a = args[i-1]
                    b = args[i] 
                    lena = len(a)
                    lenb = len(b)
                    leftmaxtric = np.row_stack((a, np.zeros((lenb, lena))))  # 先将矩阵a和一个长为a大小，宽为b大小的零矩阵垂直拼接，得到左矩阵
                    rightmaxtric = np.row_stack((np.zeros((lena, lenb)), b))  # 再将一个长为b大小，宽为a大小的零矩阵和b垂直拼接，得到右矩阵
                    maxtirc = np.hstack((leftmaxtric, rightmaxtric))  # 将左右矩阵水平拼接
                else:
                    a = maxtirc
                    b = args[i]
                    lena = len(a)
                    lenb = len(b)
                    leftmaxtric = np.row_stack((a, np.zeros((lenb, lena))))  # 先将矩阵a和一个长为a大小，宽为b大小的零矩阵垂直拼接，得到左矩阵
                    rightmaxtric = np.row_stack((np.zeros((lena, lenb)), b))  # 再将一个长为b大小，宽为a大小的零矩阵和b垂直拼接，得到右矩阵
                    maxtirc = np.hstack((leftmaxtric, rightmaxtric))  # 将左右矩阵水平拼接
            return maxtirc 
        labelset = np.unique(self.Y_S)
        W_all = list()
        D_all = list()
        X_S = self.X_S
        for i in range(len(labelset)):
            pos = np.where(self.Y_S == labelset[i])
            X_c = X_S[pos,:]#取出属于第c列的元素
            n_c = len(pos[0])
            W_c = np.zeros([n_c,n_c])
            D_l = np.zeros([n_c])
            for l in range(n_c):
                for m in range(n_c):
                    point_l = X_S[l,:]
                    point_m = X_S[m,:]
                    W_c[l,m] = np.exp(-(np.linalg.norm((point_l-point_m)/gamma, keepdims=True))**2/1)
                D_l[l] = sum(W_c[l,:])
            D_matrix = np.diag(D_l)
            W_all.append(W_c)
            D_all.append(D_matrix)
            #### step1:接着拼接D和W即可
        D = matrix_concentrate(D_all[0],D_all[1],D_all[2],D_all[3])
        W = matrix_concentrate(W_all[0],W_all[1],W_all[2],W_all[3])
        L = D - W
        result = self.X_S.T@L@self.X_S
        return result
    def calculate_Q(self,gamma):
        # 一定需要注意，这里的Q矩阵是LDE中独特的定义的存在，需要单独计算
        labelset = np.unique(self.Y_S)
        B = np.zeros([len(labelset),len(labelset)])
        X_S = self.X_S
        temp =  np.empty([self.X_S.shape[1],0])
        for i in range(len(labelset)):
            pos = np.where(self.Y_S == labelset[i])[0]
            X_c = X_S[pos,:]#取出属于第c类的元素
            m_c = np.mean(X_c,axis = 0) #这里的逻辑可能需要再行调整
            temp = np.hstack((temp,m_c.reshape(-1,1)))
        F = temp #得出类中心对应的F矩阵
        E_l = np.zeros([F.shape[1]])
        for l in range(F.shape[1]):
            for m in range(F.shape[1]):
                point_l = F[:,l]
                point_m = F[:,m]
                B[l,m] = np.exp(-(np.linalg.norm((point_l-point_m)/gamma, keepdims=True))**2/1)
            E_l[l] = sum(B[l,:])
        E = np.diag(E_l)
        H = E - B
        result = F@H@(F.T)
        return result
        
            
    def fit_transform(self,k,u,gamma):
        Ns = self.X_S.shape[0] #获取源域数据和目标域数据的样本量
        Nt = self.X_T.shape[0]
        Ms = np.mean(self.X_S,axis=0)
        Mt = np.mean(self.X_T,axis=0)
        dim = np.size(self.X_S,1)
        A = np.zeros((dim,dim))
        Ms = Ms.reshape((1, dim))  # reshape
        Mt = Mt.reshape((1, dim))  # reshape
        tem = np.linalg.pinv(np.dot((Ms-Mt).T,(Ms-Mt)))
        # Ss = self.func_source_domain_scatter(self.X_S)  # source domain scatter matrix
        # St = self.func_target_domain_scatter(self.X_T)  # target domain scatter matrix
        L = self.calculate_L(gamma)
        Q = self.calculate_Q(gamma)
        tem1 =  -k*L+u*Q
        A = A + tem@tem1
        eigenvalue,eigenvector = np.linalg.eig(A)
        eigenvalue = np.real(eigenvalue) # Complex numbers appear during eigen decomposition, which makes it impossible to compare eigenvalues, so take the real part of the eigenvalue and the eigenvector
        Idex = np.argsort(-eigenvalue)
        eigenvector = np.real(eigenvector) # Same as above
        neweigenvector = eigenvector[:,Idex]
        return neweigenvector


class paper2(subspace_transfer_learning_base):
    def __init__(self, X_S, Y_S, X_T, Y_T):
        super().__init__(X_S, Y_S, X_T, Y_T)

    def func_target_domain_scatter(self, data):
        return super(paper2, self).target_domain_scatter(data)

    def func_source_domain_scatter(self, data):
        return super(paper2, self).source_domain_scatter(data)

    def calculate_L(self, gamma):
        def matrix_concentrate(*args):
            for i in range(1, len(args)):
                if i == 1:
                    a = args[i - 1]
                    b = args[i]
                    lena = len(a)
                    lenb = len(b)
                    leftmaxtric = np.row_stack((a, np.zeros((lenb, lena))))  # 先将矩阵a和一个长为a大小，宽为b大小的零矩阵垂直拼接，得到左矩阵
                    rightmaxtric = np.row_stack((np.zeros((lena, lenb)), b))  # 再将一个长为b大小，宽为a大小的零矩阵和b垂直拼接，得到右矩阵
                    maxtirc = np.hstack((leftmaxtric, rightmaxtric))  # 将左右矩阵水平拼接
                else:
                    a = maxtirc
                    b = args[i]
                    lena = len(a)
                    lenb = len(b)
                    leftmaxtric = np.row_stack((a, np.zeros((lenb, lena))))  # 先将矩阵a和一个长为a大小，宽为b大小的零矩阵垂直拼接，得到左矩阵
                    rightmaxtric = np.row_stack((np.zeros((lena, lenb)), b))  # 再将一个长为b大小，宽为a大小的零矩阵和b垂直拼接，得到右矩阵
                    maxtirc = np.hstack((leftmaxtric, rightmaxtric))  # 将左右矩阵水平拼接
            return maxtirc

        labelset = np.unique(self.Y_S)
        W_all = list()
        D_all = list()
        X_S = self.X_S
        for i in range(len(labelset)):
            pos = np.where(self.Y_S == labelset[i])
            X_c = X_S[pos, :]  # 取出属于第c列的元素
            n_c = len(pos[0])
            W_c = np.zeros([n_c, n_c])
            D_l = np.zeros([n_c])
            for l in range(n_c):
                for m in range(n_c):
                    point_l = X_S[l, :]
                    point_m = X_S[m, :]
                    W_c[l, m] = np.exp(-(np.linalg.norm((point_l - point_m) / gamma, keepdims=True)) ** 2 / 1)
                D_l[l] = sum(W_c[l, :])
            D_matrix = np.diag(D_l)
            W_all.append(W_c)
            D_all.append(D_matrix)
            #### step1:接着拼接D和W即可
        D = matrix_concentrate(D_all[0], D_all[1], D_all[2], D_all[3])
        W = matrix_concentrate(W_all[0], W_all[1], W_all[2], W_all[3])

        L = D - W
        result = self.X_S.T @ L @ self.X_S
        return result

    def calculate_Q(self, gamma):
        # 一定需要注意，这里的Q矩阵是LDE中独特的定义的存在，需要单独计算
        labelset = np.unique(self.Y_S)
        B = np.zeros([len(labelset), len(labelset)])
        X_S = self.X_S
        temp = np.empty([self.X_S.shape[1], 0])
        for i in range(len(labelset)):
            pos = np.where(self.Y_S == labelset[i])[0]
            X_c = X_S[pos, :]  # 取出属于第c类的元素
            m_c = np.mean(X_c, axis=0)  # 这里的逻辑可能需要再行调整
            temp = np.hstack((temp, m_c.reshape(-1, 1)))
        F = temp  # 得出类中心对应的F矩阵
        E_l = np.zeros([F.shape[1]])
        for l in range(F.shape[1]):
            for m in range(F.shape[1]):
                point_l = F[:, l]
                point_m = F[:, m]
                B[l, m] = np.exp(-(np.linalg.norm((point_l - point_m) / gamma, keepdims=True)) ** 2 / 1)
            E_l[l] = sum(B[l, :])
        E = np.diag(E_l)
        H = E - B
        result = F @ H @ (F.T)
        return result

    def fit_transform(self, u1, u2, u3,u4,alpha,gamma):
        Ns = self.X_S.shape[0]  # 获取源域数据和目标域数据的样本量
        Nt = self.X_T.shape[0]
        Ms = np.mean(self.X_S, axis=0)
        Mt = np.mean(self.X_T, axis=0)
        dim = np.size(self.X_S, 1)
        A = np.zeros((dim, dim))
        Ms = Ms.reshape((1, dim))  # reshape
        Mt = Mt.reshape((1, dim))  # reshape
        L = self.calculate_L(gamma)
        Q = self.calculate_Q(gamma)
        w1 = 0.7
        w2 = 0.3
        tem = np.linalg.pinv(u2*L+w2*u3*np.dot((Ms - Mt).T, (Ms - Mt)))
        Ss = self.func_source_domain_scatter(self.X_S)  # source domain scatter matrix
        St = self.func_target_domain_scatter(self.X_T)  # target domain scatter matrix

        tem1 = u1*(Ss + alpha*St) + w1*u4*Q
        A = A + tem @ tem1
        eigenvalue, eigenvector = np.linalg.eig(A)
        eigenvalue = np.real(
            eigenvalue)  # Complex numbers appear during eigen decomposition, which makes it impossible to compare eigenvalues, so take the real part of the eigenvalue and the eigenvector
        Idex = np.argsort(-eigenvalue)
        eigenvector = np.real(eigenvector)  # Same as above
        neweigenvector = eigenvector[:, Idex]
        return neweigenvector

class myLDE_weighted(subspace_transfer_learning_base):
    def __init__(self,X_S,Y_S,X_T,Y_T):
        super().__init__(X_S,Y_S,X_T,Y_T)
    def func_target_domain_scatter(self,data):
        return super(myLDE_weighted, self).target_domain_scatter(data)
    def func_source_domain_scatter(self,data):
        return super(myLDE_weighted, self).source_domain_scatter(data)
    def calculate_L(self,gamma):
        def matrix_concentrate(*args):
            for i in range(1,len(args)):
                if i == 1:
                    a = args[i-1]
                    b = args[i] 
                    lena = len(a)
                    lenb = len(b)
                    leftmaxtric = np.row_stack((a, np.zeros((lenb, lena))))  # 先将矩阵a和一个长为a大小，宽为b大小的零矩阵垂直拼接，得到左矩阵
                    rightmaxtric = np.row_stack((np.zeros((lena, lenb)), b))  # 再将一个长为b大小，宽为a大小的零矩阵和b垂直拼接，得到右矩阵
                    maxtirc = np.hstack((leftmaxtric, rightmaxtric))  # 将左右矩阵水平拼接
                else:
                    a = maxtirc
                    b = args[i]
                    lena = len(a)
                    lenb = len(b)
                    leftmaxtric = np.row_stack((a, np.zeros((lenb, lena))))  # 先将矩阵a和一个长为a大小，宽为b大小的零矩阵垂直拼接，得到左矩阵
                    rightmaxtric = np.row_stack((np.zeros((lena, lenb)), b))  # 再将一个长为b大小，宽为a大小的零矩阵和b垂直拼接，得到右矩阵
                    maxtirc = np.hstack((leftmaxtric, rightmaxtric))  # 将左右矩阵水平拼接
            return maxtirc 
        labelset = np.unique(self.Y_S)
        W_all = list()
        D_all = list()
        X_S = self.X_S
        for i in range(len(labelset)):
            pos = np.where(self.Y_S == labelset[i])
            X_c = X_S[pos,:]#取出属于第c列的元素
            n_c = len(pos[0])
            W_c = np.zeros([n_c,n_c])
            D_l = np.zeros([n_c])
            for l in range(n_c):
                for m in range(n_c):
                    point_l = X_S[l,:]
                    point_m = X_S[m,:]
                    W_c[l,m] = np.exp(-(np.linalg.norm((point_l-point_m)/gamma, keepdims=True))**2/1)
                D_l[l] = sum(W_c[l,:])
            D_matrix = np.diag(D_l)
            W_all.append(W_c)
            D_all.append(D_matrix)
            #### step1:接着拼接D和W即可
        D = matrix_concentrate(D_all[0],D_all[1],D_all[2],D_all[3])
        W = matrix_concentrate(W_all[0],W_all[1],W_all[2],W_all[3])
        L = D - W
        result = self.X_S.T@L@self.X_S
        return result
    def calculate_Q(self,gamma):
        # 一定需要注意，这里的Q矩阵是LDE中独特的定义的存在，需要单独计算
        labelset = np.unique(self.Y_S)
        B = np.zeros([len(labelset),len(labelset)])
        X_S = self.X_S
        temp =  np.empty([self.X_S.shape[1],0])
        for i in range(len(labelset)):
            pos = np.where(self.Y_S == labelset[i])[0]
            X_c = X_S[pos,:]#取出属于第c类的元素
            m_c = np.mean(X_c,axis = 0) #这里的逻辑可能需要再行调整
            temp = np.hstack((temp,m_c.reshape(-1,1)))
        F = temp #得出类中心对应的F矩阵
        E_l = np.zeros([F.shape[1]])
        for l in range(F.shape[1]):
            for m in range(F.shape[1]):
                point_l = F[:,l]
                point_m = F[:,m]
                B[l,m] = np.exp(-(np.linalg.norm((point_l-point_m)/gamma, keepdims=True))**2/1)
            E_l[l] = sum(B[l,:])
        E = np.diag(E_l)
        H = E - B
        result = F@H@(F.T)
        return result
        
            
    def fit_transform(self,Lambda,k,u,beta,gamma,w1):
        Ns = self.X_S.shape[0] #获取源域数据和目标域数据的样本量
        Nt = self.X_T.shape[0]
        Ms = np.mean(self.X_S,axis=0)
        Mt = np.mean(self.X_T,axis=0)
        dim = np.size(self.X_S,1)
        A = np.zeros((dim,dim))
        Ms = Ms.reshape((1, dim))  # reshape
        Mt = Mt.reshape((1, dim))  # reshape
        # Ss = self.func_source_domain_scatter(self.X_S)  # source domain scatter matrix
        # St = self.func_target_domain_scatter(self.X_T)  # target domain scatter matrix
        L = self.calculate_L(gamma)
        Q = self.calculate_Q(gamma)
        tem1 = 1/Ns*self.X_S.T@self.X_S + 1/Nt*Lambda*self.X_T.T@self.X_T-k*L+u*Q-beta*np.dot((Ms-Mt).T,(Ms-Mt))
        A = A + tem1
        eigenvalue,eigenvector = np.linalg.eig(A)
        eigenvalue = np.real(eigenvalue) # Complex numbers appear during eigen decomposition, which makes it impossible to compare eigenvalues, so take the real part of the eigenvalue and the eigenvector
        Idex = np.argsort(-eigenvalue)
        eigenvector = np.real(eigenvector) # Same as above
        neweigenvector = eigenvector[:,Idex]
        return neweigenvector

class paper3(subspace_transfer_learning_base):
    def __init__(self,X_S,Y_S,X_T,Y_T,gamma):
        super().__init__(X_S,Y_S,X_T,Y_T)
        # =============================================================================
        #         计算W矩阵
        # =============================================================================
        Ns = self.X_S.shape[0]
        Nt = self.X_T.shape[0]
        W =np.zeros([Ns,Nt]) 
        for l in range(W.shape[0]):
            for m in range(W.shape[1]):
                point_l = self.X_S[l,:]
                point_m = self.X_T[m,:]
                W[l,m] = np.exp(-(np.linalg.norm((point_l-point_m)/1, keepdims=True))**2/gamma) #这里的gamma可以考虑进一步调整
        self.W = W
    def func_target_domain_scatter(self,data):
        return super(paper3, self).target_domain_scatter(data)
    def func_source_domain_scatter(self,data):
        return super(paper3, self).source_domain_scatter(data)
    def calculate_L(self,gamma):
        def matrix_concentrate(*args):
            for i in range(1,len(args)):
                if i == 1:
                    a = args[i-1]
                    b = args[i] 
                    lena = len(a)
                    lenb = len(b)
                    leftmaxtric = np.row_stack((a, np.zeros((lenb, lena))))  # 先将矩阵a和一个长为a大小，宽为b大小的零矩阵垂直拼接，得到左矩阵
                    rightmaxtric = np.row_stack((np.zeros((lena, lenb)), b))  # 再将一个长为b大小，宽为a大小的零矩阵和b垂直拼接，得到右矩阵
                    maxtirc = np.hstack((leftmaxtric, rightmaxtric))  # 将左右矩阵水平拼接
                else:
                    a = maxtirc
                    b = args[i]
                    lena = len(a)
                    lenb = len(b)
                    leftmaxtric = np.row_stack((a, np.zeros((lenb, lena))))  # 先将矩阵a和一个长为a大小，宽为b大小的零矩阵垂直拼接，得到左矩阵
                    rightmaxtric = np.row_stack((np.zeros((lena, lenb)), b))  # 再将一个长为b大小，宽为a大小的零矩阵和b垂直拼接，得到右矩阵
                    maxtirc = np.hstack((leftmaxtric, rightmaxtric))  # 将左右矩阵水平拼接
            return maxtirc 
        labelset = np.unique(self.Y_S)
        W_all = list()
        D_all = list()
        X_S = self.X_S
        for i in range(len(labelset)):
            pos = np.where(self.Y_S == labelset[i])
            X_c = X_S[pos,:]#取出属于第c列的元素
            n_c = len(pos[0])
            W_c = np.zeros([n_c,n_c])
            D_l = np.zeros([n_c])
            for l in range(n_c):
                for m in range(n_c):
                    point_l = X_S[l,:]
                    point_m = X_S[m,:]
                    W_c[l,m] = np.exp(-(np.linalg.norm((point_l-point_m)/gamma, keepdims=True))**2/1)
                D_l[l] = sum(W_c[l,:])
            D_matrix = np.diag(D_l)
            W_all.append(W_c)
            D_all.append(D_matrix)
            #### step1:接着拼接D和W即可
        D = matrix_concentrate(D_all[0],D_all[1],D_all[2],D_all[3])
        W = matrix_concentrate(W_all[0],W_all[1],W_all[2],W_all[3])
        L = D - W
        result = self.X_S.T@L@self.X_S
        return result
    def calculate_Q(self,gamma):
        # 一定需要注意，这里的Q矩阵是LDE中独特的定义的存在，需要单独计算
        labelset = np.unique(self.Y_S)
        B = np.zeros([len(labelset),len(labelset)])
        X_S = self.X_S
        temp =  np.empty([self.X_S.shape[1],0])
        for i in range(len(labelset)):
            pos = np.where(self.Y_S == labelset[i])[0]
            X_c = X_S[pos,:]#取出属于第c类的元素
            m_c = np.mean(X_c,axis = 0) #这里的逻辑可能需要再行调整
            temp = np.hstack((temp,m_c.reshape(-1,1)))
        F = temp #得出类中心对应的F矩阵
        E_l = np.zeros([F.shape[1]])
        for l in range(F.shape[1]):
            for m in range(F.shape[1]):
                point_l = F[:,l]
                point_m = F[:,m]
                B[l,m] = np.exp(-(np.linalg.norm((point_l-point_m)/gamma, keepdims=True))**2/1)
            E_l[l] = sum(B[l,:])
        E = np.diag(E_l)
        H = E - B
        result = F@H@(F.T)
        return result
        
            
    def fit_transform(self,Lambda,k,u,gamma):
        Ns = self.X_S.shape[0] #获取源域数据和目标域数据的样本量
        Nt = self.X_T.shape[0]
        Ms = np.mean(self.X_S,axis=0)
        Mt = np.mean(self.X_T,axis=0)
        dim = np.size(self.X_S,1)
        A = np.zeros((dim,dim))
        Ms = Ms.reshape((1, dim))  # reshape
        Mt = Mt.reshape((1, dim))  # reshape
        tem = np.linalg.pinv(np.dot((Ms-Mt).T,(Ms-Mt)))
        # Ss = self.func_source_domain_scatter(self.X_S)  # source domain scatter matrix
        # St = self.func_target_domain_scatter(self.X_T)  # target domain scatter matrix
        L = self.calculate_L(gamma)

        tem1 = 1/Ns*self.X_S.T@self.X_S + 1/Nt*Lambda*self.X_T.T@self.X_T-k*L+u*self.X_S.T@self.W@self.X_T
        A = A + np.dot(tem,tem1)
        eigenvalue,eigenvector = np.linalg.eig(A)
        eigenvalue = np.real(eigenvalue) # Complex numbers appear during eigen decomposition, which makes it impossible to compare eigenvalues, so take the real part of the eigenvalue and the eigenvector
        Idex = np.argsort(-eigenvalue)
        eigenvector = np.real(eigenvector) # Same as above
        neweigenvector = eigenvector[:,Idex]
        return neweigenvector