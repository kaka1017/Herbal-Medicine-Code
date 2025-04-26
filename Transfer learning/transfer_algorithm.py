
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

    

    def fit_new(self, Xs, Xt, Xt2):
        '''
        Map Xt2 to the latent space created from Xt and Xs
        :param Xs : ns * n_feature, source feature
        :param Xt : nt * n_feature, target feature
        :param Xt2: n_s, n_feature, target feature to be mapped
        :return: Xt2_new, mapped Xt2 with projection created by Xs and Xt
        '''
        
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

        
        Xt2 = Xt2.T
        K = kernel(self.kernel_type, X1=Xt2, X2=X, gamma=self.gamma)

        
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
    
    def __init__(self,X_S,Y_S,X_T,Y_T):
        self.X_S = X_S 
        self.Y_S = Y_S 
        self.X_T = X_T 
        self.Y_T = Y_T 
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
        return hsic/(row*row) 
    def NCCO(self,data,label):
        row = data.shape[0]
        H = np.eye(row) - (1 / row) * np.ones((row, row))
        label_en = LabelBinarizer(neg_label=0,pos_label=1)
        Y = label_en.fit_transform(label)
        Y = Y/sum(Y)
        X1 = H@data
        ncco_matrix = np.linalg.pinv(data)@Y@np.linalg.pinv(Y)@data
        return ncco_matrix 
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
            
            Sw = Sw + (1-possize/row)*S
            
            
            
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
    
    def LLE_W(self,data, n_dims = 2, n_neighbors = 10):
        def cal_pairwise_dist(x):
        
        
        
        
            N,D = np.shape(x)
            
            dist = np.zeros([N,N])
            
            for i in range(N):
                for j in range(N):
                    dist[i,j] = np.sqrt(np.dot((x[i]-x[j]),(x[i]-x[j]).T))

            
            return dist


        
        def get_n_neighbors(data, n_neighbors = n_neighbors):
        
        
        
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
        
        Index_NN = get_n_neighbors(data,n_neighbors)
        
        
        w = np.zeros([N,n_neighbors])
        
        for i in range(N):
            
            X_k = data[Index_NN[i]]  
            X_i = [data[i]]       
            I = np.ones([n_neighbors,1])
            
            Si = np.dot((np.dot(I,X_i)-X_k), (np.dot(I,X_i)-X_k).T)
            
            
            Si = Si+np.eye(n_neighbors)*tol*np.trace(Si)
            
            Si_inv = np.linalg.pinv(Si)
            w[i] = np.dot(I.T,Si_inv)/(np.dot(np.dot(I.T,Si_inv),I))
         
        
        W = np.zeros([N,N])
        for i in range(N):
            W[i,Index_NN[i]] = w[i]
        return W
    def Laplace_Matrix(self,data,n_neighbors):
        def cal_pairwise_dist(x):
            N,D = np.shape(x)
            
            dist = np.zeros([N,N])
            
            for i in range(N):
                for j in range(N):
                    dist[i,j] = np.sqrt(np.dot((x[i]-x[j]),(x[i]-x[j]).T))

            
            return dist

        def get_n_neighbors(data, n_neighbors):
        
        
        
            dist = cal_pairwise_dist(data)
            dist[dist < 0] = 0
            N = dist.shape[0] 
            Index = np.argsort(dist,axis=1)[:,1:n_neighbors+1]
            return Index
        N,D = np.shape(data)
        Index_NN = get_n_neighbors(data,n_neighbors)
        W = np.zeros([N,N])

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
        Ms = Ms.reshape((1, dim))  
        Mt = Mt.reshape((1, dim))  
        tem = np.linalg.pinv(np.dot((Ms-Mt).T,(Ms-Mt)))
        Ss = self.func_source_domain_scatter(self.X_S)  
        St = self.func_target_domain_scatter(self.X_T)  
        tem1 = Ss + Lambda*St
        A = A + np.dot(tem,tem1)
        eigenvalue,eigenvector = np.linalg.eig(A)
        eigenvalue = np.real(eigenvalue) 
        Idex = np.argsort(-eigenvalue)
        eigenvector = np.real(eigenvector) 
        neweigenvector = eigenvector[:,Idex]
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
        dim = np.size(self.X_S,1)                
        M = np.zeros((dim,dim))
        Ss = self.func_source_domain_scatter(self.X_S)     
        St = self.func_target_domain_scatter(self.X_T)     
        Ms = Ms.reshape((1,dim))
        Mt = Mt.reshape((1,dim))
        MDD = np.dot((Ms-Mt).T,(Ms-Mt))        
        hsic = self.func_hisc(self.X_S,self.Y_S)                 
        M = M + Ss + alpha*St - lambdas*MDD + detla*hsic 
        eigenvalue, eigenvector = np.linalg.eig(M)
        index = np.argsort(-eigenvalue)        
        neweigenvector = eigenvector[:,index]  
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
        dim = np.size(batchS,1)                
        M = np.zeros((dim,dim))
        Ss = self.func_source_domain_scatter(batchS)     
        Sw = self.func_within_class_scatter_nonweight(batchS,Ys)   
        St = self.func_target_domain_scatter(batchT)     
        Ms = Ms.reshape((1,dim))               
        Mt = Mt.reshape((1,dim))               
        MDD = np.dot((Ms-Mt).T,(Ms-Mt))        
        hsic = self.func_hisc(batchS,Ys)                 
        M = M + Ss + alpha*St - lambdas*MDD + detla*hsic - beta*Sw 
        eigenvalue, eigenvector = np.linalg.eig(M)
        eigenvalue = np.real(eigenvalue)  
        index = np.argsort(-eigenvalue)        
        eigenvector = np.real(eigenvector)
        neweigenvector = eigenvector[:,index]  
        return neweigenvector

class CSDL(subspace_transfer_learning_base):
    def __init__(self,X_S,Y_S,X_T,Y_T):
        super().__init__(X_S,Y_S,X_T,Y_T)
    def func_target_domain_scatter(self,data):
        return data.T@data
    def func_source_domain_scatter(self,data):
        return data.T@data
    def func_within_class_scatter_nonweight(self,data,label):
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
        W = self.func_LLE_W(self.X_S,n_neighbors=neighbors)
        Ms = np.mean(self.X_S,axis=0)
        Mt = np.mean(self.X_T,axis=0)
        dim = np.size(self.X_S, 1)
        Ns = np.size(self.X_S,0)
        Nt = np.size(self.X_T,0)
        M = np.zeros((dim, dim))
        Sw = self.func_within_class_scatter_nonweight(self.X_S, self.Y_S)
        Sb = self.func_between_class_scatter(self.X_S, self.Y_S)
        Ms = Ms.reshape((1, dim))  
        Mt = Mt.reshape((1, dim))  
        MDD = np.dot((Ms - Mt).T, (Ms - Mt))  
        LLE_1 = np.dot(np.identity(Ns)-W,(np.identity(Ns)-W).T)
        LLE_2 = np.dot(self.X_S.T,LLE_1)
        LLE_3 = np.dot(LLE_2,self.X_S)
        LLE_4 = Sw+ LLE_3 + MDD 
        tem = np.linalg.pinv(LLE_4) 
        tem1 = Sb + np.dot(self.X_S.T,self.X_S) + lamda * np.dot(self.X_T.T,self.X_T) 
        M = M + np.dot(tem,tem1)
        eigenvalue, eigenvector = np.linalg.eig(M)
        eigenvalue = np.real(eigenvalue)  
        Idex = np.argsort(-eigenvalue)
        eigenvector = np.real(eigenvector)
        neweigenvector = eigenvector[:, Idex]
        return neweigenvector

class UDA_CA(subspace_transfer_learning_base):
    def __init__(self,X_S,Y_S,X_T,Y_T):
        super().__init__(X_S,Y_S,X_T,Y_T)
    def func_Laplace_Matrix(self,data, n_neighbors):
        return super(UDA_CA,self).Laplace_Matrix(data, n_neighbors)
    def fit_transform(self,n_neighbors,alpha,lamba,beta,max_inter):
        
        XS = self.X_S.T 
        XT = self.X_T.T 
        NS = np.size(XS, 1)
        NT = np.size(XT,1)
        W = np.zeros([NS,NT]) 
        LS = self.func_Laplace_Matrix(XS.T, n_neighbors) 
        LT = self.func_Laplace_Matrix(XT.T, n_neighbors) 
        for i in range(max_inter):
            tem1a = np.dot(XS,W) 
            tem1 = np.dot(XT,XT.T)+np.dot(tem1a,tem1a.T)
            tem2 = np.dot(np.dot(XT,LT),XT.T) + np.dot(np.dot(XS,LS),XS.T)
            tem3 = np.dot(np.dot(XS,W),XT.T) + np.dot(np.dot(XT,W.T),XS.T)
            formula1 = alpha*tem1 + lamba*tem2 - (1/2*alpha+beta)*tem3
            eigenvalue, eigenvector = np.linalg.eig(formula1)
            eigenvalue = np.real(eigenvalue)  
            Idex = np.argsort(-eigenvalue)
            eigenvector = np.real(eigenvector)
            neweigenvector = eigenvector[:, Idex]
            
            B = np.linalg.norm(W,𝑎𝑥𝑖𝑠=1,keepdims=True)
            tem4a = np.dot(XS.T,neweigenvector)
            tem4 = np.linalg.pinv(B + 2*alpha* np.dot(tem4a,tem4a.T))
            tem5 = np.dot(neweigenvector.T,XT)
            W = (2*alpha+beta)*np.dot(np.dot(tem4,tem4a),tem5)
        
        return neweigenvector
    
class DAST(subspace_transfer_learning_base):
    def __init__(self,X_S,Y_S,X_T,Y_T):
        super().__init__(X_S,Y_S,X_T,Y_T)
    def func_Laplace_Matrix(self,data, n_neighbors):
        return super(DAST,self).Laplace_Matrix(data, n_neighbors)
    def fit_transform(self,neta,mu,n_neighbors,max_inter,alpha,beta,gamma):
        def soft_threshold_function(matrix,thread):
            for i in range(len(matrix)):
                for j in range(len(matrix[0])):
                    if abs(matrix[i][j])<thread:
                        matrix[i][j] = 0
                    elif matrix[i][j]> thread:
                        matrix[i][j] = matrix[i][j] - thread
                    else:
                        matrix[i][j] = matrix[i][j] + thread
            return matrix



        
        XS = self.X_S.T 
        XT = self.X_T.T 
        X = np.hstack([XS,XT])
        NS = np.size(XS, 1)
        NT = np.size(XT,1)
        Z = np.ones([NS,NT])
        R = np.zeros([NS,NT])
        Y = np.zeros([NS,NT])
        W = np.zeros([NS,NT]) 
        D = np.size(XS, 0)
        d = D 
        rho = 2
        mu_max = 1000
        P = np.ones([D,d])
        LS = self.func_Laplace_Matrix(XS.T, n_neighbors) 
        LT = self.func_Laplace_Matrix(XT.T, n_neighbors) 
        for i in range(max_inter):
            
            error = 0.1
            while(np.linalg.norm(Z-R,ord=np.Inf)>error):
                Z = 2*neta*np.linalg.pinv( ((P.T@XS).T)@P.T@XS + mu*np.identity(NS) )@(mu*R-Y +2*neta*(P.T@XS).T@P.T@XT)
                R = soft_threshold_function(Z+Y/mu,0.01)
                Y = Y+ mu*(Z-R)
                mu = min(rho*mu,mu_max)
    
    
    
            J = (XT - XS@Z)@( (XT - XS@Z).T )
            tem = neta*J - alpha*(X@X.T) + beta* XT@LT@XT.T +gamma * XS@LS@XS.T
            eigenvalue, eigenvector = np.linalg.eig(tem)
            Idex = np.argsort(-eigenvalue)
            eigenvector = np.real(eigenvector)
            P = eigenvector[:, Idex[0:d]] 
        
        
        return P

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
        
        for i in range(Ns):
            for j in range(Ns):
                point_i = X_S[i, :]
                point_j = X_S[j, :]
                A[i][j] = np.exp(-(np.linalg.norm((point_i - point_j), keepdims=True)) ** 2 / 1)
        

        
        for i in range(Ns):
            for j in range(Ns):
                if self.Y_S[i] == self.Y_S[j]:
                    pos = np.where(self.Y_S == self.Y_S[j])
                    X = self.X_S[pos]
                    nl = np.size(pos)
                    W_lbc[i][j] = A[i][j] * (1 / Ns - 1 / nl)  
                else:
                    W_lbc[i][j] = 1 / Ns

                    
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
        
        for i in range(Ns):
            for j in range(Ns):
                point_i = X_S[i, :]
                point_j = X_S[j, :]
                A[i][j] = np.exp(-(np.linalg.norm((point_i - point_j), keepdims=True)) ** 2 / 1)
        

        
        for i in range(Ns):
            for j in range(Ns):
                if self.Y_S[i] == self.Y_S[j]:
                    pos = np.where(self.Y_S == self.Y_S[j])
                    X = self.X_S[pos]
                    nl = np.size(pos)
                    W_lwc[i][j] = A[i][j] / nl  
                else:
                    W_lwc[i][j] = 0

        
        slwc = np.zeros([D, D])
        for i in range(Ns):
            for j in range(Ns):
                slwc = slwc + 1 / 2 * W_lwc[i][j] * np.dot((self.X_S[i, :] - self.X_S[j, :]).T.reshape(-1, 1),
                                                           (self.X_S[i, :] - self.X_S[j, :]).reshape(1, -1))
        return slwc

    def fit_transform(self, Lambda, k, u, beta):
        Ns = self.X_S.shape[0]  
        Nt = self.X_T.shape[0]
        Ms = np.mean(self.X_S, axis=0)
        Mt = np.mean(self.X_T, axis=0)
        dim = np.size(self.X_S, 1)
        A = np.zeros((dim, dim))
        Ms = Ms.reshape((1, dim))  
        Mt = Mt.reshape((1, dim))  
        tem = np.linalg.pinv(np.dot((Ms - Mt).T, (Ms - Mt)))
        
        
        Slwc = self.Slwc
        Slbc = self.Slbc
        tem1 = 1 / Ns * self.X_S.T @ self.X_S + 1 / Nt * Lambda * self.X_T.T @ self.X_T - k * Slwc + u * Slbc - beta * tem
        A = A + tem1
        eigenvalue, eigenvector = np.linalg.eig(A)
        eigenvalue = np.real(eigenvalue)  
        Idex = np.argsort(-eigenvalue)
        eigenvector = np.real(eigenvector)  
        neweigenvector = eigenvector[:, Idex]
        return neweigenvector



class my_alg(subspace_transfer_learning_base):
    def __init__(self,X_S,Y_S,X_T,Y_T):
        super().__init__(X_S,Y_S,X_T,Y_T)
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
        Ms = Ms.reshape((1, dim))  
        Mt = Mt.reshape((1, dim))  
        MDD = np.dot((Ms - Mt).T, (Ms - Mt))  
        M = 1/Ns*Ss + 1/Nt*alpha*St + detla*Sb - mu*(MDD + beta*Sw)
        eigenvalue, eigenvector = np.linalg.eig(M)
        
        M = M -min(eigenvalue)*np.identity(Ss.shape[0])
        eigenvalue, eigenvector = np.linalg.eig(M)
        
        eigenvalue = np.real(eigenvalue)  
        Idex = np.argsort(-eigenvalue)
        eigenvector = np.real(eigenvector)
        neweigenvector = eigenvector[:, Idex]
        return neweigenvector
    
