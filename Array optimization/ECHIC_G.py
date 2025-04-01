# -*- coding: utf-8 -*-
"""
Created on Sun Jan  8 10:15:37 2023
这个代码的主要目的：

@author: 徐棚
"""
indices_all = list()
import time
import sys# 加了这了
sys.path.append("../../") #添加上两级目录
import pandas as pd
array_optimization_name = 'ECHICG'
subspace_transfer_learning_algorithm = 'DDRCA'
import numpy as np
from sklearn.neighbors import KNeighborsClassifier
from sklearn.preprocessing import StandardScaler,MinMaxScaler,Normalizer
from sklearn import metrics

start_time = time.time()  # 开始时间
def data_loader(scaler_name ='zscore'):
    # X_S_df = pd.read_csv("batch1_medianvalue.csv",header=None)
    # X_T_df = pd.read_csv("batch2_medianvalue2.csv",header=None)
    # print(X_S_df)  # 9月份，10月份作为第一批未漂移的数据
    # print(X_T_df)  # 12月作为已经漂移的数据
    # class_list = [15,34,106,274] #[15,34,106,274] [15,34,36,274]
    X_S_df = pd.read_csv("batch1_medianvalue.csv", header=None)
    X_T_df = pd.read_csv("batch2_medianvalue2.csv", header=None)
    print(X_S_df)  # 9月份，10月份作为第一批未漂移的数据
    print(X_T_df)  # 12月作为已经漂移的数据
    class_list = [15, 34, 106, 274]  # [15,34,106,274] [15,34,36,274]
    X_S_df_seleted = pd.DataFrame()
    X_T_df_selected = pd.DataFrame()
    for c in class_list:
        temp_df = X_S_df[X_S_df.iloc[:,-2] == c]
        X_S_df_seleted = pd.concat([X_S_df_seleted,temp_df],axis=0)
    for c in class_list:
        temp_df = X_T_df[X_T_df.iloc[:,-2] == c]
        X_T_df_selected = pd.concat([X_T_df_selected,temp_df],axis=0)

    X_S_df = X_S_df_seleted
    X_T_df = X_T_df_selected
    if scaler_name == 'zscore':
        scaler = StandardScaler()
    elif scaler_name =='l2':
        scaler = Normalizer("l2")
    elif scaler_name =='minmax':
        scaler = MinMaxScaler()
    X_S = np.array(X_S_df.iloc[:, 0:-3])  # 所有源域数据
    Y_S = np.array(X_S_df.iloc[:, -2])  # 所有源域标签
    X_T = np.array(X_T_df.iloc[:, 0:-3])  # 所有目标域数据
    Y_T = np.array(X_T_df.iloc[:, -2])  # s所有目标域标签
    sensor_all = ['tempture1','tempture2','humidity1','humidity2','pressure1','pressure2','TGS813','TGS2610D','MS1100','TGS826','TGS2602','TGS822','MG812','4S','4HS+',
                  'C6H6','SMD1007','WSP2110','MP4','MP135A','TGS2620','TGS2611E','AQ201','TGS8669','2M012','TGS2600','MR516','H2S','NH3','4CH3SH','SMD1013','SMD1001','MQ135','MP503','MQ3B','MQ137','4ETO-10','4OXV',
                  'MQ138','MP901','ME3-C2H6S','ME3-CH2O','PID-AH','VCC_+5V1','VCC5V-H1']
    dict_array = {}
    fea_enviorment = 2
    fea_other_sensor = 1
    start_index = 0
    for i in range(len(sensor_all)):
        if i<=5:
            dict_array[sensor_all[i]] = list(range(start_index,start_index+fea_enviorment)) #
            start_index = start_index + fea_enviorment
        elif i >=43:
            dict_array[sensor_all[i]] = list(range(start_index,start_index+fea_enviorment)) #环境传感器
            start_index = start_index + fea_enviorment
        else:
            dict_array[sensor_all[i]] = list(range(start_index,start_index+fea_other_sensor)) #其他传感器
            start_index = start_index + fea_other_sensor
    meti_oxide = ['TGS813','TGS2610D','TGS826','TGS2602','TGS822','TGS2620','TGS2611E','TGS8669','TGS2600','MS1100','MP4','MP135A','MQ135','MP503','MQ3B','MQ137','MQ138','MP901','2M012','AQ201','WSP2110']
    # meti_oxide = sensor_all
    X_S[:,dict_array['WSP2110']] = np.zeros([114,1])
    X_T[:,dict_array['WSP2110']] = np.zeros([48,1])
    character = list()
    for string in meti_oxide:
        for index in dict_array[string]:
            character.append(index)
    character.sort()
    X_S =X_S[:,character]
    X_T = X_T[:,character]
    # X_S = scaler.fit_transform(X_S)
    # X_T = scaler.transform(X_T)
    return X_S,Y_S,X_T,Y_T

#核函数
def kernel(ker, X1, X2, gamma):
    K = None
    if not ker or ker == 'primal':
        K = X1
    elif ker == 'linear':
        if X2 is not None:
            K = metrics.pairwise.linear_kernel(np.asarray(X1).T, np.asarray(X2).T)
        else:
            K = metrics.pairwise.linear_kernel(np.asarray(X1).T)
    elif ker == 'rbf':
        if X2 is not None:
            K = metrics.pairwise.rbf_kernel(np.asarray(X1).T, np.asarray(X2).T, gamma)
        else:
            K = metrics.pairwise.rbf_kernel(np.asarray(X1).T, None, gamma)
    return K



#希尔伯特—施密特独立准则
def hsic(Kx, Ky):
    Kxy = np.dot(Kx, Ky)
    n = Kxy.shape[0]
    h = np.trace(Kxy) / n**2 + np.mean(Kx) * np.mean(Ky) - 2 * np.mean(Kxy) / n
    return h * n**2 / (n - 1)**2

def my_hsic(data_X,data_Y,kernel_type = 'linear',gamma = 1):
    Kx = kernel(kernel_type, data_X, None,gamma) #计算X核矩阵
    Ky = kernel(kernel_type, data_Y, None, gamma) #计算Y核矩阵
    n_sample = data_X.shape[1]
    H = np.identity(n_sample) - 1/n_sample*np.ones([n_sample,1])@np.ones([1,n_sample])
    KX = Kx @ H
    KY = Ky @ H
    r = np.trace(KX@KY)
    return r/ (n_sample - 1)**2
# def conditioned_hisc(data_X,data_Y,data_Z,kernel_type = 'linear',gamma = 1):
#     '''
#     徐棚编写的条件希尔伯特准则的测试代码,参考互信息的这个逻辑
#     Parameters
#     ----------
#     data_X : 变量X:(n_samples,1)
#     data_Y : 变量Y(n_samples,1)
#     data_Z : 变量Z(n_samples,1)
#     kernel_type: linear/rbf/poly and ....
#     Returns
#     -------
#     条件HISC的相关值
#     这个代码其实从本质上已经编写完成了，后面的工作就是需要确定该公式的一个有效性
#     '''
#     Kx = kernel(kernel_type, data_X, None,gamma) #计算X核矩阵
#     Ky = kernel(kernel_type, data_Y, None, gamma) #计算Y核矩阵
#     Kz = kernel(kernel_type, data_Z, None, gamma) #计算Z核矩阵
#     n_sample = data_X.shape[1]
#     H = np.identity(n_sample) - 1/n_sample*np.ones([n_sample,1])@np.ones([1,n_sample])
#     KX = Kx@H
#     KY = Ky@H
#     KZ = Kz@H
#     epthon = 1e-5#论文里面指定的
#     M_common = np.linalg.pinv(KZ + epthon*np.identity(n_sample))
#     result_matrix1 = KY@KX -2* KY @ KZ @ M_common @ M_common @KZ @ KX
#     result_matrix2 = KY@KZ @M_common @ M_common@ KZ @ KX @ KZ @M_common@ M_common@ KZ
#     matrix_norm = np.linalg.norm(Kz,ord='fro')
#     beta_z = n_sample**2 / (matrix_norm**2) #矩阵的F2范数求和,原来用的Kz,正常论文里面应该使用也就是经过归一化之后的矩阵
#     return beta_z/((n_sample-1)**2)*np.trace(result_matrix1 + result_matrix2) 
##下面是修正版希尔伯特条件算子
def conditioned_hisc(data_Y,data_X,data_Z,kernel_type = 'linear',gamma = 1):
    '''
    徐棚编写的条件希尔伯特准则的测试代码,参考互信息的这个逻辑
    Parameters
    ----------
    data_X : 变量X:(n_samples,1)
    data_Y : 变量Y(n_samples,1)
    data_Z : 变量Z(n_samples,1)
    kernel_type: linear/rbf/poly and ....
    Returns
    -------
    条件HISC的相关值
    这个代码其实从本质上已经编写完成了，后面的工作就是需要确定该公式的一个有效性
    '''
    XZ = np.vstack([data_X,data_Z])
    YZ = np.vstack([data_Y,data_Z])
    Kx = kernel(kernel_type, XZ, None,gamma) #计算X核矩阵
    Ky = kernel(kernel_type, YZ, None, gamma) #计算Y核矩阵
    Kz = kernel(kernel_type, data_Z, None, gamma) #计算Z核矩阵
    n_sample = data_X.shape[1]
    H = np.identity(n_sample) - 1/n_sample*np.ones([n_sample,1])@np.ones([1,n_sample])
    KX = Kx@H
    KY = Ky@H
    KZ = Kz@H
    # epthon = 1e-5#论文里面指定的
    epthon = 1e-5
    M_common = np.linalg.pinv(KZ + epthon*np.identity(n_sample))
    result_matrix1 = KY@KX -2* KY @ KZ @ M_common @ M_common @KZ @ KX
    result_matrix2 = KY@KZ @M_common @ M_common@ KZ @ KX @ KZ @M_common@ M_common@ KZ
    matrix_norm = np.linalg.norm(Kz,ord='fro')
    beta_z = n_sample**2 / (matrix_norm**2) #矩阵的F2范数求和,原来用的Kz,正常论文里面应该使用也就是经过归一化之后的矩阵
    return beta_z/((n_sample-1)**2)*np.trace(result_matrix1 + result_matrix2) 
from domain_corrextion import domain_correction_moudle

# X_S_new,X_T_new = domain_correction_moudle.PCA_CC_reference_class(X_S, Y_S, X_T, Y_T, n=1, reference_class=34,standard_sate=True)
X_S,Y_S,X_T,Y_T = data_loader('zscore')
X_S_new,X_T_new = X_S,X_T
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
scaler = StandardScaler()
# clf = SVC(kernel='rbf',gamma=0.5)
clf = KNeighborsClassifier(n_neighbors=1)
clf.fit(X_S,Y_S)
print('PCA_CC较准前',clf.score(X_T, Y_T))
X_S_new = scaler.fit_transform(X_S_new)
X_T_new = scaler.transform(X_T_new)
clf.fit(X_S_new,Y_S)
print('PCA_CC较准后',clf.score(X_T_new, Y_T))


from transfer_learning.transfer_algorithm import DDRCA_fake
from sklearn.svm import SVC
# knn_after_subspace_projection = SVC(kernel='rbf',C=41)
# knn_before_subspace_projection = SVC(kernel='rbf',C=41)
knn_before_subspace_projection = KNeighborsClassifier(n_neighbors=1)
knn_after_subspace_projection = KNeighborsClassifier(n_neighbors=1)
### Hyper Parameter in DRCA setting
from transfer_learning.parameter_searching import grid_search
import random
alpha_grid = [1e-4,1e-3,1e-2,1e-1,1,10,100,1000,10000]
beta_grid = [1e-4,1e-3,1e-2,1e-1,1,10,100,1000,10000]
detla_grid = [1e-4,1e-3,1e-2,1e-1,1,10,100,1000,10000]
mu_grid = [1e-4,1e-3,1e-2,1e-1,1,10,100,1000,10000]
# alpha_grid = [0.1,0.2,0.3,0.5,0.8,1,1.5,2,3]
# lambdas_grid = [0.1,0.2,0.3,0.5,0.8,1,1.5,2,3]
# detla_grid = [0.1,0.2,0.3,0.5,0.8,1,1.5,2,3]
len_grid = len(alpha_grid)*len(beta_grid)*len(detla_grid)*len(mu_grid)
# random.seed(0) #random_state, which is used to re_done experiment
# selected_grid_index = random.sample(range(len_grid),100) #select 100 possible parameter settings
lamba = 1
parameter_grid = {'alpha':alpha_grid,'beta':beta_grid,'detla':detla_grid,'mu':mu_grid}
generated_grid = grid_search(parameter_grid)
# generated_grid = pd.DataFrame(generated_grid)
# generated_grid = np.array(generated_grid.iloc[selected_grid_index]) # improve search speed
best_num_fea = -1
num_fea = X_S.shape[1]
train_acc_all = np.zeros(num_fea)
test_acc_all = np.zeros(num_fea)


X_all = np.vstack([X_S_new,X_T_new])
Y_all = np.vstack([Y_S.reshape(-1,1),Y_T.reshape(-1,1)])


label_set = np.unique(Y_S)
Y_S_unified = Y_S.copy()
for i in range(len(Y_S_unified)):
    for j in range(len(label_set)):
        if Y_S_unified[i] == label_set[j]:
            Y_S_unified[i] = j
            break
for gamma in [8]:  #8 配的老数据集
    hisc_value = -5000*np.ones([num_fea])
    for i in range(num_fea):
        hisc_value[i] = my_hsic(X_S_new[:,i].reshape(1,-1), Y_S_unified.reshape(1,-1)/6,'rbf',gamma=gamma) #一定需要注意这里到底进行归一化没有，如果没有，结果是不可靠的
        # hisc_value[i] = my_hsic(X_S_new[:,i].reshape(1,-1), Y_S_unified.reshape(1,-1)/6,'linear') #一定需要注意这里到底进行归一化没有，如果没有，结果是不可靠的
    condition_hisc = -5000*np.ones([num_fea,num_fea])
    hisc_value_between_features = -5000*np.ones([num_fea,num_fea])
    label = Y_S_unified/6
    for i in range(num_fea):
        for j in range(num_fea):
                condition_hisc[i][j] = conditioned_hisc(X_S_new[:,i].reshape(1,-1),label.reshape(1,-1),X_S_new[:,j].reshape(1,-1),'rbf',gamma=gamma)
                # condition_hisc[i][j] = conditioned_hisc(X_S_new[:,i].reshape(1,-1),label.reshape(1,-1),X_S_new[:,j].reshape(1,-1),'linear')
    F = [i for i in range(num_fea)] #initialize original feature set
    S = list()
    f0 = np.argmax(hisc_value)#feature selected in each interaction
    F.remove(f0)
    S.append(f0)
    for k in range(1,num_fea): #each iteration
        List = -5000*np.ones(num_fea)
        l1 = len(S) #selected feature set
        l2 = len(F) #candidate feature set
        min_value = 5000
        for i in range(l2): #备选传感器组合
            mi_sy = hisc_value[F[i]] # 备选传感器与label之间的互信息I(F{S-C};C)
            redun = 0
            min_value = 5000
            for j in range(l1):#已选传感器组合
                cmi_sfy = condition_hisc[F[i]][S[j]]#备选传感器与已选传感器之间的条件互信息
                if cmi_sfy<min_value:
                    min_value = cmi_sfy
            mrmd = min_value
            print(mrmd)
            List[F[i]] = mrmd
        f0 = np.argmax(List)
        S.append(f0)
        F.remove(f0)
    
    # indices = np.argsort(-hisc_value)
    indices = S
    if indices not in indices_all:
        indices_all.append(indices)
        print(indices)
    else:
        continue
    ######
    for feature_num in range(1,10):
        sub_index = indices[0:feature_num]
        sub_index.sort()
        X_S_selected = X_S_new[:,sub_index]
        X_T_selected = X_T_new[:,sub_index]
        # X_S_selected = X_S_new[:,0:feature_num]
        # X_T_selected = X_T_new[:,0:feature_num]
        # X_S_selected = scaler.fit_transform(X_S_selected)
        # X_T_selected = scaler.transform(X_T_selected)
        knn_before_subspace_projection.fit(X_S_selected,Y_S)
        acc1 = knn_before_subspace_projection.score(X_T_selected,Y_T)
        train_acc_all[feature_num-1] = acc1
        print('特征数',feature_num,'迁移前准确率',acc1)
        ddrca = DDRCA_fake(X_S_selected,Y_S,X_T_selected,Y_T)
        max_acc_after_projection = -1
        for alpha in [1e-4,1e-3,1e-2,1e-1,1,10,100,1000,10000]:
            for beta in [1e-4/3,1e-3/3,1e-2/3,1e-1/3,1/3,10/3,100/3,1000/3,10000/3]:
                for detla in [1e-4,1e-3,1e-2,1e-1,1,10,100,1000,10000]:
                    for mu in [1e-4,1e-3,1e-2,1e-1,1,10,100,1000,10000]:
        # for single_parameter in generated_grid:#对超参数lamba的调试
        #     alpha,lambdas,detla,mu = single_parameter[0],single_parameter[1]/114,single_parameter[2]/3,single_parameter[3]
                        eig_vec = ddrca.fit_transform(alpha,beta,detla,mu)
            ### for feature set {1,2,3,...feature_num}, subpace consist of d dimension (where d from 1 to feature_num)
                        max_dim = feature_num  
                        for dim in range(1,max_dim+1):
                            sub_eig_vec = eig_vec[:,0:dim]
                            X_S_transformed = np.dot(X_S_selected,sub_eig_vec)
                            X_T_transformed = np.dot(X_T_selected,sub_eig_vec)
                            
                            knn_after_subspace_projection.fit(X_S_transformed,Y_S)
                            acc2 = knn_after_subspace_projection.score(X_T_transformed,Y_T)
                            if acc2>=max_acc_after_projection:
                                max_acc_after_projection = acc2
                                # print(max_acc_after_projection)
                                # best_parameter_grid = {'alpha':single_parameter[0],'lambdas':single_parameter[1],'detla':single_parameter[2],'mu':single_parameter[3],'dim':dim}
                                best_parameter_grid = {'alpha':alpha,'beta':beta,'detla':detla,'mu':mu,'dim':dim}
                                # print(max_acc_after_projection,best_parameter_grid)
                    # print('特征数',feature_num,'迁移后最大准确率',max_acc_after_projection,'最优参数集合',best_parameter_grid)
        file_name = array_optimization_name + subspace_transfer_learning_algorithm + '_' +str(gamma)+ 'result(1NN)XS.txt'
        with open('G:\\result_chinese_medicine\\'+file_name, 'a+', encoding='utf-8') as f:
            print('特征数',feature_num,'迁移前准确率',acc1,file=f)
            print('特征数',feature_num,'迁移后最大准确率',max_acc_after_projection,'最优参数集合',best_parameter_grid,file=f)
        print('特征数',feature_num,'迁移后最大准确率',max_acc_after_projection,'最优参数集合',best_parameter_grid)
        test_acc_all[feature_num-1] = max_acc_after_projection

    
    print('ECHIC_G_WSP2110置零_中值减去基线',test_acc_all)
    from matplotlib import pyplot as plt
    import matplotlib
    # #matplotlib默认不显示中文
    font = {'family':'MicroSoft YaHei'}
    matplotlib.rc('font',**font)  #给matplotlib能够显示中文
    plt.figure(figsize=(14,8))
    x_list = [i for i in range(1,)]
    plt.plot(train_acc_all, color = 'r',label='before transfer_learning')
    plt.plot(test_acc_all, color = 'b',label='after transfer_learning')
    plt.xlim([1,X_S.shape[1]]) #这里与特征数保持一致
    plt.xlabel('Num of features')
    plt.ylabel('recognition rate')
    plt.title('DDRCA+HSIC(CMIM)_rbf(1NN分类器)XS_中值(WSP2110置零)'+str(gamma))
    # plt.title('DDRCA+HSIC(CMIM)_linear(1NN分类器)XS_中值')
    plt.legend(fontsize=16)
    plt.legend(loc="lower right", fontsize=16)
    plt.grid()
    plt.show()
    print("ECHICG"+str(gamma),test_acc_all)
    file_name = "./"+array_optimization_name+str(gamma) +"result.npy"
    np.save(file_name,test_acc_all)

    end_time = time.time()  # 结束时间
    elapsed_time = end_time - start_time  # 计算经过时间
    print(f"代码执行时间：{elapsed_time}秒")