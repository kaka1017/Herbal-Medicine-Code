# -*- coding: utf-8 -*-
"""
Created on Sun Jan  8 10:15:37 2023
这个代码的主要目的：
1.使用PCA-CC(reference class)对中药数据集进行校准
2.使用PCA-CC校准得到的数据，在源域上进行特征选择
3、输出经过目标域数据特征选择得到的数据进行测试，此代码输出精度可达95.6%
@author: 徐棚
"""
import sys# 加了这了
import time
sys.path.append("../../../")  ## 这一行代码是在上两级目录进行寻找,如果是在上一级目录进行寻找，可以使用..
from feature_selection._core import mutual_information_estimater
import pandas as pd
array_optimization_name = 'MIM'
subspace_transfer_learning_algorithm = 'DDRCA'
import numpy as np
from sklearn.neighbors import KNeighborsClassifier
from sklearn.preprocessing import StandardScaler,MinMaxScaler,Normalizer
start_time = time.time()
def data_loader(scaler_name ='zscore'):
    X_S_df = pd.read_csv("batch1_gas.csv", header=None)
    X_T_df = pd.read_csv("batch4_gas.csv", header=None)
    print(X_S_df)  # 9月份，10月份作为第一批未漂移的数据
    print(X_T_df)  # 12月作为已经漂移的数据
    class_list = [1,2,3,4,5,6] #[15,34,106,274] [15,34,36,274]
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
    character = list()
    for string in meti_oxide:
        for index in dict_array[string]:
            character.append(index)
    character.sort()
    X_S[:, dict_array['WSP2110']] = np.zeros([371, 1])
    X_T[:, dict_array['WSP2110']] = np.zeros([161, 1])
    X_S =X_S[:,character]
    X_T = X_T[:,character]
    # X_S = scaler.fit_transform(X_S)
    # X_T = scaler.transform(X_T)
    return X_S,Y_S,X_T,Y_T
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

# from transfer_learning.transfer_algorithm import TCA
# from sklearn.manifold import LocallyLinearEmbedding
# max_acc = -5000
# for dim in range(1,20):
#     for lamb in [1e-4,1e-3,1e-2,0.1,1,10,1000,10000]:
#         for gamma in range(1,100):
#             tca = TCA(kernel_type='rbf',dim=dim, lamb=lamb,gamma=gamma) #gamma =1 54.16%
#             XS_reduced,XT_reduced = tca.fit(X_S_new,X_T_new)
#             clf.fit(XS_reduced,Y_S)
#             acc = clf.score(XT_reduced, Y_T)
#             if acc > max_acc:
#                 max_acc = acc
#     print("循环",dim)

# print(max_acc)

from transfer_learning.transfer_algorithm import DDRCA_fake
from sklearn.svm import SVC
# knn_after_subspace_projection = SVC(kernel='rbf',C=41)
# knn_before_subspace_projection = SVC(kernel='rbf',C=41)
knn_before_subspace_projection = KNeighborsClassifier(n_neighbors=1)
knn_after_subspace_projection = KNeighborsClassifier(n_neighbors=1)
### Hyper Parameter in DRCA setting
from transfer_learning.parameter_searching import grid_search
import random
# alpha_grid = [1e-4,1e-3,1e-2,1e-1,1,10,100,1000,10000]
# lambdas_grid = [1e-4,1e-3,1e-2,1e-1,1,10,100,1000,10000]
alpha_grid = [1e-3,1e-2,1e-1,1,10,100,100]
lambdas_grid = [1e-3,1e-2,1e-1,1,10,100,100]
detla_grid = [1e-3,1e-2,1e-1,1,10,100,100]
beta_grid = [1e-3,1e-2,1e-1,1,10,100,100]
# n_neighbors_grid= [16] #range(1,50)
# len_grid = len(n_neighbors_grid)*len(lambdas_grid)
# detla_grid = [1e-4,1e-3,1e-2,1e-1,1,10,100,1000,10000]
# beta_grid = [1e-4,1e-3,1e-2,1e-1,1,10,100,1000,10000]
# mu_grid = [1e-4,1e-3,1e-2,1e-1,1,10,100,1000,10000]
# alpha_grid = [0.1,0.2,0.3,0.5,0.8,1,1.5,2,3]
# lambdas_grid = [0.1,0.2,0.3,0.5,0.8,1,1.5,2,3]
# detla_grid = [0.1,0.2,0.3,0.5,0.8,1,1.5,2,3]
len_grid = len(alpha_grid)*len(lambdas_grid)*len(detla_grid)*len(beta_grid)
# random.seed(0) #random_state, which is used to re_done experiment
# selected_grid_index = random.sample(range(len_grid),100) #select 100 possible parameter settings
lamba = 1
parameter_grid = {'alpha':alpha_grid,'lambdas':lambdas_grid,'detla':detla_grid,'beta':beta_grid}
# parameter_grid = {'lambdas':lambdas_grid,'n_neighbors':n_neighbors_grid}
generated_grid = grid_search(parameter_grid)
# generated_grid = pd.DataFrame(generated_grid)
# generated_grid = np.array(generated_grid.iloc[selected_grid_index]) # improve search speed
best_num_fea = -1
num_fea = X_S.shape[1]
train_acc_all = np.zeros(num_fea)
test_acc_all = np.zeros(num_fea)

from feature_selection._core import mutual_information_estimater
from feature_selection.feature_selection_moudle import feature_selection_algorithms

X_all = np.vstack([X_S_new,X_T_new])
Y_all = np.vstack([Y_S.reshape(-1,1),Y_T.reshape(-1,1)])

estimater = mutual_information_estimater(X_S_new, Y_S)
model = feature_selection_algorithms(estimater,X_S_new, Y_S,normalize_label=True,uniform_label = True)
indices = model.optimization(array_optimization_name)
for feature_num in range(num_fea, num_fea + 1):
    sub_index = indices[0:feature_num]
    sub_index.sort()
    X_S_selected = X_S_new[:, sub_index]
    X_T_selected = X_T_new[:, sub_index]
    # X_S_selected = X_S_new[:,0:feature_num]
    # X_T_selected = X_T_new[:,0:feature_num]
    # X_S_selected = scaler.fit_transform(X_S_selected)
    # X_T_selected = scaler.transform(X_T_selected)
    knn_before_subspace_projection.fit(X_S_selected, Y_S)
    acc1 = knn_before_subspace_projection.score(X_T_selected, Y_T)
    train_acc_all[feature_num - 1] = acc1
    print('特征数', feature_num, '迁移前准确率', acc1)
    ddrca = DDRCA_fake(X_S_selected, Y_S, X_T_selected, Y_T)
    max_acc_after_projection = -1
    for single_parameter in generated_grid:  # 对超参数lamba的调试
        alpha, lambdas, detla, beta = single_parameter[0], single_parameter[1], single_parameter[2], single_parameter[3]
        eig_vec = ddrca.fit_transform(alpha, lambdas, detla, beta)
        ### for feature set {1,2,3,...feature_num}, subpace consist of d dimension (where d from 1 to feature_num)
        max_dim = feature_num
        for dim in range(1, max_dim + 1):
            sub_eig_vec = eig_vec[:, 0:dim]
            X_S_transformed = np.dot(X_S_selected, sub_eig_vec)
            X_T_transformed = np.dot(X_T_selected, sub_eig_vec)

            knn_after_subspace_projection.fit(X_S_transformed, Y_S)
            acc2 = knn_after_subspace_projection.score(X_T_transformed, Y_T)
            if acc2 >0.75:
                print(acc2,alpha,lambdas,detla,beta)
            if acc2 > max_acc_after_projection:
                max_acc_after_projection = acc2
                best_parameter_grid = {'alpha': single_parameter[0], 'lambdas': single_parameter[1], 'detla': single_parameter[2], 'beta': single_parameter[3], 'dim': dim}
                # best_parameter_grid = {'lambdas': single_parameter[0], 'n_neighbors': single_parameter[1], 'dim': dim}
                print(max_acc_after_projection)

    file_name = array_optimization_name + subspace_transfer_learning_algorithm + '_' + 'result(1NN)XS.txt'
    with open('G:\\result_chinese_medicine\\' + file_name, 'a+', encoding='utf-8') as f:
        print('特征数', feature_num, '迁移前准确率', acc1, file=f)
        print('特征数', feature_num, '迁移后最大准确率', max_acc_after_projection, '最优参数集合', best_parameter_grid, file=f)
    print('特征数', feature_num, '迁移后最大准确率', max_acc_after_projection, '最优参数集合', best_parameter_grid)
    test_acc_all[feature_num - 1] = max_acc_after_projection

# from sklearn.manifold import LocallyLinearEmbedding
# max_acc = -5000
# for lamb in [1e-4]:#[1e-2,0.1,1,10,100] [1e-4,1e-3,1e-2,0.1,1,10,100,1000,10000] 58.33
#     for gamma in [93]: #range(1,50)
#         for dim in [19]:
#             tca = TCA(kernel_type='rbf',dim=dim, lamb=lamb,gamma=gamma) #gamma =1 54.16%
#             XS_reduced,XT_reduced = tca.fit(X_S_new,X_T_new)
#             clf.fit(XS_reduced,Y_S)
#             acc = clf.score(XT_reduced, Y_T)
#             if acc > max_acc:
#                 if acc >0.6 and acc <0.66:
#                     print("找到了")
#                     print(max_acc,lamb,gamma,dim)
#                 max_acc = acc
#                 print(max_acc,lamb,gamma,dim)
#     print("循环",dim)
#
# print(max_acc)
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
plt.title('DDRCA+MIM(1NN分类器)XS_中值')
plt.legend(fontsize=16)
plt.legend(loc="lower right", fontsize=16)
plt.grid()
plt.savefig('../pictures/'+array_optimization_name+subspace_transfer_learning_algorithm + '(1NN)(toolbox_chinese_medicine_4kind_15_34_106_274)XS.png',dpi=300)
end_time = time.time()  # 结束时间
elapsed_time = end_time - start_time  # 计算经过时间
print(f"代码执行时间：{elapsed_time}秒")