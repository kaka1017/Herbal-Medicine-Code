
import sys
import time
sys.path.append("../../../")  
from feature_selection._core import mutual_information_estimater
import pandas as pd
array_optimization_name = 'MIM'
subspace_transfer_learning_algorithm = 'MCSP'
import numpy as np 
from sklearn.neighbors import KNeighborsClassifier
from sklearn.preprocessing import StandardScaler,MinMaxScaler,Normalizer
start_time = time.time()
def data_loader(scaler_name ='zscore'):

    X_S_df = pd.read_csv("batch1_gas.csv", header=None)
    X_T_df = pd.read_csv("batch4_gas.csv", header=None)
    print(X_S_df)  
    print(X_T_df)  
    class_list = [1, 2, 3, 4, 5, 6]  
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
    X_S = np.array(X_S_df.iloc[:, 0:-3])  
    Y_S = np.array(X_S_df.iloc[:, -2])  
    X_T = np.array(X_T_df.iloc[:, 0:-3])  
    Y_T = np.array(X_T_df.iloc[:, -2])  
    sensor_all = ['tempture1','tempture2','humidity1','humidity2','pressure1','pressure2','TGS813','TGS2610D','MS1100','TGS826','TGS2602','TGS822','MG812','4S','4HS+',
                  'C6H6','SMD1007','WSP2110','MP4','MP135A','TGS2620','TGS2611E','AQ201','TGS8669','2M012','TGS2600','MR516','H2S','NH3','4CH3SH','SMD1013','SMD1001','MQ135','MP503','MQ3B','MQ137','4ETO-10','4OXV',
                  'MQ138','MP901','ME3-C2H6S','ME3-CH2O','PID-AH','VCC_+5V1','VCC5V-H1']
    dict_array = {}
    fea_enviorment = 2
    fea_other_sensor = 1
    start_index = 0
    for i in range(len(sensor_all)):
        if i<=5:
            dict_array[sensor_all[i]] = list(range(start_index,start_index+fea_enviorment)) 
            start_index = start_index + fea_enviorment
        elif i >=43:
            dict_array[sensor_all[i]] = list(range(start_index,start_index+fea_enviorment)) 
            start_index = start_index + fea_enviorment
        else:
            dict_array[sensor_all[i]] = list(range(start_index,start_index+fea_other_sensor)) 
            start_index = start_index + fea_other_sensor
    meti_oxide = ['TGS813','TGS2610D','TGS826','TGS2602','TGS822','TGS2620','TGS2611E','TGS8669','TGS2600','MS1100','MP4','MP135A','MQ135','MP503','MQ3B','MQ137','MQ138','MP901','2M012','AQ201','WSP2110']
    
    X_S[:, dict_array['WSP2110']] = np.zeros([371, 1])
    X_T[:, dict_array['WSP2110']] = np.zeros([161, 1])
    character = list()
    for string in meti_oxide:
        for index in dict_array[string]:
            character.append(index)
    character.sort()
    X_S =X_S[:,character]
    X_T = X_T[:,character]

    return X_S,Y_S,X_T,Y_T
from domain_corrextion import domain_correction_moudle

X_S,Y_S,X_T,Y_T = data_loader('zscore')
X_S_new,X_T_new = X_S,X_T
from sklearn.neighbors import KNeighborsClassifier 
from sklearn.svm import SVC
scaler = StandardScaler()

clf = KNeighborsClassifier(n_neighbors=1)
clf.fit(X_S,Y_S)
X_S_new = scaler.fit_transform(X_S_new)
X_T_new = scaler.transform(X_T_new)
clf.fit(X_S_new,Y_S)

from transfer_learning.transfer_algorithm import MCSP
from sklearn.svm import SVC

knn_before_subspace_projection = KNeighborsClassifier(n_neighbors=1)
knn_after_subspace_projection = KNeighborsClassifier(n_neighbors=1)

from transfer_learning.parameter_searching import grid_search
import random
alpha_grid = [1e-4,1e-3,1e-2,1e-1,1,10,100,1000,10000]
lambdas_grid = [1e-4,1e-3,1e-2,1e-1,1,10,100,1000,10000]
detla_grid = [1e-4,1e-3,1e-2,1e-1,1,10,100,1000,10000]
beta_grid = [1e-4,1e-3,1e-2,1e-1,1,10,100,1000,10000]

len_grid = len(alpha_grid)*len(lambdas_grid)*len(detla_grid)*len(beta_grid)

lamba = 1
parameter_grid = {'alpha':alpha_grid,'lambdas':lambdas_grid,'detla':detla_grid,'beta':beta_grid}
generated_grid = grid_search(parameter_grid)

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

for feature_num in range(num_fea,num_fea+1):
    sub_index = indices[0:feature_num]
    sub_index.sort()
    X_S_selected = X_S_new[:,sub_index]
    X_T_selected = X_T_new[:,sub_index]

    knn_before_subspace_projection.fit(X_S_selected,Y_S)
    acc1 = knn_before_subspace_projection.score(X_T_selected,Y_T)
    train_acc_all[feature_num-1] = acc1
    print('特征数',feature_num,'迁移前准确率',acc1)
    mcsp = MCSP(X_S_selected,Y_S,X_T_selected,Y_T)
    max_acc_after_projection = -1
    for single_parameter in generated_grid:
        alpha,lambdas,detla,beta = single_parameter[0],single_parameter[1],single_parameter[2],single_parameter[3]
        eig_vec = mcsp.fit_transform(alpha,lambdas,detla,beta)
        
        max_dim = feature_num  
        for dim in range(1,max_dim+1):
            sub_eig_vec = eig_vec[:,0:dim]
            X_S_transformed = np.dot(X_S_selected,sub_eig_vec)
            X_T_transformed = np.dot(X_T_selected,sub_eig_vec)
            
            knn_after_subspace_projection.fit(X_S_transformed,Y_S)
            acc2 = knn_after_subspace_projection.score(X_T_transformed,Y_T)
            if acc2>max_acc_after_projection:
                max_acc_after_projection = acc2
                best_parameter_grid = {'alpha':single_parameter[0],'lambdas':single_parameter[1],'detla':single_parameter[2],'beta':single_parameter[3],'dim':dim}
                print(max_acc_after_projection)
    
    file_name = array_optimization_name + subspace_transfer_learning_algorithm + '_' + 'result(1NN)XS.txt'
    with open('G:\\result_chinese_medicine\\'+file_name, 'a+', encoding='utf-8') as f:
        print('特征数',feature_num,'迁移前准确率',acc1,file=f)
        print('特征数',feature_num,'迁移后最大准确率',max_acc_after_projection,'最优参数集合',best_parameter_grid,file=f)
    print('特征数',feature_num,'迁移后最大准确率',max_acc_after_projection,'最优参数集合',best_parameter_grid)
    test_acc_all[feature_num-1] = max_acc_after_projection

print("第一篇对比MCSP算法迁移精度",max_acc_after_projection)
from matplotlib import pyplot as plt
import matplotlib

font = {'family':'MicroSoft YaHei'}
matplotlib.rc('font',**font)  
plt.figure(figsize=(14,8))
x_list = [i for i in range(1,)]
plt.plot(train_acc_all, color = 'r',label='before transfer_learning')
plt.plot(test_acc_all, color = 'b',label='after transfer_learning')
plt.xlim([1,X_S.shape[1]]) 
plt.xlabel('Num of features')
plt.ylabel('recognition rate')
plt.title('MCSP+mRMR(1NN分类器)XS_中值')
plt.legend(fontsize=16)
plt.legend(loc="lower right", fontsize=16)
plt.grid()
plt.savefig('../pictures/'+array_optimization_name+subspace_transfer_learning_algorithm + '(1NN)(toolbox_chinese_medicine_4kind_15_34_106_274)XS.png',dpi=300)
end_time = time.time()  
elapsed_time = end_time - start_time  
print(f"代码执行时间：{elapsed_time}秒")