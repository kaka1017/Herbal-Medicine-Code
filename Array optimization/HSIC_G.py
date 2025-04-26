
import sys
import time
sys.path.append("../../") 
import pandas as pd
array_optimization_name = 'HISC_G'
subspace_transfer_learning_algorithm = 'my_alg'
import numpy as np
from sklearn.neighbors import KNeighborsClassifier
from sklearn.preprocessing import StandardScaler,MinMaxScaler,Normalizer
from sklearn import metrics

start_time = time.time()
def data_loader(scaler_name ='zscore'):
    
    
    
    
    
    X_S_df = pd.read_csv("batch1_medianvalue.csv", header=None)
    X_T_df = pd.read_csv("batch2_medianvalue2.csv", header=None)
    print(X_S_df)  
    print(X_T_df)  
    class_list = [15, 34, 106, 274]  
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
    
    print("传感器个数",len(meti_oxide))
    X_S[:, dict_array['WSP2110']] = np.zeros([114, 1])
    X_T[:, dict_array['WSP2110']] = np.zeros([48, 1])
    character = list()
    for string in meti_oxide:
        for index in dict_array[string]:
            character.append(index)
    character.sort()
    X_S =X_S[:,character]
    X_T = X_T[:,character]
    
    
    return X_S,Y_S,X_T,Y_T


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




def hsic(Kx, Ky):
    Kxy = np.dot(Kx, Ky)
    n = Kxy.shape[0]
    h = np.trace(Kxy) / n**2 + np.mean(Kx) * np.mean(Ky) - 2 * np.mean(Kxy) / n
    return h * n**2 / (n - 1)**2

def my_hsic(data_X,data_Y,kernel_type = 'linear',gamma = 1):
    Kx = kernel(kernel_type, data_X, None,gamma) 
    Ky = kernel(kernel_type, data_Y, None, gamma) 
    n_sample = data_X.shape[1]
    H = np.identity(n_sample) - 1/n_sample*np.ones([n_sample,1])@np.ones([1,n_sample])
    KX = Kx @ H
    KY = Ky @ H
    r = np.trace(KX@KY)
    return r/ (n_sample - 1)**2
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


from transfer_learning.transfer_algorithm import my_alg
from sklearn.svm import SVC


knn_before_subspace_projection = KNeighborsClassifier(n_neighbors=1)
knn_after_subspace_projection = KNeighborsClassifier(n_neighbors=1)

from transfer_learning.parameter_searching import grid_search
import random
alpha_grid = [1e-4,1e-3,1e-2,1e-1,1,10,100,1000,10000]
beta_grid = [1e-4,1e-3,1e-2,1e-1,1,10,100,1000,10000]
detla_grid = [1e-4,1e-3,1e-2,1e-1,1,10,100,1000,10000]
mu_grid = [1e-4,1e-3,1e-2,1e-1,1,10,100,1000,10000]



len_grid = len(alpha_grid)*len(beta_grid)*len(detla_grid)*len(mu_grid)


lamba = 1
parameter_grid = {'alpha':alpha_grid,'beta':beta_grid,'detla':detla_grid,'mu':mu_grid}
generated_grid = grid_search(parameter_grid)


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
            
hisc_value = -5000*np.ones([num_fea])
for i in range(num_fea):
    hisc_value[i] = my_hsic(X_S_new[:,i].reshape(1,-1), Y_S_unified.reshape(1,-1)/6,'rbf',gamma=8) 
    

indices = np.argsort(-hisc_value)

for feature_num in range(1,11):
    sub_index = indices[0:feature_num]
    sub_index.sort()
    X_S_selected = X_S_new[:,sub_index]
    X_T_selected = X_T_new[:,sub_index]
    
    
    
    
    knn_before_subspace_projection.fit(X_S_selected,Y_S)
    acc1 = knn_before_subspace_projection.score(X_T_selected,Y_T)
    train_acc_all[feature_num-1] = acc1
    print('特征数',feature_num,'迁移前准确率',acc1)
    my_alg1 = my_alg(X_S_selected,Y_S,X_T_selected,Y_T)
    max_acc_after_projection = -1
    for single_parameter in generated_grid:
        alpha,beta,detla,mu = single_parameter[0],single_parameter[1]/3,single_parameter[2],single_parameter[3]
        eig_vec = my_alg1.fit_transform(alpha,beta,detla,mu)
        
        max_dim = feature_num  
        for dim in range(1,max_dim+1):
            sub_eig_vec = eig_vec[:,0:dim]
            X_S_transformed = np.dot(X_S_selected,sub_eig_vec)
            X_T_transformed = np.dot(X_T_selected,sub_eig_vec)
            
            knn_after_subspace_projection.fit(X_S_transformed,Y_S)
            acc2 = knn_after_subspace_projection.score(X_T_transformed,Y_T)
            if acc2>max_acc_after_projection:
                max_acc_after_projection = acc2
                best_parameter_grid = {'alpha':alpha,'beta':beta,'detla':detla,'dim':dim}
    
    file_name = array_optimization_name + subspace_transfer_learning_algorithm + '_' + 'result(1NN)XS.txt'
    with open('G:\\result_chinese_medicine\\'+file_name, 'a+', encoding='utf-8') as f:
        print('特征数',feature_num,'迁移前准确率',acc1,file=f)
        print('特征数',feature_num,'迁移后最大准确率',max_acc_after_projection,'最优参数集合',best_parameter_grid,file=f)
    print('特征数',feature_num,'迁移后最大准确率',max_acc_after_projection,'最优参数集合',best_parameter_grid)
    test_acc_all[feature_num-1] = max_acc_after_projection

print(test_acc_all)
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

plt.title('my_alg+HSIC_linear(1NN分类器)XS_中值')
plt.legend(fontsize=16)
plt.legend(loc="lower right", fontsize=16)
plt.grid()

np.save("./acc_HISC_G.npy",test_acc_all)

end_time = time.time()  
elapsed_time = end_time - start_time  
print(f"代码执行时间：{elapsed_time}秒")