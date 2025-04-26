
import sys
import time
sys.path.append("../../../")  
from feature_selection._core import mutual_information_estimater
import pandas as pd
array_optimization_name = 'MIM'
subspace_transfer_learning_algorithm = 'TCA'
import numpy as np
from sklearn.neighbors import KNeighborsClassifier
from sklearn.preprocessing import StandardScaler,MinMaxScaler,Normalizer

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

from transfer_learning.transfer_algorithm import TCA
from sklearn.manifold import LocallyLinearEmbedding
from feature_selection._core import mutual_information_estimater
from feature_selection.feature_selection_moudle import feature_selection_algorithms

X_all = np.vstack([X_S_new,X_T_new])
Y_all = np.vstack([Y_S.reshape(-1,1),Y_T.reshape(-1,1)])

estimater = mutual_information_estimater(X_S_new, Y_S)
model = feature_selection_algorithms(estimater,X_S_new, Y_S,normalize_label=True,uniform_label = True)
indices = model.optimization(array_optimization_name)
max_acc = -5000
best_num_fea = -1
num_fea = X_S.shape[1]
train_acc_all = np.zeros(num_fea)
test_acc_all = np.zeros(num_fea)
for feature_num in range(num_fea,num_fea+1):
    sub_index = indices[0:feature_num]
    sub_index.sort()
    X_S_selected = X_S_new[:,sub_index]
    X_T_selected = X_T_new[:,sub_index]
    
    for lamb in [1e-4,1e-3,1e-2,0.1,1,10,100,1000,10000]:
        for gamma in [1,50]: 
        
            max_dim = feature_num
            for dim in range(1, max_dim + 1):
                tca = TCA(kernel_type='rbf',dim=dim, lamb=lamb,gamma=gamma) 
                XS_reduced,XT_reduced = tca.fit(X_S_new,X_T_new)
                clf.fit(XS_reduced,Y_S)
                acc = clf.score(XT_reduced, Y_T)
                if acc > max_acc:
                    max_acc = acc
                    print(max_acc,lamb,gamma,dim)
        print("循环",dim)

    print(max_acc)

end_time = time.time()  
elapsed_time = end_time - start_time  
print(f"代码执行时间：{elapsed_time}秒")