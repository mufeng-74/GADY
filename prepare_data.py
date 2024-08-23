# from utils.AnomalyGeneration import *
from scipy import sparse
import datetime
import pandas as pd
import numpy as np
import pickle
import time
import os
import math
import argparse
from sklearn.cluster import SpectralClustering

# device_string = 'cuda:{}'.format(GPU) if torch.cuda.is_available() else 'cpu'
# device = torch.device(device_string)

def anomaly_generation(data,anomaly_pers,train_per,seed,batch_size,dataset):
    np.random.seed(seed)
    sources = data.u.values.reshape(-1,1).astype(int)
    destinations = data.i.values.reshape(-1,1).astype(int)
    timestamps = data.ts.values.reshape(-1,1)
    labels = data.label.values.reshape(-1,1).astype(int)
    all_data = np.hstack((sources,destinations,timestamps,labels))
    all_data = all_data[np.lexsort([all_data.T[-2]])]   #sort data by timestamp
    # val_time, test_time = list(np.quantile(data.ts, [0.70, 0.85]))
    train_num = int(np.floor( train_per* len(data.u.values)))
    data_train = all_data[0:train_num,:]
    data_test = all_data[train_num:,:]
    
    # kk = 42
    kk = math.floor(np.unique(data.u.values).shape[0]/37)
    sc = SpectralClustering(kk, affinity='precomputed', n_init=10, assign_labels = 'discretize',n_jobs=-1)
    adjacency_matrix = edgeList2Adj(np.hstack((data.u.values.reshape(-1,1),data.i.values.reshape(-1,1))))
    labels = sc.fit_predict(adjacency_matrix)  #谱聚类之后获得的子图标签

    #生成测试时的异常边
    idx_1_test = np.expand_dims(np.transpose(np.random.choice(np.unique(data.u.values), 3*np.size(data_test, 0))), axis=1)
    idx_2_test = np.expand_dims(np.transpose(np.random.choice(np.unique(data.u.values), 3*np.size(data_test, 0))), axis=1)
    generate_edges_test = np.concatenate((idx_1_test, idx_2_test), axis=1)
    fake_edges_test = np.array([x for x in generate_edges_test if labels[x[0] - 1] != labels[x[1] - 1]])
    fake_edges_test = processEdges(fake_edges_test, data_test[:,[0,1]])
    begin = 0
    assert fake_edges_test.shape[0] >= data_test.shape[0]
    for anomaly_rate in anomaly_pers:
        begin = 0
        print('[#s] generating anomalous dataset...\n', datetime.datetime.now())
        print('[#s] initial network edge percent: #.1f##, anomaly percent: #.1f##.\n', datetime.datetime.now(),
            train_per * 100, anomaly_rate * 100)
        gap = int(np.ceil(anomaly_rate * batch_size))
        test_batch_size = batch_size - gap
        num_test_instance = data_test.shape[0]
        num_test_batch = math.ceil(num_test_instance / test_batch_size)        
        begin = 0
        for k in range(num_test_batch):
            s_idx = k * test_batch_size
            e_idx = min(num_test_instance, s_idx + test_batch_size)
            batch_num =  e_idx-s_idx
            
            # anomaly_num = min(batch_num,int(np.ceil(anomaly_rate * batch_num)))
            anomaly_num = int(np.ceil(anomaly_rate * batch_num))
            test_instance = data_test[s_idx:e_idx,:]
            fake_instance = fake_edges_test[begin:begin+anomaly_num,:]
            begin = begin +anomaly_num 
            
            fake_ts = np.random.randint(min(test_instance[:,-2]),max(test_instance[:,-2]+1),np.size(fake_instance, 0)).reshape(-1,1)
            fake_label = np.zeros_like(fake_ts)
            fake_data = np.hstack((fake_instance,fake_ts,fake_label))
            assert len(fake_instance) != 0
            
            test_instance = np.vstack((test_instance,fake_data))
            test_instance = test_instance[np.lexsort([test_instance.T[-2]])]  

            if k==0:
                anomaly_data = test_instance
            else:
                anomaly_data = np.vstack((anomaly_data,test_instance))
        TEST_SIZE = math.ceil(num_test_instance/batch_size)
        for i in range(TEST_SIZE-1):
            s_idx = i * batch_size
            e_idx = min(num_test_instance, s_idx + batch_size)
            true_label = anomaly_data[s_idx:e_idx,-1]
            assert len(np.nonzero(np.ones_like(true_label)-true_label)[0]) != 0            
            
        all_data = np.vstack((data_train,anomaly_data))
        idx_list = [int(x)+1 for x in range(np.size(all_data, 0))]
        all_data = np.hstack((all_data,np.array(idx_list).reshape(-1,1)))
        data_train2 = all_data[0:train_num,:]
        data_test2 = all_data[train_num:,:]
        np.save('./data/'+dataset+str(anomaly_rate)+'test.npy',data_test2)
        np.save('./data/'+dataset+str(anomaly_rate)+'train.npy',data_train2)



# 做了两件事，把边的数据去重，并且算了节点数目
def preprocessDataset(dataset,use_features):
    print('Preprocess dataset: ' + dataset)
    if dataset in ['digg','uci'] :
        edges = np.loadtxt(
            'data/' +
            dataset,
            dtype=float,
            comments='%',
            delimiter=' ')
        index = np.nonzero(edges[:,0] - edges[:,1])[0]
        u_list = edges[index,0].astype(int).tolist()
        i_list = edges[index,1].astype(int).tolist()
        label_list = edges[index,2].astype(int).tolist()   #1 represents normal while 0 represents abnormal
        ts_list = edges[index,3].astype(float).tolist()
        if use_features:
            feat = edges[index,4:]
        idx_list = [int(x)+1 for x in range(len(u_list))]
    elif dataset in ['btc_alpha', 'btc_otc']:
        if dataset == 'btc_alpha':
            file_name = 'data/' + 'soc-sign-bitcoinalpha.csv'
        elif dataset =='btc_otc':
            file_name = 'data/' + 'soc-sign-bitcoinotc.csv'
        edges = np.loadtxt(
            file_name,
            dtype=float,
            comments='%',
            delimiter=',')
        index = np.nonzero(edges[:,0] - edges[:,1])[0]
        u_list = edges[index,0].astype(int).tolist()
        i_list = edges[index,1].astype(int).tolist()
        label_list = np.ones([len(u_list)]).astype(int).tolist()   #1 represents normal while 0 represents abnormal
        ts_list = edges[index,3].astype(float).tolist()
        if use_features:
            feat = edges[index,4:]
        idx_list = [int(x)+1 for x in range(len(u_list))] 
    elif dataset == 'email_dnc':
        edges = np.loadtxt(
            './data/email-dnc.edges',
            dtype=float,
            comments='%',
            delimiter=',',
            encoding="utf-8-sig")
        index = np.nonzero(edges[:,0] - edges[:,1])[0]
        u_list = edges[index,0].astype(int).tolist()
        i_list = edges[index,1].astype(int).tolist()
        label_list = np.ones([len(u_list)]).astype(int).tolist()   #1 represents normal while 0 represents abnormal
        ts_list = edges[index,2].astype(float).tolist()
        if use_features:
            feat = edges[index,4:]
        idx_list = [int(x)+1 for x in range(len(u_list))]
    elif dataset == 'as_topology':
        edges = np.loadtxt(
            './data/tech-as-topology.edges',
            dtype=float,
            comments='%',
            delimiter=' ')
        index = np.nonzero(edges[:,0] - edges[:,1])[0]
        u_list = edges[index,0].astype(int).tolist()
        i_list = edges[index,1].astype(int).tolist()
        label_list = edges[index,2].astype(int).tolist()   #1 represents normal while 0 represents abnormal
        ts_list = edges[index,3].astype(float).tolist()
        if use_features:
            feat = edges[index,4:]
        idx_list = [int(x)+1 for x in range(len(u_list))]
        
    elif dataset in ['reddit','wikipedia']:
        if dataset == 'reddit':
            file_name = 'data/raw/' + 'reddit.csv'
        elif dataset =='wikipedia':
            file_name = 'data/raw/' + 'wikipedia.csv'
        with open(file_name) as f:
            s = next(f)
            lines = f.read().splitlines()
        edges = [[float(r) for r in row.split(',')] for row in lines]
        edges = np.array(edges)
        edges = edges[edges[:, 3].argsort()]
        edges = edges[:, 0:2].astype(dtype=int)
    print("Preprocess end!")
    pd.DataFrame({'u': u_list,
                       'i': i_list,
                       'ts': ts_list,
                       'label': label_list,
                       'idx': idx_list}).to_csv("./data/"+dataset+".csv")
    if use_features:
        np.save("./data/ml_{}.npy".format(dataset),feat)
    max_idx = max(max(u_list), max(i_list))
    rand_feat = np.zeros((max_idx + 1, 172))
    np.save('./data/ml_{}_node.npy'.format(dataset), rand_feat)
    

    #生成数据集的主要函数
def generateDataset(dataset, train_per=0.7, anomaly_pers=[0.01, 0.05, 0.1],use_features=False,batch_size=200):
    print('Generating data with anomaly for Dataset: ', dataset)
    if not os.path.exists('./data/' + dataset+".csv"):
        preprocessDataset(dataset,use_features)
    data = pd.read_csv('./data/'+dataset+'.csv') 
    # m是边的数目，n是节点的数目
    anomaly_generation(data, anomaly_pers=anomaly_pers, train_per=train_per,seed=1,batch_size=batch_size,dataset=dataset)

    
def edgeList2Adj(data):
    """
    converting edge list to graph adjacency matrix
    :param data: edge list
    :return: adjacency matrix which is symmetric
    """
    data = tuple(map(tuple, data))
    n = max(max(user, item) for user, item in data)  # Get size of matrix
    matrix = np.zeros((n, n))
    for user, item in data:
        matrix[user - 1][item - 1] = 1  # Convert to 0-based index.
        matrix[item - 1][user - 1] = 1  # Convert to 0-based index.
        if user == item:
            print("error")
    return matrix

def processEdges(fake_edges, data):
    """
    remove self-loops and duplicates and order edge
    :param fake_edges: generated edge list
    :param data: orginal edge list
    :return: list of edges   作用是去除自环，并且过滤掉在真实数据集中的边
    """
    non_loop = np.nonzero(fake_edges[:, 0] - fake_edges[:,1] !=0)
    fake_edges = fake_edges[non_loop]
    a = fake_edges.tolist()
    b = data.tolist()
    c = []
    label = np.ones(fake_edges.shape[0])
    i = 0
    for value in a:
        i = i + 1
        if value in b:
            # c.append(value)
            label[i-1] = 0 
    fake_edges = fake_edges[np.nonzero(label)[0],:]
    return fake_edges


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset', type=str, choices=['uci', 'digg', 'btc_alpha', 'btc_otc','email_dnc','as_topology','reddit','wikipedia'], default='uci')
    parser.add_argument('--anomaly_per' ,choices=[0.01, 0.05, 0.1] , type=float, default=None)
    parser.add_argument('--train_per', type=float, default=0.5)
    parser.add_argument('--use_features', type=bool, default=False)
    parser.add_argument('--batch_size', type=int, default=200)
    args = parser.parse_args()

    if args.anomaly_per is None:
        anomaly_pers = [0.01, 0.05, 0.10]
    else:
        anomaly_pers = [args.anomaly_per]

    generateDataset(args.dataset, train_per=args.train_per, anomaly_pers=anomaly_pers,use_features=args.use_features,batch_size=args.batch_size)