import math
import sys
import argparse
import torch
import numpy as np
from utils.data_processing import get_data
from utils.utils import get_data_settings
import tqdm
from scipy.sparse import coo_matrix
import os
cpu_num = 10 # 这里设置成你想运行的CPU个数
os.environ["OMP_NUM_THREADS"] = str(cpu_num)  # noqa
os.environ["MKL_NUM_THREADS"] = str(cpu_num) # noqa
torch.set_num_threads(cpu_num )



torch.manual_seed(0)
np.random.seed(0)

parser = argparse.ArgumentParser('PINT - positional features')
parser.add_argument('-d', '--data', type=str, help='Dataset name (eg. wikipedia or reddit)', default='uci')
parser.add_argument('-ds', '--data_split', type=str, help='train, test, join', default='train')
parser.add_argument('--bs', type=int, default=200, help='Batch_size')
parser.add_argument('--gpu', type=int, default=0, help='Idx for the gpu to use')
parser.add_argument('--r_dim', type=int, default=4, help='dim for R')
parser.add_argument('--anomaly_per', type=float, default=0.1, help='the anomaly rate in the test data')

try:
	args = parser.parse_args()
except:
	parser.print_help()
	sys.exit(0)

BATCH_SIZE = args.bs
GPU = args.gpu
DATA = args.data
SPLIT = args.data_split

### Extract data for training, validation and testing  成功导入了数据
node_features, edge_features, full_data, train_data, test_data = get_data(DATA,anomaly_per = args.anomaly_per)

# Set device
device_string = 'cuda:{}'.format(GPU) if torch.cuda.is_available() else 'cpu'
device = torch.device(device_string)

num_instance = len(train_data.sources) #计算了例子的总数
num_batch = math.ceil(num_instance / BATCH_SIZE) #计算需要多少个batch

nextV, nextR = [], []

partition_size, last = get_data_settings(args.data)
#要特别留意V、R、P的作用
r_dim = args.r_dim
R = torch.zeros((node_features.shape[0], node_features.shape[0], r_dim), requires_grad=False, dtype=int)  # 维度是节点的最大索引
P = torch.zeros((r_dim, r_dim), requires_grad=False, dtype=int) #r_dim的作用是什么
P[1:, :-1] = torch.eye(r_dim - 1, requires_grad=False, dtype=int)
for i in range(node_features.shape[0]):
	R[i, i, 0] = 1  # 初始化成0
V = torch.eye(node_features.shape[0], requires_grad=False, dtype=int)
Rprime = R.clone()  # 太太太慢了！！
prevV, prevR = V.clone(), R.clone()



def update_VR(sources_batch, destinations_batch, V, R, P):
	for idx in range(sources_batch.shape[0]):  #复杂度是On^2
		u, v = sources_batch[idx], destinations_batch[idx]
		for i in V[u].nonzero():  #对头结点进行处理
			R[i, v, :] = (P @ Rprime[i, u, :].T).T + Rprime[i, v, :]  # @表示矩阵乘法
		for i in V[v].nonzero():  #对尾结点进行处理
			R[i, u] = (P @ Rprime[i, v, :].T).T + Rprime[i, u, :]
		V[u, :] = V[u, :] + V[v, :] - V[u, :] * V[v, :]  # 对应论文中的等式9
		V[v, :] = V[u, :]
	return V, R



if SPLIT == 'train':
	for k in tqdm.tqdm(range(0, num_batch)):
		batch_idx = k

		start_idx = batch_idx * BATCH_SIZE
		end_idx = min(num_instance, start_idx + BATCH_SIZE)
		sources_batch, destinations_batch = train_data.sources[start_idx:end_idx], \
			train_data.destinations[start_idx:end_idx]
		edge_idxs_batch = train_data.edge_idxs[start_idx: end_idx]
		timestamps_batch = train_data.timestamps[start_idx:end_idx]

		prevV[:,:], prevR[:,:,:] = V[:,:], R[:,:,:]
		V, R = update_VR(sources_batch, destinations_batch, V, R, P)

		nextV.append((V - prevV).to('cpu').to_sparse())   # 最终的保存信息
		nextR.append((R - prevR).to('cpu').to_sparse())	  #最终的保存信息
		if ((k + 1) % partition_size == 0) or ((k + 1) == num_batch):  # savepoint， 防止内存占用过大
			prt = k // partition_size
			torch.save([nextV, nextR], 'pos_features/' + args.data + '_nextVR_part_' + str(prt) + '_bs_' + str(args.bs) + '_rdim_' + str(args.r_dim) + str(args.anomaly_per))
			nextV, nextR = [], []
else:
	nV, nR = torch.load('pos_features/' + args.data + '_nextVR_part_' + str(last) + '_bs_' + str(args.bs) + '_rdim_'+ str(args.r_dim) + str(args.anomaly_per))
	V, R = nV[-1].to_dense().clone(), nR[-1].to_dense().clone()  # save state at end of training
	TEST_BATCH_SIZE = args.bs
	if SPLIT == 'test':
		num_test_instance = len(test_data.sources)
		num_test_batch = math.ceil(num_test_instance / TEST_BATCH_SIZE)

		test_V, test_R = [], []
		prevV, prevR = V.clone(), R.clone()
		for k in range(num_test_batch):
			prevV[:,:], prevR[:,:,:] = V[:,:], R[:,:,:]
			s_idx = k * TEST_BATCH_SIZE
			e_idx = min(num_test_instance, s_idx + TEST_BATCH_SIZE)
			sources_batch = test_data.sources[s_idx:e_idx]
			destinations_batch = test_data.destinations[s_idx:e_idx]

			V, R = update_VR(sources_batch, destinations_batch, V, R, P)

			test_V.append((V - prevV).to('cpu').to_sparse())
			test_R.append((R - prevR).to('cpu').to_sparse())

		torch.save([test_V, test_R], 'pos_features/' + args.data + '_VR_test_bs_' + str(args.bs) + '_rdim_' + str(args.r_dim) + str(args.anomaly_per))



