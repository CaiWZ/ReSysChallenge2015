import torch

#使用gpu
device=torch.device('cuda' if torch.cuda.is_available() else 'cpu')

embed_dim = 128
batch_size = 1024
num_embeds = 52739

#数据文件夹
folder='data'

#click
c_file='data/yoochoose-data/yoochoose-clicks.dat'
#buy
b_file='data/yoochoose-data/yoochoose-buys.dat'

#click_index
c_index=['session_id','timestamp','item_id','category']
#buy_index
b_index=['session_id','timestamp','item_id','price','quantity']

#test data
test_data_file='data/yoochoose-test.dat'

#processed file folder
process_path='data/yoochoose_click_binary_1M_sess.dataset'

#training paramters
num_workers = 4 #data-loading
grad_clip = 5. #clip gradients at an absolute of
print_freq = 10 #print training/validation stats every batchs
checkpoint = None #path of checkpoint