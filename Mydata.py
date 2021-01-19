import torch
from sklearn.preprocessing import LabelEncoder
from torch_geometric.data import Data
from torch_geometric.data import InMemoryDataset
from tqdm import tqdm
# from config import process_path

df =  torch.load('data/yoochoose_click_binary_1M_sess.dataset')


class YoochooseBinaryDataset(InMemoryDataset):
    def __init__(self,root,transform=None,pre_transform=None):
        super(YoochooseBinaryDataset,self).__init__(root,transform,pre_transform)
        self.data, self.slices = torch.load(self.processed_paths[0])
        print('文件路径，',self.processed_paths[0])

        
    @property
    def raw_file_names(self):
        return []
    
    @property
    def processed_file_names(self):
        return ['yoochoose_click_binary_1M_sess.dataset']


    def download(self):
        pass


    def process(self):
        data_list = []

        grouped = df.groupby('session_id')
        for session_id, group in tqdm(grouped):
            session_item_id = LabelEncoder().fit_transform(group.item_id)
            group = group.reset_index(drop=True)
            group['session_item_id'] = session_item_id
            #loc首先得到，session_id相同的且在session_item_id到item_id列的
            #sort_values根据session_item_id排序
            #在根据item_id进行去重（drop_duplicates)
            #最后得到其值
            node_features = group.loc[group.session_id == session_id,['session_item_id','item_id']].sort_values(
                'session_item_id').item_id.drop_duplicates().values

            node_features = torch.LongTensor(node_features).unsqueeze(1)
            target_nodes = group.session_item_id.values[1:]
            source_nodes = group.session_item_id.values[:-1]

            edge_index = torch.tensor([source_nodes,target_nodes],dtype=torch.long)

            x = node_features
            y = torch.FloatTensor([group.label.values[0]])

            data = Data(x=x,edge_index=edge_index,y=y)
            data_list.append(data)
        data, slices = self.collate(data_list)
        torch.save((data,slices), self.processed_paths[0] )
     

