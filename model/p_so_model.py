from os.path import dirname, abspath
import sys
import codecs
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from numpy.random import randint
from torch.utils.data import DataLoader, Dataset
from torch.nn.modules.loss import BCEWithLogitsLoss
from torch.nn.utils.rnn import pad_packed_sequence, pack_padded_sequence
from torch import Tensor
from tqdm import tqdm

sys.path.append('..')
sys.path.append('.')

from utils.logger import Logger
from utils.function import * 
from embedding.torchEmbedding import TorchEmbedding, PositionEmbedding, EmbeddingProcessor
from embedding.AlbertEmbedding import AlbertEmbedding
from embedding.bert_embedding import Bert_embedding
from model.attention import SelfAttention, MultiHeadAttention
from model.Decoder import DGCNNConcatDecoder,DGCNNDecoder
from model.Encoder import RNNEncoder
from model.loss_function import FocalLossBCE, FocalLossCrossEntropy
from model.model_utils import EMA
from config import Config

log = Logger('p_so_model', std_out=False, save2file=True).get_logger()

parent_path = abspath(dirname(dirname(__file__)))
TRAIN_FILE = parent_path + '/data/my_train_data.json'
DEV_FILE = parent_path + '/data/my_dev_data.json'
ID_PREDICATE_FILE = parent_path + '/data/id_and_predicate_no_unk.json'
PREDICATE_INFO_FILE = parent_path + '/data/predicate_info.json'
CHAR2ID_FILE = parent_path + '/data/char2id.json'


class TextData(object):
    '''
    '''
    def __init__(self, json_file: str, id_predicate_file: str=ID_PREDICATE_FILE, char2id_file: str=CHAR2ID_FILE):
        super().__init__()
        raw_data = read_json(json_file)

        self.id2predicate, self.predicate2id = read_json(id_predicate_file)
        self.predicate_info = read_json(PREDICATE_INFO_FILE)

        self.num_predicate = len(self.id2predicate)
        self.max_seq_len = self.__compute_max_len(raw_data)
        self.len = len(raw_data)
        
        self.inputs_outputs = self.__process_data(raw_data)

    def __process_data(self, raw_data: list):
        text = []
        p_labels = []
        so_q_text = []
        s_start = []
        s_end = []
        o_start = []
        o_end = []

        num_predicate = self.num_predicate
        predicate2id = self.predicate2id
        max_seq_len = self.max_seq_len

        for data in raw_data:
            text.append(data['text'])
            spo_list = data['spo_list']
            
            p_label = np.zeros((num_predicate, ), dtype=np.float32)
            subject_start = np.zeros((max_seq_len, ), dtype=np.float32)
            subject_end = np.zeros((max_seq_len, ), dtype=np.float32)

            object_start = np.zeros((max_seq_len, ), dtype=np.float32)
            object_end = np.zeros((max_seq_len, ), dtype=np.float32)

            # 随机采样一个spo_list中的predicate作为so模型的输入
            choice_index = randint(low=0, high=len(spo_list))
            spo_choose = spo_list[choice_index]
            p_choose = spo_choose['predicate']

            # so模型查询输入
            p_info = self.predicate_info[p_choose]
            so_query_text = '{}，{}，{}。{}'.format(p_info['s_type'], p_choose, p_info['o_type'], data['text'])

            for spo in spo_list:
                predicate_id = predicate2id[spo['predicate']]
                p_label[predicate_id] = 1.0

                if spo['predicate'] == p_choose:
                    subject_start[spo['subject_start']] = 1.0
                    subject_end[spo['subject_end'] - 1] = 1.0

                    object_start[spo['object_start']] = 1.0
                    object_end[spo['object_end'] - 1] = 1.0
            
            p_labels.append(p_label)
            so_q_text.append(so_query_text)
            s_start.append(subject_start)
            s_end.append(subject_end)
            o_start.append(object_start)
            o_end.append(object_end)
           

        return (text, p_labels, so_q_text, s_start, s_end, o_start, o_end)

    def __compute_max_len(self, raw_data: list):
        max_seq_len = 0
        for data in raw_data:
            max_seq_len = max(len(data['text']), max_seq_len)

        return max_seq_len
    
    def __len__(self):
        return self.len

class SpoDataset(Dataset):
    def __init__(self, inputs_outputs: list, predicate_info: dict, predicate2id: dict, max_seq_len: int):
        super(SpoDataset, self).__init__()
        text, p_labels, so_q_text, s_start, s_end, o_start, o_end = inputs_outputs

        self.text = text
        self.predicate_labels = p_labels
        self.so_query_text = so_q_text
        self.subject_start = s_start
        self.subject_end = s_end
        self.object_start = o_start
        self.object_end = o_end

        self.predicate2id = predicate2id
        self.max_seq_len = max_seq_len
        self.predicate_info = predicate_info

        self.num_predicate = len(predicate2id)
        self.len = len(text)

    def __getitem__(self, index):
        text = self.text[index]

        return text, self.predicate_labels[index], self.so_query_text[index], \
        self.subject_start[index], self.subject_end[index], self.object_start[index], self.object_end[index], len(text)

    def __len__(self):
        return self.len

def collate_fn(data):
    '''
    '''
    array = np.array
    
    lens = [item[7] for item in data]
    max_len = max(lens)

    text = [item[0] for item in data]
    predicate_label = array([item[1] for item in data])

    so_query_text = [item[2] for item in data]

    subject_start = array([item[3] for item in data])[:, 0: max_len]
    subject_end = array([item[4] for item in data])[:, 0: max_len]
    object_start = array([item[5] for item in data])[:, 0: max_len]
    object_end = array([item[6] for item in data])[:, 0: max_len]

    from_numpy = torch.from_numpy

    ret = {
        'text': text,
        'predicate_label': from_numpy(predicate_label),
        'so_query_text': so_query_text,
        'subject_start': from_numpy(subject_start),
        'subject_end': from_numpy(subject_end),
        'object_start': from_numpy(object_start),
        'object_end': from_numpy(object_end),
    }

    return ret 

class PredicateModel(nn.Module):
    def __init__(self, embedding_size: int, num_predicate: int, num_heads: int, forward_dim: int=128, device: str='cuda', dropout_prob: float=0.1):
        '''
        embedding_size： 词向量的大小
        num_predicate: 模型要预测的关系总数
        '''
        super(PredicateModel, self).__init__()

        self.embedding_processor = EmbeddingProcessor(embedding_dim=embedding_size, dropout_prob=dropout_prob)

        self.embedding_encoder = RNNEncoder(
            embedding_dim=embedding_size,
            num_layers=2,
            rnn_type='gru',
        )

        d = [1, 2, 3, 1, 2, 3]
        k = [3, 3, 3, 5, 5, 5]
        self.cnn = DGCNNDecoder(
            dilations=d,
            kernel_sizes=k,
            in_channels=embedding_size,
        )

        
        self.self_attention = SelfAttention(
            embedding_dim=embedding_size,
            num_heads=num_heads,
            device=device,
        )

        forward_dim_in = embedding_size * 2
        self.predicate_fc = nn.Sequential(
            nn.Linear(in_features=forward_dim_in, out_features=forward_dim),
            nn.GELU(),
            nn.Linear(in_features=forward_dim, out_features=num_predicate),
        )

    def forward(self, input_embedding: Tensor, mask: Tensor, position_embedding: Tensor):
        '''
        '''
        input_embedding = self.embedding_processor(input_embedding, position_embedding)

        share_feature = self.embedding_encoder(
            input_embedding=input_embedding,
            mask=mask,
        )
        share_feature = self.cnn(share_feature, mask)
        #===========================================================================================#

        attention, _ = self.self_attention(
            inputs=share_feature,
            mask=mask,
        )

        att_max_pool = tensor_max_poll(attention, mask)
        att_avg_pool = tensor_avg_poll(attention, mask)

        inputs = torch.cat([att_max_pool, att_avg_pool], dim=1)
        preidcate_pred = self.predicate_fc(inputs)
        
        return share_feature, preidcate_pred


class SubjectObjectModel(nn.Module):
    def __init__(self, embedding_size: int, num_heads, forward_dim: int=128, device: str='cuda', dropout_prob: float=0.1):
        '''
        '''
        super(SubjectObjectModel, self).__init__()

        self.embedding_processor = EmbeddingProcessor(embedding_dim=embedding_size, dropout_prob=dropout_prob)

        self.embedding_encoder = RNNEncoder(
            embedding_dim=embedding_size,
            num_layers=2,
            rnn_type='gru',
        )

        d = [1, 2, 3, 1, 2, 3]
        k = [3, 3, 3, 5, 5, 5]
        self.cnn = DGCNNDecoder(
            dilations=d,
            kernel_sizes=k,
            in_channels=embedding_size,
        )


        self.multi_head_attention = MultiHeadAttention(
            embedding_dim=embedding_size,
            num_heads=num_heads,
            device=device,
        )
         
        forward_dim_in = embedding_size * 1

        self.subject_start_fc = nn.Sequential(
            nn.Linear(in_features=forward_dim_in, out_features=forward_dim),
            nn.GELU(),
            nn.Linear(forward_dim, 1),
        )

        self.subject_end_fc = nn.Sequential(
            nn.Linear(in_features=forward_dim_in, out_features=forward_dim),
            nn.GELU(),
            nn.Linear(forward_dim, 1),
        )

        self.object_start_fc = nn.Sequential(
            nn.Linear(in_features=forward_dim_in, out_features=forward_dim),
            nn.GELU(),
            nn.Linear(forward_dim, 1),
        )

        self.object_end_fc = nn.Sequential(
            nn.Linear(in_features=forward_dim_in, out_features=forward_dim),
            nn.GELU(),
            nn.Linear(forward_dim, 1),
        )

    def forward(self, share_feature: Tensor, share_mask: Tensor, query_embedding: Tensor, query_mask: Tensor, query_pos_embedding: Tensor):
        '''
        '''
        query_embedding = self.embedding_processor(query_embedding, query_pos_embedding)

        rnn_out = self.embedding_encoder(
            input_embedding=query_embedding,
            mask=query_mask,
        )

        rnn_out = self.cnn(rnn_out, query_mask)
        #===========================================================================================#

        out, _ = self.multi_head_attention(
            query=share_feature,
            key=rnn_out,
            value=rnn_out,
            query_mask=share_mask,
            key_mask=query_mask,
        )

        subject_start = self.subject_start_fc(out)
        subject_end = self.subject_end_fc(out)
        object_start = self.object_start_fc(out)
        object_end = self.object_end_fc(out)

        return subject_start, subject_end, object_start, object_end

# ======================================================================= end model ======================================================================#

class Trainer(object):
    def __init__(self):
        super().__init__()

        log.info('加载数据集...')
        self.train_data = TextData(TRAIN_FILE, ID_PREDICATE_FILE)
        self.dev_data = read_json(DEV_FILE)  # 评估数据直接用原始数据

        # 统一使用训练集的长度
        self.max_seq_len = self.train_data.max_seq_len
        self.num_predicate = self.train_data.num_predicate

        self.id2predicate = self.train_data.id2predicate
        self.predicate2id = self.train_data.predicate2id
        self.predicate_info = self.train_data.predicate_info

    def train(self, config: Config, device):

        # 训练数据加载器
        train_data_loader = DataLoader(
            dataset=SpoDataset(
                inputs_outputs=self.train_data.inputs_outputs,
                predicate_info=self.predicate_info,
                predicate2id=self.predicate2id,
                max_seq_len=self.max_seq_len,
            ),
            batch_size=config.batch_size,
            shuffle=True,
            num_workers=0,  # win10下，DataLoader使用lmdb进行多进程加载词向量会报错
            collate_fn=collate_fn, # 不自己写collate_fn处理numpy到tensor的转换会导致数据转换非常慢
            pin_memory=True,
        )

        # 报存模型的路径
        model_path = parent_path + '/model_file'
        pertrain_file = model_path + '/pertrain/merge_sgns_bigram_char300.npy'

        # torch_embedding
        embedding = None
        if config.from_pertrained == 'word2vec':
            config.embedding_size = 300
            embedding = TorchEmbedding(config.embedding_size, device,from_pertrain=pertrain_file).to(device)
        elif config.from_pertrained == 'albert':
            config.embedding_size = 768
            embedding = AlbertEmbedding(device=config.bert_device)
        elif config.from_pertrained == 'bert':
            config.embedding_size = 768
            embedding = Bert_embedding(device=config.bert_device)
        elif config.from_pertrained == 'none':
            embedding = TorchEmbedding(config.embedding_size, device).to(device)
        else:
            raise ValueError('value config.from_pertrained "{}" error.'.format(config.from_pertrained))

        position_embedding = PositionEmbedding(config.embedding_size).to(device)

        p_model = PredicateModel(
            embedding_size=config.embedding_size,
            num_predicate=self.num_predicate,
            num_heads=config.num_heads,
            forward_dim=config.forward_dim,
            device=device,
        ).to(device)

        so_model = SubjectObjectModel(
            embedding_size=config.embedding_size,
            num_heads=config.num_heads,
            forward_dim=config.forward_dim,
            device=device,
        ).to(device)

        p_ema = EMA(model=p_model, decay=0.9999)
        so_ema = EMA(model=so_model, decay=0.9999)

        # focal_bce_loss = FocalLossBCE(alpha=0.3, gamma=1.0, with_logits=True, device=device).to(device)
        bce_loss = BCEWithLogitsLoss(reduction='none').to(device)

        # 网络参数
        params = []
        if config.from_pertrained not in ['bert', 'albert']:
            params.extend(get_models_parameters([embedding], weight_decay=0.0))
        params.extend(get_models_parameters(model_list=[p_model, so_model], weight_decay=0.0))

        # 优化器
        optimizer = torch.optim.Adam(params=params, lr=config.learning_rate)

        steps = int(np.round(self.train_data.len / config.batch_size))
        log.info('epoch: {}, steps: {}'.format(config.epoch, steps))

        best_f1 = 0.0
        best_epoch = 0

        # warm_up_lambda是一个倍数算子
        warm_up_steps = int(steps * config.warm_up_epoch)
        warm_up_lambda = lambda step: (step + 1) / warm_up_steps
        warm_up_schedule = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda=warm_up_lambda)

        lr_scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=config.lr_T_max)
        
        for epoch in range(config.epoch):
            loss_cpu = 0.0
            p_model.train()
            so_model.train()
            loss_sum = 0.0
            log.info('epoch: {}, lr: {:.6f}'.format(epoch, optimizer.state_dict()['param_groups'][0]['lr']))
            print('epoch: {}, start time: {}'.format(epoch, get_formated_time()))

            with tqdm(total=steps) as pbar:
                for step, inputs_outputs in enumerate(train_data_loader):
                    pbar.update(1)
                    pbar.set_description('training epoch {}'.format(epoch))
                    pbar.set_postfix_str('loss: {:0.4f}'.format(loss_cpu))

                    text = inputs_outputs['text']
                    so_query_text = inputs_outputs['so_query_text']
                    predicate_true = inputs_outputs['predicate_label'].to(device)

                    s_start_true = inputs_outputs['subject_start'].to(device)
                    s_end_true = inputs_outputs['subject_end'].to(device)
                    o_start_true = inputs_outputs['object_start'].to(device)
                    o_end_true = inputs_outputs['object_end'].to(device)
                    

                    # 字符转向量
                    if config.from_pertrained in ['bert', 'albert']:
                        input_embedding, cls_embedding, input_length, attention_mask = embedding.text_to_embedding(text)
                        so_query_embedding, cls_embedding, so_query_length, attention_mask = embedding.text_to_embedding(so_query_text)
                    else:    
                        input_embedding, input_length = embedding(text)
                        so_query_embedding, so_query_length = embedding(so_query_text, requires_grad=False)
                    
                    input_pos_embedding = position_embedding(input_length)
                    so_query_pos_embedding = position_embedding(so_query_length)

                    input_mask = create_mask_from_lengths(input_length)
                    query_mask = create_mask_from_lengths(so_query_length)

                    share_feature, predicate_pred = p_model(
                        input_embedding=input_embedding,
                        mask=input_mask,
                        position_embedding=input_pos_embedding,
                    )
                    
                    subject_start, subject_end, object_start, object_end = so_model(
                        share_feature=share_feature,
                        share_mask=input_mask,
                        query_embedding=so_query_embedding,
                        query_mask=query_mask,
                        query_pos_embedding=so_query_pos_embedding,
                    )
                
                    subject_start = torch.squeeze(subject_start, dim=2)
                    subject_end = torch.squeeze(subject_end, dim=2)
                    object_start = torch.squeeze(object_start, dim=2)
                    object_end = torch.squeeze(object_end, dim=2)

                    if config.from_pertrained in ['bert', 'albert']:
                        subject_start = subject_start[:, 1: -1]
                        subject_end = subject_end[:, 1: -1]
                        object_start = object_start[:, 1: -1]
                        object_end = object_end[:, 1: -1]
                        input_length -= 2
                        input_mask = create_mask_from_lengths(input_length)

                    loss_mask = torch.ge(input_mask, 1)
                    # 计算p模型损失，mask掉的损失不做处理
                    p_loss = bce_loss(predicate_pred, predicate_true)
                    p_loss = torch.mean(p_loss)

                    # 计算so模型的损失
                    s_start_loss = torch.masked_select(bce_loss(subject_start, s_start_true), loss_mask)
                    s_end_loss = torch.masked_select(bce_loss(subject_end, s_end_true), loss_mask)
                    s_start_loss = torch.mean(s_start_loss)
                    s_end_loss = torch.mean(s_end_loss)

                    o_start_loss = torch.masked_select(bce_loss(object_start, o_start_true), loss_mask)
                    o_end_loss = torch.masked_select(bce_loss(object_end, o_end_true), loss_mask)
                    o_start_loss = torch.mean(o_start_loss)
                    o_end_loss = torch.mean(o_end_loss)

                    # 计算总的损失
                    loss_sum = p_loss + 2.5 * (s_start_loss + s_end_loss + o_start_loss + o_end_loss)
                    loss_cpu = loss_sum.cpu().detach().numpy()

                    optimizer.zero_grad()
                    loss_sum.backward()
                    nn.utils.clip_grad_norm_(parameters=p_model.parameters(), max_norm=10, norm_type=2.0)
                    nn.utils.clip_grad_norm_(parameters=so_model.parameters(), max_norm=10, norm_type=2.0)
                    optimizer.step()

                    # apply ema 
                    p_ema.update_params()
                    so_ema.update_params()

                    if config.log_loss and (step % 100 == 0 or step == steps - 1) :
                        log.info('epoch: {}; step: {}; loss: {:.5f}; completed: {:.3f}%.'.format(epoch, step, loss_sum.cpu().detach().numpy(), (step / steps) * 100))
                    
                    # warm up
                    if epoch < config.warm_up_epoch:
                        warm_up_schedule.step()

            # 评估数据要设置模型为eval模式
            p_model.eval()
            so_model.eval()

            print('{}, evaluate epoch: {} ...'.format(get_formated_time(), epoch))

            p_ema.apply_shadow()
            so_ema.apply_shadow()

            with torch.no_grad():
                f1, precision, recall, _ = evaluate(
                    models=(p_model, so_model),
                    embeddings=(embedding, position_embedding),
                    predicate_info=self.predicate_info,
                    dev_data=self.dev_data, 
                    config=config,
                    id2predicate=self.id2predicate,
                )

            if f1 >= best_f1:
                best_f1 = f1
                best_epoch = epoch
                if config.from_pertrained not in ['bert', 'albert']:
                    torch.save(embedding.state_dict(), '{}/{}_p_so_embedding.pkl'.format(model_path, config.from_pertrained))
                torch.save(p_model.state_dict(), '{}/{}_p_model.pkl'.format(model_path, config.from_pertrained))
                torch.save(so_model.state_dict(), '{}/{}_so_model.pkl'.format(model_path, config.from_pertrained))
            
            p_ema.restore()
            so_ema.restore()
             
            info = 'epoch: {}, f1: {:.5f}, precision: {:.5f}, recall: {:.5f}, best_f1: {:.5f}, best_epoch: {}'.format(
                    epoch, f1, precision, recall, best_f1, best_epoch)
            print_and_log('=' * 64, log)
            print_and_log(info, log)
            print_and_log('=' * 64, log)
            
            # 调整学习率
            # Note that step should be called after validate()
            lr_scheduler.step()


def evaluate(models: tuple, embeddings: tuple, predicate_info: dict, dev_data: list, config: Config, id2predicate: dict, show_details: bool=False):
    '''
    评估
    '''
    # 最小的评估batch是64
    batch_size = config.batch_size if config.batch_size >= 64 else 64

    spo_list_true = []
    spo_list_pred = []

    spo_list_true = []
    spo_list_pred = []
    batch_text = []
    n_dev_data = len(dev_data)
    

    for index, data in tqdm(enumerate(dev_data), total=n_dev_data):
        # 取出当前文本的真实spo
        current_spo = []
        for spo in data['spo_list']:
            current_spo.append( (spo['subject'], spo['predicate'], spo['object']) )
        spo_list_true.append(current_spo)

        batch_text.append(data['text'])
        if len(batch_text) == batch_size or index == n_dev_data - 1:
            batch_spo = compute_batch_spo(
            models=models,
            embeddings=embeddings,
            text=batch_text,
            predicate_info=predicate_info,
            id2predicate=id2predicate,
            config=config,
        )
            spo_list_pred.extend(batch_spo)
            batch_text = []

    if show_details: 
        def get_spo(spo_list_all: list):
            s_list, so_list, p_list, o_list = [], [], [], []
            for spo_list in spo_list_all:
                current_s, current_so, current_p, current_o = [], [], [], []
                for spo in spo_list:
                    current_s.append((spo[0],))
                    current_p.append((spo[1],))
                    current_o.append((spo[2], ))
                    current_so.append((spo[0], spo[2]))

                s_list.append(current_s)
                p_list.append(current_p)
                o_list.append(current_o)
                so_list.append(current_so)
            return s_list, p_list, o_list, so_list

        def show_f1_p_r(p: list, t: list, name: str):
            f1, precision, recall = f1_p_r_compute(p, t)
            info = '{}, f1: {:.5f}; precision: {:.5f}; recall: {:.5f} '.format(name ,f1, precision, recall)
            print_and_log(info, log)

        s_pred, p_pred, o_pred, so_pred = get_spo(spo_list_pred)
        s_true, p_true, o_true, so_true = get_spo(spo_list_true)
        show_f1_p_r(s_pred, s_true,'subject')
        show_f1_p_r(p_pred, p_true,'preidcate')
        show_f1_p_r(o_pred, o_true,'object')
        show_f1_p_r(so_pred, so_true,'subject and object')

    f1, precision, recall = f1_p_r_compute(spo_list_pred, spo_list_true)

    return f1, precision, recall, (spo_list_pred, spo_list_true)

def compute_batch_spo(models: tuple, embeddings: tuple, text: list, predicate_info: dict, id2predicate: dict, config: Config):
    p_model, so_model = models
    sigmoid_threshold = config.sigmoid_threshold

    share_features, predicate_preds, input_masks = compute_batch_p(
        p_model=p_model,
        text=text,
        embeddings=embeddings,
        config=config,
    )

    batch_size = share_features.shape[0]
    batch_p = []
    batch_share_feature = []
    batch_mask = []
    batch_so_query_text = []
    batch_ids = []

    for bs_id, (share_feature, input_mask, predicate_pred) in enumerate(zip(share_features, input_masks, predicate_preds)):
        text_ = text[bs_id]
        p_pred = np.where(predicate_pred > 0.4)[0]
        for predicate_id in p_pred:
            predicate = id2predicate[str(predicate_id)]
            p_info = predicate_info[predicate]
            so_query_text = '{}，{}，{}。{}'.format(p_info['s_type'], predicate, p_info['o_type'], text_)
            
            batch_ids.append(bs_id)
            batch_p.append(predicate)
            batch_share_feature.append(share_feature)
            batch_mask.append(input_mask)
            batch_so_query_text.append(so_query_text)

    last_start = 0
    n = int(np.ceil(len(batch_so_query_text) / batch_size))
    batch_spo_pred = [[] for _ in range(batch_size)]

    max_seq_len = share_features.shape[1]
    embedding_dim = share_features.shape[2]

    # 对一个batch中抽取到的所有po抽取s
    for _ in range(n):
        end = last_start + batch_size

        bs_share_feature = torch.cat(batch_share_feature[last_start: end], dim=0)
        # 直接cat结果不对，要重新reshape
        bs_share_feature = bs_share_feature.reshape(-1, max_seq_len, embedding_dim)

        bs_input_mask = batch_mask[last_start: end]
        bs_input_mask = torch.cat(bs_input_mask, dim=0)
        bs_input_mask = bs_input_mask.reshape(-1, max_seq_len)

        s_starts, s_ends, o_starts, o_ends = compute_so(
            so_model=so_model,
            embeddings=embeddings,
            share_feature=bs_share_feature,
            input_mask=bs_input_mask, 
            so_query_text=batch_so_query_text[last_start: end],
            config=config,
        )
        
        for bs_id, p, s_start, s_end, o_start, o_end in zip(batch_ids[last_start: end], batch_p[last_start: end], s_starts, s_ends, o_starts, o_ends):
            text_ = text[bs_id]
            subject_list = []
            for i, s_s in enumerate(s_start[0: len(text_)]):
                if s_s >= sigmoid_threshold:
                    for j, s_e in enumerate(s_end[i: len(text_)]):
                        if s_e >= 0.4:
                            subject_list.append(text_[i: i + j + 1])
                            break
            
            object_list = []
            for i, o_s in enumerate(o_start[0: len(text_)]):
                if o_s >= sigmoid_threshold:
                    for j, o_e in enumerate(o_end[i: len(text_)]):
                        if o_e >= 0.4:
                            object_list.append(text_[i: i + j + 1])
                            break
            
            # 一个p对对应一个s、一个o的情况
            if len(subject_list) == 1 and len(object_list) == 1:
                batch_spo_pred[bs_id].append((subject_list[0], p, object_list[0]))

             # 同一个predicate,一个object对应多个subject的情况
            if len(object_list) == 1 and len(subject_list) >= 2:
                o = object_list[0]
                for s in subject_list:
                    batch_spo_pred[bs_id].append((s, p, o))
           
            # 同一个predicate, 一个subjec对应多个object的情况
            if len(subject_list) == 1 and len(object_list) >= 2:
                s = subject_list[0]
                for o in object_list:
                    batch_spo_pred[bs_id].append((s, p, o))
            
            # 同一个predicat，多个subject对应多个object的情况
            if len(subject_list) >=2 and len(object_list) >= 2:
                # print(text)
                # 就近匹配
                min_len = min(len(subject_list), len(object_list))
                for i in range(min_len):
                    batch_spo_pred[bs_id].append((subject_list[i], p, object_list[i]))
            
        # end extract spo
        # 更新last
        last_start += batch_size

    return batch_spo_pred

def compute_batch_p(p_model: PredicateModel, text: list, embeddings: tuple, config: Config):
    '''
    '''
    embedding, position_embedding = embeddings

    if config.from_pertrained in ['bert', 'albert']:
        input_embedding, cls_embedding, input_length, attention_mask = embedding.text_to_embedding(text)
    else:    
        input_embedding, input_length = embedding(text)
    
    input_pos_embedding = position_embedding(input_length)
    input_mask = create_mask_from_lengths(input_length)

    share_feature, predicate_pred = p_model(
        input_embedding=input_embedding,
        mask=input_mask,
        position_embedding=input_pos_embedding,
    )

    predicate_pred = torch.sigmoid(predicate_pred)
    predicate_pred = predicate_pred.cpu().detach().numpy()
    
    return share_feature, predicate_pred, input_mask

def compute_so(so_model: SubjectObjectModel, embeddings: tuple, share_feature: Tensor, input_mask: Tensor, so_query_text: list, config: Config):
    '''
    '''
    embedding, position_embedding = embeddings
    squeeze = torch.squeeze
    sigmoid = torch.sigmoid

    if config.from_pertrained in ['bert', 'albert']:
        so_query_embedding, cls_embedding, so_query_length, attention_mask = embedding.text_to_embedding(so_query_text)
    else:    
        so_query_embedding, so_query_length = embedding(so_query_text)
    so_query_pos_embedding = position_embedding(so_query_length)
    query_mask = create_mask_from_lengths(so_query_length)

    subject_start, subject_end, object_start, object_end = so_model(
        share_feature=share_feature,
        share_mask=input_mask,
        query_embedding=so_query_embedding,
        query_mask=query_mask,
        query_pos_embedding=so_query_pos_embedding,
    )
    
    subject_start = squeeze(subject_start, dim=2)
    subject_end = squeeze(subject_end, dim=2)
    object_start = squeeze(object_start, dim=2)
    object_end = squeeze(object_end, dim=2)

    if config.from_pertrained in ['bert', 'albert']:
        subject_start = subject_start[:, 1: - 1]
        subject_end = subject_end[:, 1: - 1]
        object_start = object_start[:, 1: - 1]
        object_end = object_end[:, 1: - 1]
    
    
    subject_start = sigmoid(subject_start).cpu().detach().numpy()
    subject_end = sigmoid(subject_end).cpu().detach().numpy()
    object_start = sigmoid(object_start).cpu().detach().numpy()
    object_end = sigmoid(object_end).cpu().detach().numpy()

    return subject_start, subject_end, object_start, object_end


def load_model_and_evalute(config: Config, device, best_f1: float=0.0):
    base_path = parent_path + '/model_file'
    dev_data = read_json(DEV_FILE)
    id2predicate, predicate2id = read_json(ID_PREDICATE_FILE)
    predicate_info = read_json(PREDICATE_INFO_FILE)

    num_predicate = len(id2predicate)

    # torch_embedding
    embedding = TorchEmbedding(config.embedding_size, device).to(device)
    position_embedding = PositionEmbedding(config.embedding_size).to(device)

    p_model = PredicateModel(
            embedding_size=config.embedding_size,
            num_predicate=num_predicate,
            num_heads=config.num_heads,
            forward_dim=config.forward_dim,
            device=device,
        ).to(device)

    so_model = SubjectObjectModel(
        embedding_size=config.embedding_size,
        num_heads=config.num_heads,
        forward_dim=config.forward_dim,
        device=device,
    ).to(device)

    p_model.eval()
    so_model.eval()

    embedding.load_state_dict(torch.load('{}/{}_p_so_embedding.pkl'.format(base_path, config.from_pertrained), map_location=device))
    p_model.load_state_dict(torch.load('{}/{}_p_model.pkl'.format(base_path, config.from_pertrained), map_location=device))
    so_model.load_state_dict(torch.load('{}/{}_so_model.pkl'.format(base_path, config.from_pertrained), map_location=device))

    with torch.no_grad():
        f1, precision, recall, (spo_list_pred, spo_list_true) = evaluate(
            models=(p_model, so_model),
            embeddings=(embedding, position_embedding),
            predicate_info=predicate_info,
            dev_data=dev_data, 
            config=config,
            id2predicate=id2predicate,
            show_details=True,
        )
    save_spo_list(dev_data, spo_list_pred, spo_list_true, parent_path + '/data/spo_list_pred.json')
    
    print('f1: {:.5f}; precision: {:.5f}; recall: {:.5f}'.format(f1, precision, recall))