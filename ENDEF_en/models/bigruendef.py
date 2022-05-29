import os
import torch
import tqdm
import torch.nn as nn
from .layers import *
from sklearn.metrics import *
from transformers import BertModel
from utils.utils import data2gpu, Averager, metrics, Recorder
from utils.dataloader import get_dataloader

class BiGRU_ENDEFModel(torch.nn.Module):
    def __init__(self, emb_dim, mlp_dims, dropout, num_layers):
        super(BiGRU_ENDEFModel, self).__init__()
        self.fea_size = emb_dim
        self.bert = BertModel.from_pretrained('bert-base-uncased').requires_grad_(False)
        self.embedding = self.bert.embeddings
        
        self.rnn = nn.GRU(input_size = emb_dim,
                          hidden_size = self.fea_size,
                          num_layers = num_layers, 
                          batch_first = True, 
                          bidirectional = True)

        input_shape = self.fea_size * 2
        self.attention = MaskAttention(input_shape)
        self.mlp = MLP(input_shape, mlp_dims, dropout)

        feature_kernel = {1: 64, 2: 64, 3: 64, 5: 64, 10: 64}
        mlp_input_shape = sum([feature_kernel[kernel] for kernel in feature_kernel])
        self.entity_convs = cnn_extractor(feature_kernel, emb_dim)
        self.entity_mlp = MLP(mlp_input_shape, mlp_dims, dropout)
        self.entity_net = torch.nn.Sequential(self.entity_convs, self.entity_mlp)
    
    def forward(self, **kwargs):
        inputs = kwargs['content']
        masks = kwargs['content_masks']
        emb_feature = self.embedding(inputs)
        feature, _ = self.rnn(emb_feature)
        feature, _ = self.attention(feature, masks)
        bias_pred = self.mlp(feature).squeeze(1)

        entity = kwargs['entity']
        entity_feature = self.embedding(entity)
        entity_prob = self.entity_net(entity_feature).squeeze(1)
        return torch.sigmoid(0.9 * bias_pred + 0.1 * entity_prob), torch.sigmoid(entity_prob), torch.sigmoid(bias_pred)


class Trainer():
    def __init__(self,
                 config
                 ):
        self.config = config
        
        self.save_path = os.path.join(self.config['save_param_dir'], self.config['model_name'])
        if os.path.exists(self.save_path):
            self.save_param_dir = self.save_path
        else:
            self.save_param_dir = os.makedirs(self.save_path)
        

    def train(self, logger = None):
        print('lr:', self.config['lr'])
        if(logger):
            logger.info('start training......')
        self.model = BiGRU_ENDEFModel(self.config['emb_dim'], self.config['model']['mlp']['dims'], self.config['model']['mlp']['dropout'], num_layers=1)
        if self.config['use_cuda']:
            self.model = self.model.cuda()
        loss_fn = torch.nn.BCELoss()
        optimizer = torch.optim.Adam(params=self.model.parameters(), lr=self.config['lr'], weight_decay=self.config['weight_decay'])
        recorder = Recorder(self.config['early_stop'])
        val_loader = get_dataloader(self.config['root_path'] + 'val.json', self.config['max_len'], self.config['batchsize'], shuffle=False, use_endef=True, aug_prob=self.config['aug_prob'])

        for epoch in range(self.config['epoch']):
            self.model.train()
            train_loader = get_dataloader(self.config['root_path'] + 'train.json', self.config['max_len'], self.config['batchsize'], shuffle=True, use_endef=True, aug_prob=self.config['aug_prob'])
            train_data_iter = tqdm.tqdm(train_loader)
            avg_loss = Averager()

            for step_n, batch in enumerate(train_data_iter):
                batch_data = data2gpu(batch, self.config['use_cuda'])
                label = batch_data['label']

                pred, entity_pred, _ = self.model(**batch_data)
                loss = loss_fn(pred, label.float()) + 0.2 * loss_fn(entity_pred, label.float())
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
                avg_loss.add(loss.item())
            print('Training Epoch {}; Loss {}; '.format(epoch + 1, avg_loss.item()))

            results = self.test(val_loader)
            mark = recorder.add(results)
            if mark == 'save':
                torch.save(self.model.state_dict(),
                    os.path.join(self.save_path, 'parameter_bigruendef.pkl'))
            elif mark == 'esc':
                break
            else:
                continue
        self.model.load_state_dict(torch.load(os.path.join(self.save_path, 'parameter_bigruendef.pkl')))

        test_future_loader = get_dataloader(self.config['root_path'] + 'test.json', self.config['max_len'], self.config['batchsize'], shuffle=False, use_endef=True, aug_prob=self.config['aug_prob'])
        future_results = self.test(test_future_loader)
        if(logger):
            logger.info("start testing......")
            logger.info("test score: {}.".format(future_results))
            logger.info("lr: {}, aug_prob: {}, avg test score: {}.\n\n".format(self.config['lr'], self.config['aug_prob'], future_results['metric']))
        print('future results:', future_results)
        return future_results, os.path.join(self.save_path, 'parameter_bigruendef.pkl')

    def test(self, dataloader):
        pred = []
        label = []
        self.model.eval()
        data_iter = tqdm.tqdm(dataloader)
        for step_n, batch in enumerate(data_iter):
            with torch.no_grad():
                batch_data = data2gpu(batch, self.config['use_cuda'])
                batch_label = batch_data['label']
                _, __, batch_pred = self.model(**batch_data)

                label.extend(batch_label.detach().cpu().numpy().tolist())
                pred.extend(batch_pred.detach().cpu().numpy().tolist())

        return metrics(label, pred)