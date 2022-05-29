import os
import torch
import tqdm
import torch.nn as nn
from .layers import *
from sklearn.metrics import *
from transformers import BertModel
from utils.utils import data2gpu, Averager, metrics, Recorder
from utils.dataloader import get_dataloader

class BERTEmoModel(torch.nn.Module):
    def __init__(self, emb_dim, mlp_dims, dropout):
        super(BERTEmoModel, self).__init__()
        self.bert = BertModel.from_pretrained('hfl/chinese-bert-wwm-ext').requires_grad_(False)

        for name, param in self.bert.named_parameters():
            #param.requires_grad = True
            if name.startswith("encoder.layer.11"): \
                    # or name.startswith('encoder.layer.10') \
                    # or name.startswith('encoder.layer.9'): \
                    # or name.startswith('encoder.layer.8') \
                    # or name.startswith('encoder.layer.7') \
                    # or name.startswith('encoder.layer.6')\
                    # or name.startswith('encoder.layer.5') \
                    # or name.startswith('encoder.layer.4')\
                    # or name.startswith('encoder.layer.3'):
                param.requires_grad = True
            else:
                param.requires_grad = False

        self.fea_size = emb_dim
        self.mlp = MLP(emb_dim * 2 + 47, mlp_dims, dropout)
        self.rnn = nn.GRU(input_size = emb_dim,
                          hidden_size = self.fea_size,
                          num_layers = 1, 
                          batch_first = True, 
                          bidirectional = True)
        self.attention = MaskAttention(emb_dim * 2)
    
    def forward(self, **kwargs):
        inputs = kwargs['content']
        masks = kwargs['content_masks']
        emotion = kwargs['emotion']
        bert_feature = self.bert(inputs, attention_mask = masks)[0]
        feature, _ = self.rnn(bert_feature)
        feature, _ = self.attention(feature, masks)
        output = self.mlp(torch.cat([feature, emotion], dim = 1))
        return torch.sigmoid(output.squeeze(1))


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
        if(logger):
            logger.info('start training......')
        self.model = BERTEmoModel(self.config['emb_dim'], self.config['model']['mlp']['dims'], self.config['model']['mlp']['dropout'])
        if self.config['use_cuda']:
            self.model = self.model.cuda()
        loss_fn = torch.nn.BCELoss()
        optimizer = torch.optim.Adam(params=self.model.parameters(), lr=self.config['lr'], weight_decay=self.config['weight_decay'])
        recorder = Recorder(self.config['early_stop'])
        val_loader = get_dataloader(self.config['root_path'] + 'val.json', self.config['max_len'], self.config['batchsize'], shuffle=False, use_endef=False, aug_prob=self.config['aug_prob'])

        for epoch in range(self.config['epoch']):
            self.model.train()
            train_loader = get_dataloader(self.config['root_path'] + 'train.json', self.config['max_len'], self.config['batchsize'], shuffle=True, use_endef=False, aug_prob=self.config['aug_prob'])
            train_data_iter = tqdm.tqdm(train_loader)
            avg_loss = Averager()

            for step_n, batch in enumerate(train_data_iter):
                batch_data = data2gpu(batch, self.config['use_cuda'])
                label = batch_data['label']

                pred = self.model(**batch_data)
                loss = loss_fn(pred, label.float())
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
                avg_loss.add(loss.item())
            print('Training Epoch {}; Loss {}; '.format(epoch + 1, avg_loss.item()))

            results = self.test(val_loader)
            mark = recorder.add(results)
            if mark == 'save':
                torch.save(self.model.state_dict(),
                    os.path.join(self.save_path, 'parameter_bertemo.pkl'))
            elif mark == 'esc':
                break
            else:
                continue
        self.model.load_state_dict(torch.load(os.path.join(self.save_path, 'parameter_bertemo.pkl')))

        test_future_loader = get_dataloader(self.config['root_path'] + 'test.json', self.config['max_len'], self.config['batchsize'], shuffle=False, use_endef=False, aug_prob=self.config['aug_prob'])
        future_results = self.test(test_future_loader)
        if(logger):
            logger.info("start testing......")
            logger.info("test score: {}.".format(future_results))
            logger.info("lr: {}, aug_prob: {}, avg test score: {}.\n\n".format(self.config['lr'], self.config['aug_prob'], future_results['metric']))
        print('test results:', future_results)
        return future_results, os.path.join(self.save_path, 'parameter_bertemo.pkl')

    def test(self, dataloader):
        pred = []
        label = []
        self.model.eval()
        data_iter = tqdm.tqdm(dataloader)
        for step_n, batch in enumerate(data_iter):
            with torch.no_grad():
                batch_data = data2gpu(batch, self.config['use_cuda'])
                batch_label = batch_data['label']
                batch_pred = self.model(**batch_data)
                label.extend(batch_label.detach().cpu().numpy().tolist())
                pred.extend(batch_pred.detach().cpu().numpy().tolist())
        return metrics(label, pred)