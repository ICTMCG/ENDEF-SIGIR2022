import logging
import os
import json

from models.bigru import Trainer as BiGRUTrainer
from models.bert import Trainer as BertTrainer
from models.eann import Trainer as EANNTrainer
from models.mdfend import Trainer as MDFENDTrainer
from models.bertemo import Trainer as BertEmoTrainer
from models.bigruendef import Trainer as BiGRU_ENDEFTrainer
from models.bertendef import Trainer as BERT_ENDEFTrainer
from models.bertemoendef import Trainer as BERTEmo_ENDEFTrainer
from models.eannendef import Trainer as EANN_ENDEFTrainer
from models.mdfendendef import Trainer as MDFEND_ENDEFTrainer


def frange(x, y, jump):
  while x < y:
      x = round(x, 8)
      yield x
      x += jump
class Run():
    def __init__(self,
                 config
                 ):
        self.config = config
    

    def getFileLogger(self, log_file):
        logger = logging.getLogger()
        logger.setLevel(level = logging.INFO)
        handler = logging.FileHandler(log_file)
        handler.setLevel(logging.INFO)
        formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
        handler.setFormatter(formatter)
        logger.addHandler(handler)
        return logger
    
    def config2dict(self):
        config_dict = {}
        for k, v in self.configinfo.items():
            config_dict[k] = v
        return config_dict

    def main(self):
        param_log_dir = self.config['param_log_dir']
        if not os.path.exists(param_log_dir):
            os.makedirs(param_log_dir)
        param_log_file = os.path.join(param_log_dir, self.config['model_name'] +'_'+ 'param.txt')
        logger = self.getFileLogger(param_log_file)  
        
        train_param = {
            'lr': [self.config['lr']] * 10,
        }
        print(train_param)
        param = train_param
        best_param = []
        json_path = './logs/json/' + self.config['model_name'] + str(self.config['aug_prob']) + '.json'
        json_result = []
        for p, vs in param.items():
            best_metric = {}
            best_metric['metric'] = 0
            best_v = vs[0]
            best_model_path = None
            for i, v in enumerate(vs):
                self.config['lr'] = v
                if self.config['model_name'] == 'eann':
                    trainer = EANNTrainer(self.config)
                elif self.config['model_name'] == 'bertemo':
                    trainer = BertEmoTrainer(self.config)
                elif self.config['model_name'] == 'bigru':
                    trainer = BiGRUTrainer(self.config)
                elif self.config['model_name'] == 'mdfend':
                    trainer = MDFENDTrainer(self.config)
                elif self.config['model_name'] == 'bert':
                    trainer = BertTrainer(self.config)
                elif self.config['model_name'] == 'bigru_endef':
                    trainer = BiGRU_ENDEFTrainer(self.config)
                elif self.config['model_name'] == 'bert_endef':
                    trainer = BERT_ENDEFTrainer(self.config)
                elif self.config['model_name'] == 'bertemo_endef':
                    trainer = BERTEmo_ENDEFTrainer(self.config)
                elif self.config['model_name'] == 'eann_endef':
                    trainer = EANN_ENDEFTrainer(self.config)
                elif self.config['model_name'] == 'mdfend_endef':
                    trainer = MDFEND_ENDEFTrainer(self.config)
                metrics, model_path = trainer.train(logger)
                json_result.append(metrics)
                if metrics['metric'] > best_metric['metric']:
                    best_metric['metric'] = metrics['metric']
                    best_v = v
                    best_model_path = model_path
            best_param.append({p: best_v})
            print("best model path:", best_model_path)
            print("best metric:", best_metric)
            logger.info("best model path:" + best_model_path)
            logger.info("best param " + p + ": " + str(best_v))
            logger.info("best metric:" + str(best_metric))
            logger.info('--------------------------------------\n')
        with open(json_path, 'w') as file:
            json.dump(json_result, file, indent=4, ensure_ascii=False)
