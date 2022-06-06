# -*- coding: utf-8 -*-
import argparse
import logging

import numpy as np
import pandas as pd
from pytorch_lightning import Trainer
from pytorch_lightning.callbacks import ModelCheckpoint
from pytorch_lightning.core.lightning import LightningModule
from torch.utils.data import DataLoader, Dataset
from transformers.optimization import AdamW, get_cosine_schedule_with_warmup
from transformers import PreTrainedTokenizerFast, GPT2LMHeadModel, AutoModel, BertModel
#import tensorflow as tf

import torch
from torch import nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import gluonnlp as nlp
from tqdm import tqdm, tqdm_notebook

#transformers
from transformers import BertModel

import konlpy
import gluonnlp as nlp
from kobert import get_pytorch_kobert_model
from kobert_tokenizer import KoBERTTokenizer
from kobert.utils import get_tokenizer

parser = argparse.ArgumentParser(description='Simsimi based on KoGPT-2')

parser.add_argument('--chat',
                    action='store_true',
                    default=False,
                    help='response generation on given user input')

parser.add_argument('--sentiment',
                    type=str,
                    default='0',
                    help='sentiment for system. 0 is neutral, 1 is negative, 2 is positive.')

parser.add_argument('--model_params',
                    type=str,
                    default='model/model_--last.ckpt',
                    help='model binary for starting chat')

parser.add_argument('--train',
                    action='store_true',
                    default=False,
                    help='for training')

logger = logging.getLogger()
logger.setLevel(logging.INFO)

U_TKN = '<usr>'
S_TKN = '<sys>'
BOS = '</s>'
EOS = '</s>'
MASK = '<unused0>'
SENT = '<unused1>'
PAD = '<pad>'

TOKENIZER = PreTrainedTokenizerFast.from_pretrained("skt/kogpt2-base-v2",
            bos_token=BOS, eos_token=EOS, unk_token='<unk>',
            pad_token=PAD, mask_token=MASK)

class CharDataset(Dataset):
    def __init__(self, chats, max_len=32):
        self._data = chats
        self.first = True
        self.q_token = U_TKN
        self.a_token = S_TKN
        self.sent_token = SENT
        self.bos = BOS
        self.eos = EOS
        self.mask = MASK
        self.pad = PAD
        self.max_len = max_len
        self.tokenizer = TOKENIZER 

    def __len__(self):
        return len(self._data)

    def __getitem__(self, idx):
        turn = self._data.iloc[idx]
        q = turn['Q']
        a = turn['A']
        sentiment = str(turn['label'])
        q_toked = self.tokenizer.tokenize(self.q_token + q + \
                                          self.sent_token + sentiment)   
        q_len = len(q_toked)
        a_toked = self.tokenizer.tokenize(self.a_token + a + self.eos)
        a_len = len(a_toked)
        if q_len + a_len > self.max_len:
            a_len = self.max_len - q_len
            if a_len <= 0:
                q_toked = q_toked[-(int(self.max_len/2)):]
                q_len = len(q_toked)
                a_len = self.max_len - q_len
                assert a_len > 0
            a_toked = a_toked[:a_len]
            a_len = len(a_toked)
            assert a_len == len(a_toked), f'{a_len} ==? {len(a_toked)}'
        # [mask, mask, ...., mask, ..., <bos>,..A.. <eos>, <pad>....]
        labels = [
            self.mask,
        ] * q_len + a_toked[1:]
        if self.first:
            logging.info("contexts : {}".format(q))
            logging.info("toked ctx: {}".format(q_toked))
            logging.info("response : {}".format(a))
            logging.info("toked response : {}".format(a_toked))
            logging.info('labels {}'.format(labels))
            self.first = False
        mask = [0] * q_len + [1] * a_len + [0] * (self.max_len - q_len - a_len)
        self.max_len
        labels_ids = self.tokenizer.convert_tokens_to_ids(labels)
        while len(labels_ids) < self.max_len:
            labels_ids += [self.tokenizer.pad_token_id]
        token_ids = self.tokenizer.convert_tokens_to_ids(q_toked + a_toked)
        while len(token_ids) < self.max_len:
            token_ids += [self.tokenizer.pad_token_id]
        return(token_ids, np.array(mask),
               labels_ids)

bertmodel, vocab = get_pytorch_kobert_model()

class BERTDataset(Dataset):
    def __init__(self, dataset, sent_idx, label_idx, bert_tokenizer, max_len,
                 pad, pair):
        transform = nlp.data.BERTSentenceTransform(
            bert_tokenizer, max_seq_length=max_len, pad=pad, pair=pair)

        self.sentences = [transform([i[sent_idx]]) for i in dataset]
        self.labels = [np.int32(i[label_idx]) for i in dataset]

    def __getitem__(self, i):
        return (self.sentences[i] + (self.labels[i], ))

    def __len__(self):
        return (len(self.labels))

class BERTClassifier(nn.Module):
    def __init__(self,
                 bert,
                 hidden_size = 768,
                 num_classes=7,   ##클래스 수 조정##
                 dr_rate=None,
                 params=None):
        super(BERTClassifier, self).__init__()
        self.bert = bert
        self.dr_rate = dr_rate
                 
        self.classifier = nn.Linear(hidden_size , num_classes)
        if dr_rate:
            self.dropout = nn.Dropout(p=dr_rate)
    
    def gen_attention_mask(self, token_ids, valid_length):
        attention_mask = torch.zeros_like(token_ids)
        for i, v in enumerate(valid_length):
            attention_mask[i][:v] = 1
        return attention_mask.float()

    def forward(self, token_ids, valid_length, segment_ids):
        attention_mask = self.gen_attention_mask(token_ids, valid_length)
        
        _, pooler = self.bert(input_ids = token_ids, token_type_ids = segment_ids.long(), attention_mask = attention_mask.float().to(token_ids.device))
        if self.dr_rate:
            out = self.dropout(pooler)
        return self.classifier(out)


class KoGPT2Chat(LightningModule):
    def __init__(self, hparams, **kwargs):
        super(KoGPT2Chat, self).__init__()
        self.hparams = hparams
        self.neg = -1e18
        self.kogpt2 = GPT2LMHeadModel.from_pretrained('skt/kogpt2-base-v2')
        self.loss_function = torch.nn.CrossEntropyLoss(reduction='none')

    @staticmethod
    def add_model_specific_args(parent_parser):
        # add model specific args
        parser = argparse.ArgumentParser(parents=[parent_parser], add_help=False)
        parser.add_argument('--max-len',
                            type=int,
                            default=32,
                            help='max sentence length on input (default: 32)')

        parser.add_argument('--batch-size',
                            type=int,
                            default=96,
                            help='batch size for training (default: 96)')
        parser.add_argument('--lr',
                            type=float,
                            default=5e-5,
                            help='The initial learning rate')
        parser.add_argument('--warmup_ratio',
                            type=float,
                            default=0.1,
                            help='warmup ratio')
        return parser

    def forward(self, inputs):
        # (batch, seq_len, hiddens)
        output = self.kogpt2(inputs, return_dict=True)
        return output.logits

    def training_step(self, batch, batch_idx):
        token_ids, mask, label = batch
        out = self(token_ids)
        mask_3d = mask.unsqueeze(dim=2).repeat_interleave(repeats=out.shape[2], dim=2)
        mask_out = torch.where(mask_3d == 1, out, self.neg * torch.ones_like(out))
        loss = self.loss_function(mask_out.transpose(2, 1), label)
        loss_avg = loss.sum() / mask.sum()
        self.log('train_loss', loss_avg)
        return loss_avg

    def configure_optimizers(self):
        # Prepare optimizer
        param_optimizer = list(self.named_parameters())
        no_decay = ['bias', 'LayerNorm.bias', 'LayerNorm.weight']
        optimizer_grouped_parameters = [
            {'params': [p for n, p in param_optimizer if not any(nd in n for nd in no_decay)], 'weight_decay': 0.01},
            {'params': [p for n, p in param_optimizer if any(nd in n for nd in no_decay)], 'weight_decay': 0.0}
        ]
        optimizer = AdamW(optimizer_grouped_parameters,
                          lr=self.hparams.lr, correct_bias=False)
        # warm up lr
        num_train_steps = len(self.train_dataloader()) * self.hparams.max_epochs
        num_warmup_steps = int(num_train_steps * self.hparams.warmup_ratio)
        scheduler = get_cosine_schedule_with_warmup(
            optimizer,
            num_warmup_steps=num_warmup_steps, num_training_steps=num_train_steps)
        lr_scheduler = {'scheduler': scheduler, 'name': 'cosine_schedule_with_warmup',
                        'monitor': 'loss', 'interval': 'step',
                        'frequency': 1}
        return [optimizer], [lr_scheduler]

    def _collate_fn(self, batch):
        data = [item[0] for item in batch]
        mask = [item[1] for item in batch]
        label = [item[2] for item in batch]
        return torch.LongTensor(data), torch.LongTensor(mask), torch.LongTensor(label)

    def train_dataloader(self):
        data = pd.read_csv('data/ChatbotData.csv')
        self.train_set = CharDataset(data, max_len=self.hparams.max_len)
        train_dataloader = DataLoader(
            self.train_set, batch_size=self.hparams.batch_size, num_workers=2,
            shuffle=True, collate_fn=self._collate_fn)
        return train_dataloader

    def chat(self, sent='0'):
        tok = TOKENIZER
        sent_tokens = tok.tokenize(sent)
        #emo_model = AutoModel.from_pretrained("skt/kobert-base-v1")  # 전체 모델을 통째로 불러옴, 클래스 선언 필수
        checkpoint = torch.load('/content/drive/MyDrive/emotion_model.pth', map_location=torch.device('cpu'))
        emo_model = BERTClassifier(bertmodel,  dr_rate=0.5)
        emo_model.load_state_dict(checkpoint)  # state_dict를 불러 온 후, 모델에 저장
        EMO_TOKENIZER = get_tokenizer()
        emo_tok = nlp.data.BERTSPTokenizer(EMO_TOKENIZER, vocab, lower=False)
        with torch.no_grad():
            while 1:
                q = input('나 > ').strip()
                if q == 'quit':
                    break
                a = ''
                while 1:
                    input_ids = torch.LongTensor(tok.encode(U_TKN + q + SENT + sent + S_TKN + a)).unsqueeze(dim=0)
                    pred = self(input_ids)
                    gen = tok.convert_ids_to_tokens(
                        torch.argmax(
                            pred,
                            dim=-1).squeeze().numpy().tolist())[-1]
                    if gen == EOS:
                        data = [q, '0']
                        dataset_another = [data]
                        another_test = BERTDataset(dataset_another, 0, 1, emo_tok, 64, True, False)
                        test_dataloader = torch.utils.data.DataLoader(another_test, batch_size=64, num_workers=2)
                        for batch_id, (token_ids, valid_length, segment_ids, label) in enumerate(test_dataloader):
                          token_ids = token_ids.long()
                          segment_ids = segment_ids.long()
                          valid_length= valid_length
                        out = emo_model(token_ids, valid_length, segment_ids)
                        for i in out:
                          logits=i
                          logits = logits.detach().cpu().numpy()

                          if np.argmax(logits) == 0: #공포
                              a += "😱"
                          elif np.argmax(logits) == 1: #놀람
                              a += "😲"
                          elif np.argmax(logits) == 2: #분노
                              a += "😓"
                          elif np.argmax(logits) == 3: #슬픔
                              a += "😢"
                          elif np.argmax(logits) == 4: #중립
                              a += "😶"
                          elif np.argmax(logits) == 5: #행복
                              a += "😊"
                          elif np.argmax(logits) == 6: #혐오
                              a += "😫"
                          else:
                              a += "🤨"
                        break
                    a += gen.replace('▁', ' ')
                print("위로봇 > {}".format(a.strip()))


parser = KoGPT2Chat.add_model_specific_args(parser)
parser = Trainer.add_argparse_args(parser)
args = parser.parse_args()
logging.info(args)

if __name__ == "__main__":
    if args.train:
        checkpoint_callback = ModelCheckpoint(
            dirpath='model_chp',
            filename='{epoch:02d}-{train_loss:.2f}',
            verbose=True,
            save_last=True,
            monitor='train_loss',
            mode='min',
            prefix='model_'
        )
        # python train_torch.py --train --gpus 1 --max_epochs 3
        model = KoGPT2Chat(args)
        model.train()
        trainer = Trainer.from_argparse_args(
            args,
            checkpoint_callback=checkpoint_callback, gradient_clip_val=1.0)
        trainer.fit(model)
        logging.info('best model path {}'.format(checkpoint_callback.best_model_path))
    if args.chat:
        model = KoGPT2Chat.load_from_checkpoint(args.model_params)
        model.chat()