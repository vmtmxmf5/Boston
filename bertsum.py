import torch
import torch.nn as nn
from torch.nn.init import xavier_uniform_
from pytorch_transformers import BertModel, BertConfig
# pip install pytorch-transformers

import argparse
import os
from msvcrt import kbhit

from models.encoder import Classifier, ExtTransformerEncoder
# from models.optimizers import Optimizer


# class Bert(nn.Module):
#     def __init__(self, large, temp_dir, fine_tune=False):
#         super(Bert, self).__init__()
#         if (large):
#             self.model = BertModel.from_pretrained('bert-large-uncased', cache_dir=temp_dir)
#         else:
#             self.model = BertModel.from_pretrained('bert-base-uncased', cache_dir=temp_dir)
#         self.finetune = fine_tune
    
#     def forward(self, x, segs, mask):
#         if (self.finetune):
#             top_vec, _ = self.model(x, segs, attention_mask=mask)
#         else:
#             self.eval()
#             with torch.no_grad():
#                 top_vec, _ = self.model(x, segs, attention_mask=mask)
#         return top_vec

class Bert(nn.Module):
    def __init__(self, temp_dir, load_pretrained_bert, bert_config):
        super(Bert, self).__init__()
        if (load_pretrained_bert):
            self.model = BertModel.from_pretrained('bert-base-uncased', cache_dir=temp_dir)
        else:
            self.model = BertModel(bert_config)
    
    def forward(self, x, segs, mask):
        encoded_layers, _ = self.model(x, segs, attention_mask=mask)
        top_vec = encoded_layers[-1]
        return top_vec


class ExtSummarizer(nn.Module):
    def __init__(self, args, device):#, checkpoint):
        super(ExtSummarizer, self).__init__()
        self.args = args
        self.device = device
        self.optimizers = None
        config = BertConfig.from_json_file(args.bert_config_path)
        ## TODO True 대신 args로 변경
        self.bert = Bert(args.temp_dir, load_pretrained_bert=False, bert_config=config)

        self.ext_layer = ExtTransformerEncoder(self.bert.model.config.hidden_size,
                                               args.ext_ff_size,
                                               args.ext_heads,
                                               args.ext_dropout,
                                               args.ext_layers)
        
        if (args.max_pos > 512): # 마지막 pos emb 반복
            my_pos_embeddings = nn.Embedding(args.max_pos, self.bert.model.config.hidden_size)
            my_pos_embeddings.weight.data[:512] = self.bert.model.embeddings.position_embeddings.weight.data
            my_pos_embeddings.weight.data[512:] = self.bert.model.embeddings.position_embeddings.weight.data[-1][None, :].repeat(args.max_pos-512, 1)
            self.bert.model.embeddings.position_embeddings = my_pos_embeddings
        
        # if checkpoint is not None:
            # self.load_state_dict(checkpoint['model'], strict=True)
        else:
            if args.param_init != 0.0:
                for p in self.ext_layer.parameters():
                    p.data.uniform_(-args.param_init, args.param_init)
            if args.param_init_glorot:
                for p in self.ext_layer.parameters():
                    if p.dim() > 1:
                        xavier_uniform_(p)
        self.to(device)
    
    def forward(self, src, segs, clss, mask_src, mask_cls):
        top_vec = self.bert(src, segs, mask_src)
        sents_vec = top_vec[torch.arange(top_vec.size(0)).unsqueeze(1), clss]
        sents_vec = sents_vec * mask_cls[:, :, None].float()

        sents_scores = self.ext_layer(sents_vec, mask_cls).squeeze(-1)
        return sents_scores, mask_cls
    

if __name__=='__main__':
    parser = argparse.ArgumentParser()
    # parser.add_argument("-task", default='ext', type=str, choices=['ext', 'abs'])
    # parser.add_argument("-encoder", default='bert', type=str, choices=['bert', 'baseline'])
    # parser.add_argument("-mode", default='train', type=str, choices=['train', 'validate', 'test'])
    # parser.add_argument("-bert_data_path", default='../bert_data_new/cnndm')
    parser.add_argument("-model_path", default='./models')
    parser.add_argument("-result_path", default='./results')
    parser.add_argument("-temp_dir", default='./temp')
    parser.add_argument("-bert_config_path", default='./bert_config_uncased_base.json')

    parser.add_argument("-batch_size", default=140, type=int)
    parser.add_argument("-test_batch_size", default=200, type=int)

    parser.add_argument("-max_pos", default=512, type=int)
    # parser.add_argument("-use_interval", type=bool, nargs='?',const=True, default=True)
    # parser.add_argument("-large", type=bool, default=False)
    # parser.add_argument("-load_from_extractive", default='', type=str)

    parser.add_argument("-sep_optim", type=bool, nargs='?',const=True,default=False)
    parser.add_argument("-lr_bert", default=2e-3, type=float)
    # parser.add_argument("-lr_dec", default=2e-3, type=float)
    parser.add_argument("-use_bert_emb", type=bool, nargs='?',const=True,default=False)

    # parser.add_argument("-share_emb", type=bool, nargs='?', const=True, default=False)
    # parser.add_argument("-finetune_bert", type=str, default=True)
    # parser.add_argument("-dec_dropout", default=0.2, type=float)
    # parser.add_argument("-dec_layers", default=6, type=int)
    # parser.add_argument("-dec_hidden_size", default=768, type=int)
    # parser.add_argument("-dec_heads", default=8, type=int)
    # parser.add_argument("-dec_ff_size", default=2048, type=int)
    # parser.add_argument("-enc_hidden_size", default=512, type=int)
    # parser.add_argument("-enc_ff_size", default=512, type=int)
    # parser.add_argument("-enc_dropout", default=0.2, type=float)
    # parser.add_argument("-enc_layers", default=6, type=int)

    # params for EXT
    parser.add_argument("-ext_dropout", default=0.2, type=float)
    parser.add_argument("-ext_layers", default=2, type=int)
    parser.add_argument("-ext_hidden_size", default=768, type=int)
    parser.add_argument("-ext_heads", default=8, type=int)
    parser.add_argument("-ext_ff_size", default=2048, type=int)

    # parser.add_argument("-label_smoothing", default=0.1, type=float)
    # parser.add_argument("-generator_shard_size", default=32, type=int)
    # parser.add_argument("-alpha",  default=0.6, type=float)
    # parser.add_argument("-beam_size", default=5, type=int)
    # parser.add_argument("-min_length", default=15, type=int)
    # parser.add_argument("-max_length", default=150, type=int)
    # parser.add_argument("-max_tgt_len", default=140, type=int)



    parser.add_argument("-param_init", default=0, type=float)
    parser.add_argument("-param_init_glorot", type=bool, default=True)
    parser.add_argument("-optim", default='adam', type=str)
    parser.add_argument("-lr", default=2e-3, type=float)
    parser.add_argument("-beta1", default= 0.9, type=float)
    parser.add_argument("-beta2", default=0.999, type=float)
    parser.add_argument("-warmup_steps", default=10000, type=int)
    parser.add_argument("-warmup_steps_bert", default=8000, type=int)
    # parser.add_argument("-warmup_steps_dec", default=8000, type=int)
    parser.add_argument("-max_grad_norm", default=0, type=float)

    # parser.add_argument("-save_checkpoint_steps", default=1000, type=int)
    # parser.add_argument("-accum_count", default=2, type=int)
    # parser.add_argument("-report_every", default=1, type=int)
    # parser.add_argument("-train_steps", default=1000, type=int)
    # parser.add_argument("-recall_eval", type=bool, nargs='?',const=True,default=False)

    parser.add_argument("-cuda", default=False, type=bool)
    # parser.add_argument('-visible_gpus', default='-1', type=str)
    # parser.add_argument('-gpu_ranks', default='0', type=str)
    # parser.add_argument('-log_file', default='../logs/cnndm.log')
    # parser.add_argument('-seed', default=666, type=int)

    # parser.add_argument("-test_all", type=bool, nargs='?',const=True,default=False)
    # parser.add_argument("-test_from", default='')
    # parser.add_argument("-test_start_from", default=-1, type=int)

    # parser.add_argument("-train_from", default='')
    # parser.add_argument("-report_rouge", type=bool, nargs='?',const=True,default=True)
    # parser.add_argument("-block_trigram", type=bool, nargs='?', const=True, default=True)

    args = parser.parse_args()
    # args.gpu_ranks = [int(i) for i in range(len(args.visible_gpus.split(',')))]
    # args.world_size = len(args.gpu_ranks)
    # os.environ["CUDA_VISIBLE_DEVICES"] = args.visible_gpus

    device = "cpu" if not args.cuda else "cuda"
    
    models = ExtSummarizer(args, device)#, checkpoint)
    optimizer = torch.optim.Adam(models.parameters(), lr=args.lr, betas=[args.beta1, args.beta2], eps=1e-9)
    checkpoint = torch.load('bertsum.pt', map_location=torch.device('cpu'))
    
    # build model
    model_dict = models.state_dict()
    model_dict.update(checkpoint['model'])
    models.load_state_dict(model_dict)

    ## build original optimizer
    # saved_optim = checkpoint['optim'].optimizer.state_dict()
    # optimizer.load_state_dict(saved_optim)

    optimizer.load_state_dict(checkpoint['optim'])

    
    

    # torch.save({'model':models.state_dict(),
    #             'optim':optimizer.state_dict(),
    #             'opt':args}, 'bertsum.pt')