""" RNN class wrapper, taking symbols, producing vector representations of prefixes

Sort of a vestigial part of more complex older code, but here in case we'd like
to hand-write RNNs again.
"""
import math
import torch
from tqdm import tqdm
import torch.nn as nn
from gpt import GPT2LMHeadModelCustom
from bert import BertForMaskedLMCustom
from transformers import BertConfig, BertForMaskedLM, GPT2Config, GPT2LMHeadModel
MAX_LEN = 6000


class PytorchTransformerModel(nn.Module):
    """
    Class for mapping sequences of symbols to sequences
    of vectors representing prefixes, using PyTorch
    RNN classes.
    """

    def __init__(self, args):
        super(PytorchTransformerModel, self).__init__()
        num_hidden_layers = args['lm']['num_layers']
        self.input_size = args['lm']['embedding_dim']
        self.hidden_size = args['lm']['hidden_dim']
        self.vocab_size = args['language']['vocab_size']
        self.n_heads = args['lm']['num_heads']
        self.e_type = args['lm']['embedding_type']
        self.token_embedding_type = args['lm']['token_embedding_type']
        self.objective = args['training']['objective']
        self.p_drop = args['training']['dropout']
        self.lm_type = args['lm']['lm_type']
        self.language_depth = args['language']['train_max_stack_depth']
        # Customize
        if("Custom" in args['lm']['lm_type']):
            self.residual = args['lm']['residual']
            self.attn_output_fc = args['lm']['attn_output_fc']
            self.layer_norm = args['lm']['layer_norm']
            self.freeze_decoder_to_I = args['lm']['freeze_decoder_to_I']
            if(args['lm']['lm_type'] == 'BertForMaskedLMCustom'):
                # Bert Customize
                self.bert_intermediate = args['lm']['bert_intermediate']
                self.bert_output = args['lm']['bert_output']
                self.bert_head_transform = args['lm']['bert_head_transform']
            elif(args['lm']['lm_type'] == 'GPT2LMHeadModelCustom'):
                # GPT Customize
                self.gpt_intermediate = args['lm']['gpt_intermediate']

        self.freeze_uniform_attention = args['lm']['freeze_uniform_attention']
        self.device = args['device']
        self.small_initialization = args['lm']['small_initialization']
        self.one_layer_norm = args['lm']['one_layer_norm']
        self.first_residual = args['lm']['first_residual']
        self.another_first_residual = args['lm']['another_first_residual']
        
        if args['lm']['lm_type'] == 'BertForMaskedLM':
            config = BertConfig(
                vocab_size=self.vocab_size,
                hidden_size=self.hidden_size,
                num_hidden_layers=num_hidden_layers,
                num_attention_heads=self.n_heads,
                intermediate_size=self.hidden_size,
                hidden_dropout_prob=self.p_drop,
                attention_probs_dropout_prob=self.p_drop,
                max_position_embeddings=MAX_LEN,
            )
            self.model = BertForMaskedLM(config)
        elif args['lm']['lm_type'] == 'BertForMaskedLMCustom':
            config = BertConfig(
                vocab_size=self.vocab_size,
                hidden_size=self.hidden_size,
                num_hidden_layers=num_hidden_layers,
                num_attention_heads=self.n_heads,
                intermediate_size=self.hidden_size,
                hidden_dropout_prob=self.p_drop,
                attention_probs_dropout_prob=self.p_drop,
                max_position_embeddings=MAX_LEN,
                residual=self.residual,
                attn_output_fc=self.attn_output_fc,
                bert_intermediate=self.bert_intermediate,
                bert_output=self.bert_output,
                bert_head_transform=self.bert_head_transform,
                layer_norm=self.layer_norm,
                freeze_uniform_attention=self.freeze_uniform_attention,
                freeze_decoder_to_I = self.freeze_decoder_to_I,
            )
            self.model = BertForMaskedLMCustom(config)
        elif args['lm']['lm_type'] == 'GPT2LMHeadModel':
            config = GPT2Config(
                n_embd=self.input_size,
                n_layer=num_hidden_layers,
                n_inner=self.hidden_size,
                attn_pdrop=self.p_drop,
                embd_pdrop=self.p_drop,
                resid_pdrop=self.p_drop,
                vocab_size=self.vocab_size,
                n_head=self.n_heads,
                n_positions=MAX_LEN,
                n_ctx=MAX_LEN,
            )
            self.model = GPT2LMHeadModel(config)
        elif args['lm']['lm_type'] == 'GPT2LMHeadModelCustom':
            config = GPT2Config(
                n_embd=self.input_size,
                n_layer=num_hidden_layers,
                n_inner=self.hidden_size,
                attn_pdrop=self.p_drop,
                embd_pdrop=self.p_drop,
                resid_pdrop=self.p_drop,
                vocab_size=self.vocab_size,
                n_head=self.n_heads,
                n_positions=MAX_LEN,
                n_ctx=MAX_LEN,
                residual=self.residual,
                attn_output_fc=self.attn_output_fc,
                gpt_intermediate=self.gpt_intermediate,
                layer_norm=self.layer_norm,
                one_layer_norm = self.one_layer_norm,
                freeze_decoder_to_I = self.freeze_decoder_to_I,
                freeze_uniform_attention = self.freeze_uniform_attention,
                small_initialization = self.small_initialization,
                first_residual = self.first_residual,
                another_first_residual = self.another_first_residual
            )
            self.model = GPT2LMHeadModelCustom(config)
        else:
            raise NotImplementedError('Model not supported.')

        print('config', config)
        print('model', self.model)

        self.model.to(self.device)
        tqdm.write('Constructing a {} pytorch model w hidden size {}, layers {}, dropout {}'.format(
            args['lm']['lm_type'],
            self.hidden_size,
            num_hidden_layers,
            self.p_drop,
        ))

        if args['lm']['lm_type'] in {'BertForMaskedLM', 'BertForMaskedLMCustom'}:
            position_embeddings = self.model.bert.embeddings.position_embeddings
        elif args['lm']['lm_type'] in {'GPT2LMHeadModel', 'GPT2LMHeadModelCustom'}:
            position_embeddings = self.model.transformer.wpe
        else:
            raise NotImplementedError('Model not supported.')
        

        if self.e_type == 'cos':
            funcs = [math.sin, math.cos]
            position_embeddings.weight.data = torch.tensor(
                [[funcs[i % 2](pos / 10000 ** (2 * i / self.hidden_size))
                  for i in range(self.hidden_size)]
                 for pos in range(MAX_LEN)]
            ).to(self.device)
            position_embeddings.weight.requires_grad = False
        if self.e_type in ['p', 'z']:
            position_embeddings.weight.data.zero_()
            position_embeddings.weight.requires_grad = False
            self.embedding = nn.Embedding(self.vocab_size, self.input_size - 1)
            self.embedding.to(self.device)
            self.embedding_p = nn.Embedding(MAX_LEN, 1)
            self.embedding_p.weight.data = torch.tensor([[i / MAX_LEN] for i in range(MAX_LEN)])
            self.embedding_p.weight.requires_grad = False
            self.embedding_p.to(self.device)

        if self.e_type == 'same_trained':
            same_emb = torch.zeros((1, self.input_size))
            position_embeddings.weight.data = same_emb.repeat((MAX_LEN, 1)).to(self.device)
            position_embeddings.weight.requires_grad = True

        if self.e_type == 'none':
            position_embeddings.weight.data.zero_()
            position_embeddings.weight.requires_grad = False

        if self.token_embedding_type in ['one_hot','sub_state','sub_state_easy','sub_state_theory', 'sub_state_large']:
            assert self.input_size - 1>= self.vocab_size, 'each token should be able to have different embedding'
            if(self.e_type in ['p','z']):
                embed_size = self.input_size - 1
            else:
                embed_size = self.input_size
            one_hot_embeddings = torch.zeros((self.vocab_size, embed_size))
            for i in range(self.vocab_size):
                one_hot_embeddings[i][i] = 1.0
            self.embedding = nn.Embedding.from_pretrained(one_hot_embeddings, freeze = True)
            self.embedding.to(self.device)
            self.set_embeddings(one_hot_embeddings, freeze=True)
        
        
    def set_embeddings(self, embeddings, freeze):
        if self.lm_type in {'BertForMaskedLM', 'BertForMaskedLMCustom'}:
            self.model.bert.embeddings.word_embeddings = nn.Embedding.from_pretrained(
                embeddings, freeze=freeze
            ).to(self.device)
        elif self.lm_type in {'GPT2LMHeadModel', 'GPT2LMHeadModelCustom'}:
            self.model.transformer.wte = nn.Embedding.from_pretrained(
                embeddings, freeze=freeze
            ).to(self.device)
        else:
            raise NotImplementedError('Model not supported.')
        
        
    def custom_embed(self, batch):
        vec1 = self.embedding(batch)
        pos = torch.ones(batch.size(), device=self.device).cumsum(-1) - 1
        vec2 = self.embedding_p(pos.long())
        if self.e_type in ['p']:
            vec = torch.cat((vec1, vec2), -1)
        elif self.e_type in ['z']:
            vec2 = torch.zeros(vec2.shape, device = self.device)
            vec = torch.cat((vec1, vec2), -1)
        else:
            vec3 = self.embedding_e(batch)
            vec = torch.cat((vec1, vec2, vec3), -1)
        
        # Currently this kind of tokens only work with p or z input
        if self.token_embedding_type in ['sub_state','sub_state_raw', 'sub_state_easy', 'sub_state_theory', 'sub_state_large']:
            agg = vec.cumsum(1)
            depth = agg[:,:,4] + agg[:,:,5] - agg[:,:,6] - agg[:,:,7] # open - close
            count = torch.ones(depth.size(), device = self.device).cumsum(-1)
            tantheta = depth / (self.language_depth  + 2 - depth)
            costheta = 1 / torch.sqrt(1 + tantheta**2)
            sintheta = tantheta / torch.sqrt(1 + tantheta**2)
            if(self.token_embedding_type == 'sub_state'):
                vec[:,:,-2] = costheta
                vec[:,:,-3] = sintheta
            if(self.token_embedding_type == 'sub_state_easy'):
                depth += vec[:,:,6]
                depth += vec[:,:,7]
                vec[:,:,-2] = costheta
                vec[:,:,-3] = sintheta
            if(self.token_embedding_type == 'sub_state_theory'):    
                depth += vec[:,:,6]
                depth += vec[:,:,7]
                for i in range(self.language_depth + 1):
                    vec[:,:, - i - 1] = (depth == i).float()
            if(self.token_embedding_type == 'sub_state_large'):
                depth += vec[:,:,6]
                depth += vec[:,:,7]
                for i in range(self.language_depth + 1):
                    for j in range(4):
                        vec[:,:, - i * 4 - j - 1] = (depth == i).float() * (vec[:,:,j + 4] == 1).float()
                for j in range(4):
                    vec[:,:,j + 4] = 0
            elif(self.token_embedding_type == 'sub_state_raw'):
                newvec = torch.zeros(vec.size(), device = self.device)
                for t in range(newvec.shape[2]):
                    newvec[:,:,t] = agg[:,:,t] / count
                vec = vec + self.embedding_projector(newvec)
        return vec
    
    def component_forward(self, batch, attention_mask):
        """ Computes the forward pass to construct prefix representations.
        Arguments:
          batch: (batch_len, seq_len) vectors representing
                 contexts
        Returns:
          hiddens: (batch_len, seq_len, hidden_size)
                   recurrent state vectors for each token in input.
        """
        if self.e_type in ['default', 'cos', 'same_trained', 'none']:
            return self.model.forward(batch, attention_mask=attention_mask).values()
        else:
            vec = self.custom_embed(batch)
            return self.model.forward(inputs_embeds=vec, attention_mask=attention_mask).values()

    def forward(self, batch, attention_mask, label_batch=None, random_batch=None):
        assert label_batch is None
        assert random_batch is None
        return self.component_forward(batch, attention_mask)
