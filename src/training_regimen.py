""" Training loop for LMs, with mostly hard-coded decisions.
"""
import sys
import random
import torch
import torch.nn as nn
from torch import optim
from tqdm.auto import tqdm
import utils
import wandb
import sam
from reporter import report_dyck_k_constraint_eval, deterministic_eval, classification_eval

def translate_ids_to_tokens(token_ids, token2id):
    ids_to_tokens = {token2id[token]: token for token in token2id}
    return [ids_to_tokens[token_id] for token_id in token_ids]
def calculate_balance_gpt(lm, dataset):
    # only work for one layer one head currently
    assert lm.e_type == 'z'
    input_ids =  [4] * lm.language_depth  + [6] * lm.language_depth + [5] * lm.language_depth + [7] * lm.language_depth
    input_ids = torch.tensor(input_ids).to(lm.device).unsqueeze(dim = 0)
    vec = lm.custom_embed(input_ids)
    attn_block = lm.model.transformer.h[0].attn
    attn_block.is_cross_attention = True
    tokens = translate_ids_to_tokens(input_ids[0].to('cpu').numpy(), dataset.vocab)
    attn_heatmap = attn_block(
            vec,
            output_attentions=True,
    )[-1].squeeze()
    length = attn_heatmap.shape[0] // 2
    import numpy as np
    propotion_statistics = torch.zeros((length, length -2 ))
    for idx in range(length//2):
        corresponding_brackets = attn_heatmap[:,idx] / attn_heatmap[:,length - 1 - idx]
        corresponding_brackets_at_close = torch.cat([corresponding_brackets[length//2 + 1: length], corresponding_brackets[length//2 + 1 + length:]]) 
        propotion_statistics[idx] = corresponding_brackets_at_close
    for idx in range(length//2):
        corresponding_brackets = attn_heatmap[:,idx + length] / attn_heatmap[:,2* length - 1 - idx]
        corresponding_brackets_at_close = torch.cat([corresponding_brackets[length//2 + 1: length], corresponding_brackets[length//2 + 1 + length:]]) 
        propotion_statistics[idx + length//2] = corresponding_brackets_at_close 
    attn_block.is_cross_attention = False
    return propotion_statistics.std(axis = 1).mean()

def better_balance_gpt(lm, dataset):
    assert lm.e_type == 'z'
    input_ids =  [4] * lm.language_depth  + [6] * lm.language_depth + [5] * lm.language_depth + [7] * lm.language_depth
    input_ids = torch.tensor(input_ids).to(lm.device).unsqueeze(dim = 0)
    vec = lm.custom_embed(input_ids)
    attn_block = lm.model.transformer.h[0].attn
    attn_block.is_cross_attention = True
    tokens = translate_ids_to_tokens(input_ids[0].to('cpu').numpy(), dataset.vocab)
    output = attn_block(
                vec,
                output_attentions=True,
        )
    output_vec = output[0]
    value = output[1].squeeze()
    attn_heatmap = output[2].squeeze()
    norm_statistics = 0
    length = attn_heatmap.shape[0] // 2
    for idx in range(length//2):
        op_weight = attn_heatmap[:,idx] 
        cp_weight = attn_heatmap[:,length - 1 - idx]
        op_value = value[idx]
        cp_value = value[length - 1 - idx]
        op = op_weight.unsqueeze(dim = -1) @ op_value.unsqueeze(dim = 0)
        cp = cp_weight.unsqueeze(dim = -1) @ cp_value.unsqueeze(dim = 0)
        norm_statistics += (op + cp).norm() ** 2
    for idx in range(length//2):
        op_weight = attn_heatmap[:,idx + length] 
        cp_weight = attn_heatmap[:,2 * length - 1 - idx]
        op_value = value[idx + length]
        cp_value = value[2 * length - 1 - idx]
        op = op_weight.unsqueeze(dim = -1) @ op_value.unsqueeze(dim = 0)
        cp = cp_weight.unsqueeze(dim = -1) @ cp_value.unsqueeze(dim = 0)
        norm_statistics += (op + cp).norm() ** 2
    attn_block.is_cross_attention = False
    return norm_statistics / (2 * value.norm()**2)


def balance_variation(lm, dataset):
    input_ids =  [2] + [4] * lm.language_depth  + [6] * lm.language_depth + [5] * lm.language_depth + [7] * lm.language_depth
    input_ids = torch.tensor(input_ids).to(lm.device).unsqueeze(dim = 0)
    vec = lm.custom_embed(input_ids)
    attn_block = lm.model.transformer.h[0].attn
    attn_block.is_cross_attention = True
    tokens = translate_ids_to_tokens(input_ids[0].to('cpu').numpy(), dataset.vocab)
    output = attn_block(
            vec,
            output_attentions=True,
    )
    output_vec = output[0]
    value = output[1].squeeze()
    value = value - value.mean(dim = 1, keepdim = True)
    attn_heatmap = output[2].squeeze()
    norm_statistics = 0
    depth = attn_heatmap.shape[0] // 4
    def get_loc(t, d, o):
        return t * 2 * depth + o * depth + d + 1
    balance_violation = []
    for t_prime in range(2):
        for d_prime in range(1, depth):
            loc =  get_loc(t_prime, d_prime, 1)
            embedding = value[loc]
            attention_weight = attn_heatmap[loc]
            nominators = []
            for t in range(2):
                for d in range(depth):
                    nominators.append((attention_weight[get_loc(t, d, 0)] * value[get_loc(t, d, 0)] + attention_weight[get_loc(t, d, 1)] * value[get_loc(t, d, 1)]).norm())
            nominator = max(nominators)
            base_denominator = attention_weight[0] * value[0] + attention_weight[get_loc(t_prime, d_prime, 0)] * value[get_loc(t_prime, d_prime, 0)] + attention_weight[get_loc(t_prime, d_prime, 1)] * value[get_loc(t_prime, d_prime, 1)]
            denominators = []
            r_list = []
            for r in range(2 << d_prime):
                denominator = base_denominator.clone()
                type_list = []
                for d_iter in range(d_prime):
                    new_type = r%2
                    denominator = denominator + attention_weight[get_loc(new_type, d_iter, 0)] * value[get_loc(new_type, d_iter, 0)]
                    r = r//2
                    type_list.append(new_type)
                denominators.append(denominator.norm())
                r_list.append(type_list)
            denominator = min(denominators)
            balance_violation.append((nominator/denominator).detach().cpu())
    import numpy as np
    attn_block.is_cross_attention = False
    return (np.mean(balance_violation))

def abbrv_module_name(name):
    name = name.replace('module.','')
    name = name.replace('bias','b')
    name = name.replace('weight','w')
    name = name.replace('features','L')        
    split_name = name.split('.')
    name = '.'.join(split_name[:])
    return name
def _log_norm(model):
    logs = {'norm/total':0}
    for n,p in model.named_parameters():
        if('c_attn.weight' in n):
            d = p.shape[0]
            names = ['query','key','value']
            for idx in range(3):
                norm = p[:,idx*d : idx * d + d].norm().detach() 
                logs['norm/total'] += norm ** 2
                logs['norm/'+abbrv_module_name(n)+'.'+names[idx]] = norm
        else:
            norm = p.norm().detach()
            logs['norm/total'] += norm ** 2
            logs['norm/'+ abbrv_module_name(n)] = norm
    logs['norm/total'] = torch.sqrt(logs['norm/total'])
    return logs

    





def compute_batch_loss(
        args,
        lm,
        observation_batch,
        label_batch,
        attention_mask,
        dataset,
        length,
        include_reg=True,
):
    def _compute_batch_loss(loss, logits, label_batch, include_reg):
        batch_loss = loss(logits, label_batch)
        if include_reg:
            for param in lm.parameters():
                if param.requires_grad:
                    batch_loss += args['training']['weight_decay'] / 2 * torch.norm(param.data, p='fro') ** 2
        return batch_loss
    
    
    if args['training']['objective'] not in {"classify"}:
        loss = nn.CrossEntropyLoss()
        batch_size, seq_len = label_batch.size()[0], label_batch.size()[1]
        if args['lm']['lm_type'] in {'BertForMaskedLM', 'BertForMaskedLMCustom'}:
            logits, = lm(observation_batch, attention_mask)
        elif args['lm']['lm_type'] in {'GPT2LMHeadModel','GPT2LMHeadModelCustom'}:
            logits, _ = lm(observation_batch, attention_mask)
        else:
            raise NotImplementedError('Model not supported.')

        logits = logits.view(batch_size * seq_len, -1)

        if len(label_batch.size()) == 2:  # (batch_size, seq_len)
            label_batch = label_batch.view(batch_size * seq_len, )
        else:
            assert len(label_batch.size()) == 3  # (batch_size, seq_len, vocab_size)
            label_batch = label_batch.view(batch_size * seq_len, -1)
        batch_loss = _compute_batch_loss(loss, logits, label_batch, include_reg)
    else:
        loss = nn.CrossEntropyLoss()
        if args['lm']['lm_type'] in {'BertForMaskedLM', 'BertForMaskedLMCustom'}:
            logits, = lm(observation_batch, attention_mask)
        elif args['lm']['lm_type'] in {'GPT2LMHeadModel','GPT2LMHeadModelCustom'}:
            logits, _ = lm(observation_batch, attention_mask)
        else:
            raise NotImplementedError('Model not supported.')
        prediction = torch.stack([logits[x, y - 1] for x, y in enumerate(length)])
        prediction = prediction[:, :2]
        label_batch = torch.tensor(label_batch).to(prediction.device)
        batch_loss = _compute_batch_loss(loss, prediction, label_batch, include_reg)

    if args['training']['objective'] in {'default', 'classify'}:
        return (batch_loss,)
    elif args['training']['objective'] in {'contrastive'}:
        input_ids =  [4] * lm.language_depth  + [6] * lm.language_depth + [5] * lm.language_depth + [7] * lm.language_depth
        d = random.randint(0, lm.language_depth - 1)
        t = random.randint(0, 1)
        input_ids = input_ids[:t * 2 * lm.language_depth + d] + [4 + t, 6 + t] + input_ids[t * 2 * lm.language_depth + d : ]
        input_ids = torch.tensor(input_ids).to(lm.device).unsqueeze(dim = 0).repeat(observation_batch.shape[0],1)
        extend_observation_batch = torch.cat([input_ids, observation_batch], dim = 1)
        if args['lm']['lm_type'] in {'BertForMaskedLM', 'BertForMaskedLMCustom'}:
            raise NotImplementedError
        elif args['lm']['lm_type'] in {'GPT2LMHeadModel','GPT2LMHeadModelCustom'}:
            extend_logits, _ = lm(extend_observation_batch, attention_mask)
            extend_logits = extend_logits[:,-observation_batch.shape[1]:]
        else:
            raise NotImplementedError('Model not supported.')
        extend_logits = extend_logits.reshape(batch_size * seq_len, -1)
        contrastive_loss = nn.MSELoss(reduction='mean')(extend_logits, logits)
        return (batch_loss, contrastive_loss)
    elif args['training']['objective'] in {'balance'}:
        balance_loss = better_balance_gpt(lm, dataset)
        return (batch_loss, balance_loss)
    
    raise NotImplementedError(f"Undefined args['training']['objective']: {args['training']['objective']}")


def train(
        args,
        lm,
        train_batches,
        dev_batches,
        steps_between_logging=5,
        steps_between_evals=None,
        dataset=None,
):
    """Trains the language model with Adam,

    Arguments:
      lm: a LanguageModel object
      train_batches: PyTorch DataLoader of training data from Dataset
      dev_batches: PyTorch DataLoader of dev data from Dataset
    """
    lm_params_path = utils.get_lm_path_of_args(args)
    print(lm_params_path)

    if args['training']['optimizer'] == 'Adam':
        optimizer = optim.AdamW(
            [param for param in lm.parameters() if param.requires_grad],
            args['training']['learning_rate'],
            weight_decay=0.0,  # not args['training']['weight_decay'] because l2 reg is added to loss
        )
    elif args['training']['optimizer'] == 'SGD':
        optimizer = optim.SGD(
            [param for param in lm.parameters() if param.requires_grad],
            args['training']['learning_rate'],
            weight_decay=0.0,  # not args['training']['weight_decay'] because l2 reg is added to loss
        )
    elif args['training']['optimizer'] == 'SAM':
        optimizer = sam.SAM([param for param in lm.parameters() if param.requires_grad], optim.AdamW, rho = args['training']['rho'], lr = args['training']['learning_rate'], weight_decay=0.0)
    else:
        raise NotImplementedError(f"optimizer {args['training']['optimizer']} is not supported ")
    max_epochs = args['training']['max_epochs']
    if steps_between_evals is None:
        steps_between_evals = len(train_batches) 
        
    if args['training']['scheduler'] == 'None':
        scheduler = None
    elif args['training']['scheduler'] == 'Cosine':
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, max_epochs)
    

    min_dev_loss = sys.maxsize
    min_dev_loss_epoch = -1
    torch.save(args, utils.get_arg_path_of_args(args))
    torch.save(dataset, utils.get_dataset_path_of_args(args))
    torch.save(lm.state_dict(), lm_params_path)
    total_gradient_steps = 0
    for epoch_index in range(max_epochs):
        epoch_train_loss = 0
        epoch_regularize_loss = 0
        train_batch_count = 0
        for observation_batch, label_batch, attention_mask, length in train_batches:
            if args['training']['mask_prob'] == 0.0:
                assert attention_mask is None
            else:
                assert attention_mask is not None
            # Compute forward, backward, and take gradient step
            lm.train()
            
            
            batch_loss = compute_batch_loss(
                args,
                lm,
                observation_batch,
                label_batch,
                attention_mask,
                dataset,
                length,
                include_reg=True,
            )
            if(len(batch_loss) == 1):
                batch_loss, = batch_loss
                new_loss = batch_loss
            elif(len(batch_loss) == 2):
                batch_loss, regularize_loss = batch_loss
                epoch_regularize_loss += regularize_loss.detach().cpu().numpy()
                new_loss = args['training']['regularize_penalty'] * regularize_loss + batch_loss
            epoch_train_loss += batch_loss.detach().cpu().numpy()
            train_batch_count += 1
            new_loss.backward(retain_graph=True)
            if total_gradient_steps % steps_between_logging == 0:
                epoch_avg_train_loss = epoch_train_loss / train_batch_count
                epoch_avg_regularize_loss = epoch_regularize_loss / train_batch_count
                results = {
                    'epoch': epoch_index,
                    'train_loss': epoch_avg_train_loss,
                    'regularize_loss': epoch_avg_regularize_loss
                }
                if(args['training']['log_balance']):
                    results['balance_variation'] = balance_variation(lm, dataset)
                results.update(_log_norm(lm))
                wandb.log(results)
                
            
            if(not isinstance(optimizer, sam.SAM)):
                optimizer.step()
                optimizer.zero_grad()
            else:
                optimizer.first_step(zero_grad=True)
                batch_loss = compute_batch_loss(
                    args,
                    lm,
                    observation_batch,
                    label_batch,
                    attention_mask,
                    dataset,
                    length,
                    include_reg=True,
                )
                if(len(batch_loss) == 1):
                    batch_loss, = batch_loss
                    new_loss = batch_loss
                elif(len(batch_loss) == 2):
                    batch_loss, regularize_loss = batch_loss
                    epoch_regularize_loss += regularize_loss.detach().cpu().numpy()
                    new_loss = args['training']['regularize_penalty'] * regularize_loss + batch_loss
                new_loss.backward()
                optimizer.second_step(zero_grad=True)


            # Determine whether it's time to evaluate on dev data
            if total_gradient_steps % steps_between_evals == 0:
                epoch_dev_loss = 0
                epoch_dev_regularize_loss = 0
                dev_batch_count = 0
                # Compute dev loss
                for observation_batch, label_batch, attention_mask, length in dev_batches:
                    if args['training']['mask_prob'] == 0.0 or args['training']['clean_valid']:
                        assert attention_mask is None
                    else:
                        assert attention_mask is not None
                    dev_batch_count += 1
                    optimizer.zero_grad()
                    lm.eval()
                    batch_loss = compute_batch_loss(
                        args,
                        lm,
                        observation_batch,
                        label_batch,
                        attention_mask,
                        dataset,
                        length,
                        include_reg=False,
                    )
                    if(len(batch_loss) == 1):
                        batch_loss, = batch_loss
                    elif(len(batch_loss) == 2):
                        batch_loss, regularize_loss = batch_loss
                        epoch_dev_regularize_loss += regularize_loss.detach().cpu().numpy()
                    epoch_dev_loss += batch_loss.detach().cpu().numpy()

                epoch_avg_dev_loss = epoch_dev_loss/ dev_batch_count
                epoch_avg_dev_regularize_loss = epoch_dev_regularize_loss / dev_batch_count
                epoch_avg_train_loss = epoch_train_loss/ train_batch_count
                epoch_avg_regularize_loss = epoch_regularize_loss / train_batch_count

                results = {
                    'epoch': epoch_index,
                    'train_loss': epoch_avg_train_loss,
                    'dev_loss': epoch_avg_dev_loss,
                    'regularize_loss': epoch_avg_regularize_loss,
                    'regularize_dev_loss': epoch_avg_dev_regularize_loss
                }
                if(args['training']['log_balance']):
                    results['balance_variation'] = balance_variation(lm, dataset)
                if args['language']['name'] == 'dyck' and args['training']['objective'] not in {'classify'}:
                  # Constraint eval acc only implemented for Dyck
                  if dataset is not None:
                      for split in ['train', 'dev']:
                        if args['lm']['lm_type'] in {'BertForMaskedLM', 'BertForMaskedLMCustom'}:
                            acc = deterministic_eval(args, lm, dataset, split)
                        elif args['lm']['lm_type'] in {'GPT2LMHeadModel','GPT2LMHeadModelCustom'}:
                            acc = report_dyck_k_constraint_eval(args, lm, dataset, split)
                        else:
                            raise NotImplementedError('Model not supported.')
                        results.update({f'{split}_acc': acc})
                        torch.save([acc], utils.get_acc_path_of_args(args, split))
                elif args['language']['name'] == 'dyck' and args['training']['objective'] in {'classify'}:
                    if dataset is not None:
                        for split in ['train', 'dev']:
                            acc = classification_eval(args, lm, dataset, split)
                            results.update({f'{split}_acc': acc})
                results.update(_log_norm(lm))
                wandb.log(results)
                tqdm.write(
                    '[epoch {}] Train loss: {}, Dev loss: {}'.format(
                        epoch_index,
                        epoch_avg_train_loss,
                        epoch_avg_dev_loss,
                    )
                )
                torch.save(lm.state_dict(), lm_params_path)
                min_dev_loss = epoch_avg_dev_loss
                min_dev_loss_epoch = epoch_index
                tqdm.write('Saving lm parameters')

            total_gradient_steps += 1
        if(scheduler):
            scheduler.step()

        tqdm.write("Min dev loss: {}".format(min_dev_loss))
