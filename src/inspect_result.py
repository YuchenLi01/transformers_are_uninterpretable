from argparse import ArgumentParser
from seaborn import heatmap
import numpy as np
import os
import matplotlib.pyplot as plt
import torch
import utils
from run_lm import create_args, init_lm
import dataset
from reporter import report_dyck_k_constraint_eval

def cut_sentences_at_length(batch, max_sentence_len):
    assert type(batch) is tuple
    new_batch0 = batch[0][:, :max_sentence_len].clone().detach()  # observations
    new_batch1 = batch[1][:, :max_sentence_len].clone().detach()  # labels
    if batch[2] is not None:
        new_batch2 = batch[2][:, :max_sentence_len].clone().detach()  # attention_mask_all
    else:
        new_batch2 = None
    new_batch3 = [max_sentence_len]  # sentence length
    return new_batch0, new_batch1, new_batch2, new_batch3


def prepare_dev_data(dataset, num_sentences_to_plot, max_sentence_len=None):
    """
    Note: one batch contains `arg.training.batch_size` sentences
    In this case, `arg.training.batch_size` should be set to 1
    Return all the first `num_sentences_to_plot` sentences as a list.
    """
    dev_batches = dataset.get_dev_dataloader()
    batches = []
    i = 0
    for batch in dev_batches:
        if max_sentence_len is not None:
            batch = cut_sentences_at_length(batch, max_sentence_len)
        batches.append(batch)
        i += 1
        if i == num_sentences_to_plot:
            break
    return batches


def translate_ids_to_tokens(token_ids, token2id):
    ids_to_tokens = {token2id[token]: token for token in token2id}
    return [ids_to_tokens[token_id] for token_id in token_ids]


def get_bert_embeddings(bert, input_ids, token_type_ids):
    return bert.embeddings(
        input_ids=input_ids,
        position_ids=None,
        token_type_ids=token_type_ids,
        inputs_embeds=None,
        past_key_values_length=0,
    )


def get_encoder_self_attention(lm_model, input_ids):
    """
    Get the attention weights for ONE sentence
    """
    assert len(input_ids) == 1, 'the input must contain exactly 1 sentence'
    model = lm_model.model
    if lm_model.e_type in ['default', 'cos', 'same_trained', 'none']:
        all_attention_outputs = model.forward(
            input_ids,
            output_attentions=True,
        ).attentions
    else:
        vec = lm_model.custom_embed(input_ids)
        all_attention_outputs = model.forward(
            inputs_embeds=vec,
            position_ids=None,
            output_attentions=True,
        ).attentions   
    all_attention_outputs = np.array([
        a[0].cpu().detach().numpy()
        for a in all_attention_outputs
    ])  # has shape [num_hidden_layers, num_attention_heads, sentence_len, sentence_len]
    return all_attention_outputs


def plot_attention_bert(lm_model, batches, plot_save_dir):
    for sentence_idx, batch in enumerate(batches):
        all_attention_outputs = get_encoder_self_attention(lm_model.model.bert, batch[0])
        tokens = translate_ids_to_tokens(batch[0][0].to('cpu').numpy(), dataset.vocab)
        for layer_idx in range(len(all_attention_outputs)):
            for head_idx in range(len(all_attention_outputs[0])):
                plt.figure(figsize=(15, 12))
                ax = plt.axes()
                heatmap(all_attention_outputs[layer_idx][head_idx], xticklabels=tokens, yticklabels=tokens)
                ax.set_title(f"sentence{sentence_idx} layer{layer_idx} head{head_idx}\n{' '.join(tokens)}")
                plt.savefig(
                    os.path.join(plot_save_dir, f"sentence{sentence_idx}_layer{layer_idx}_head{head_idx}.png"))
                plt.show()


def get_gpt_embeddings(gpt, input_ids):
    return gpt.get_input_embeddings()(input_ids)


def plot_attention_gpt(lm_model, batches, plot_save_dir):
    for sentence_idx, batch in enumerate(batches):
        all_attention_outputs = get_encoder_self_attention(lm_model, batch[0])
        tokens = translate_ids_to_tokens(batch[0][0].to('cpu').numpy(), dataset.vocab)
        for layer_idx in range(len(all_attention_outputs)):
            for head_idx in range(len(all_attention_outputs[0])):
                plt.figure(figsize=(15, 12))
                ax = plt.axes()
                open_stack = []
                sentence = batch[0][0]
                depths = []
                opens = []
                current_depth = -1
                for rid in range(sentence.shape[0]):
                    mid = all_attention_outputs[layer_idx][head_idx][rid].argmax()
                    if(sentence[rid] in [2, 4, 5]):
                        open_stack.append(rid)
                        opens.append(1)
                        current_depth += 1
                    else:
                        open_stack.pop()
                        opens.append(0)
                        current_depth -= 1
                    depths.append(current_depth)
                    if(lm_model.e_type != 'z'):
                        cid = open_stack[-1]
                        rec_xs = [cid, cid+1, cid+1, cid, cid]
                        rec_ys = [rid, rid, rid+1, rid+1, rid]
                        if(cid == mid):
                            ax.plot(rec_xs, rec_ys, color="blue")
                        else:
                            ax.plot(rec_xs, rec_ys, color="yellow")
                    else:
                        for cid in range(len(depths)):
                            if(depths[cid] == current_depth and opens[cid]):
                                rec_xs = [cid, cid+1, cid+1, cid, cid]
                                rec_ys = [rid, rid, rid+1, rid+1, rid]
                                if(sentence[cid] in [4,6]):
                                    color = "yellow"
                                else:
                                    color = "blue"
                                ax.plot(rec_xs, rec_ys, color=color)
                                ax.plot(rec_xs, rec_ys, color=color)
                heatmap(all_attention_outputs[layer_idx][head_idx], xticklabels=tokens, yticklabels=tokens)
                ax.set_title(f"sentence{sentence_idx} layer{layer_idx} head{head_idx}\n{' '.join(tokens)}")
                plt.savefig(os.path.join(plot_save_dir, f"sentence{sentence_idx}_layer{layer_idx}_head{head_idx}.png"))
                plt.show()


def check_bert_attention_weights(
        lm_model,
        inspect_results_dir,
):
    bert = lm_model.model.bert
    num_layers = bert.config.num_hidden_layers

    fn = os.path.join(inspect_results_dir, 'attention.txt')
    with open(fn, 'wt') as f:
        for i in range(num_layers):
            attn_weights = bert.encoder.layer[0].attention.self

            # Key
            Wk = attn_weights.key.weight.detach().cpu().numpy()
            f.write(f"layer{i}_key\n")
            f.write(str(Wk))
            f.write('\n\n')
            plt.figure(figsize=(15, 12))
            ax = plt.axes()
            heatmap(Wk)
            ax.set_title(f"layer{i}_key")
            plt.savefig(os.path.join(inspect_results_dir, f"layer{i}_key.png"))
            plt.show()

            # Query
            Wq = attn_weights.query.weight.detach().cpu().numpy()
            f.write(f"layer{i}_query\n")
            f.write(str(Wq))
            f.write('\n\n')
            plt.figure(figsize=(15, 12))
            ax = plt.axes()
            heatmap(Wq)
            ax.set_title(f"layer{i}_query")
            plt.savefig(os.path.join(inspect_results_dir, f"layer{i}_query.png"))
            plt.show()

            # Value
            Wv = attn_weights.value.weight.detach().cpu().numpy()
            f.write(f"layer{i}_value\n")
            f.write(str(Wv))
            f.write('\n\n')
            plt.figure(figsize=(15, 12))
            ax = plt.axes()
            heatmap(Wv)
            ax.set_title(f"layer{i}_value")
            plt.savefig(os.path.join(inspect_results_dir, f"layer{i}_value.png"))
            plt.show()

def calculate_balance_gpt(lm_model, plot_save_dir):
    # only work for one layer one head currently
    # from IPython import embed;embed()
    assert lm_model.e_type == 'z'
    input_ids =  [4] * lm_model.language_depth  + [6] * lm_model.language_depth + [5] * lm_model.language_depth + [7] * lm_model.language_depth
    input_ids = torch.tensor(input_ids).to(lm_model.device).unsqueeze(dim = 0)
    vec = lm_model.custom_embed(input_ids)
    attn_block = lm_model.model.transformer.h[0].attn
    attn_block.is_cross_attention = True
    tokens = translate_ids_to_tokens(input_ids[0].to('cpu').numpy(), dataset.vocab)
    attn_heatmap = attn_block(
            vec,
            output_attentions=True,
    )[-1].cpu().detach().squeeze()
    length = attn_heatmap.shape[0] // 2
    import numpy as np
    propotion_statistics = np.zeros((length, length -2 ))
    for idx in range(length//2):
        corresponding_brackets = attn_heatmap[:,idx] / attn_heatmap[:,length - 1 - idx]
        corresponding_brackets_at_close = np.concatenate([corresponding_brackets[length//2 + 1: length], corresponding_brackets[length//2 + 1 + length:]]) 
        propotion_statistics[idx] = corresponding_brackets_at_close
    for idx in range(length//2):
        corresponding_brackets = attn_heatmap[:,idx + length] / attn_heatmap[:,2* length - 1 - idx]
        corresponding_brackets_at_close = np.concatenate([corresponding_brackets[length//2 + 1: length], corresponding_brackets[length//2 + 1 + length:]]) 
        propotion_statistics[idx + length//2] = corresponding_brackets_at_close 
    attn_block.is_cross_attention = False
    torch.save(propotion_statistics.std(axis = 1).mean(), os.path.join(plot_save_dir, f"balance.pt"))
    
def better_balance_gpt(lm, plot_save_dir):
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
        norm_statistics += ((op + cp)[length//2 + 1: length].std(dim = 1)**2).sum()
        norm_statistics += ((op + cp)[length//2 + 1 + length:].std(dim = 1)**2).sum()
    for idx in range(length//2):
        op_weight = attn_heatmap[:,idx + length] 
        cp_weight = attn_heatmap[:,2 * length - 1 - idx]
        op_value = value[idx + length]
        cp_value = value[2 * length - 1 - idx]
        op = op_weight.unsqueeze(dim = -1) @ op_value.unsqueeze(dim = 0)
        cp = cp_weight.unsqueeze(dim = -1) @ cp_value.unsqueeze(dim = 0)
        norm_statistics += ((op + cp)[length//2 + 1: length].std(dim = 1)**2).sum()
        norm_statistics += ((op + cp)[length//2 + 1 + length:].std(dim = 1)**2).sum()
    attn_block.is_cross_attention = False
    torch.save(norm_statistics / (2 * value.norm()**2), os.path.join(plot_save_dir, f"new_balance.pt"))

def visualize_attention_map_gpt(lm_model, plot_save_dir):
    # only work for one layer one head currently
    # from IPython import embed;embed()
    assert lm_model.e_type == 'z'
    input_ids =  [4] * lm_model.language_depth  + [6] * lm_model.language_depth + [5] * lm_model.language_depth + [7] * lm_model.language_depth
    input_ids = torch.tensor(input_ids).to(lm_model.device).unsqueeze(dim = 0)
    vec = lm_model.custom_embed(input_ids)
    attn_block = lm_model.model.transformer.h[0].attn
    attn_block.is_cross_attention = True
    tokens = translate_ids_to_tokens(input_ids[0].to('cpu').numpy(), dataset.vocab)
    attn_outputs = attn_block(
            vec,
            output_attentions=True,
    )[-1].cpu().detach().squeeze()
    plt.figure(figsize=(15, 12))
    ax = plt.axes()
    ax.set_title(f"Heatmap over \n{' '.join(tokens)}")
    heatmap(attn_outputs, xticklabels=tokens, yticklabels=tokens)
    plt.savefig(os.path.join(plot_save_dir, f"example.png"))
    torch.save(attn_outputs, os.path.join(plot_save_dir, f"heatmap.pt"))
    attn_block.is_cross_attention = False
if __name__ == '__main__':
    # For synthetic Dyck dataset

    # Parse args
    argp = ArgumentParser()
    argp.add_argument('config')
    args = argp.parse_args()
    args = create_args(args.config)
    # Only one sentence in a batch, so that displayed sentence length is different for different sentences
    args['training']['batch_size'] = 1
    # Do not mask any token during evaluation
    args['training']['mask_prob'] = 0.0

    dataset = dataset.Dataset(args)

    name_base = args['name']
    for experiment_index in range(args['experiment']['repeat']):
        args['name'] = name_base + str(experiment_index)
        lm_model = init_lm(args)
        
        # load_dict
        if not args['reporting']['random']:
            lm_params_path = utils.get_lm_path_of_args(args)
            if not os.path.exists(lm_params_path):
                print(f"Warning: trained model does not exist: {lm_params_path}")
                continue
            device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")  # Determine whether CUDA is available
            lm_model.load_state_dict(torch.load(lm_params_path, map_location=device))
        lm_model.eval()
        inspect_results_dir = os.path.join(args['reporting']['inspect_results_dir'], args['name'])
        os.makedirs(inspect_results_dir, exist_ok=True)

        # Plot attention
        num_sentences_to_plot = args['reporting']['num_sentences_to_plot']
        batches = prepare_dev_data(dataset, num_sentences_to_plot)

        plot_save_dir = os.path.join(args['reporting']['plot_attention_dir'], args['name'])
        os.makedirs(plot_save_dir, exist_ok=True)

        if args['lm']['lm_type'] in {'BertForMaskedLM', 'BertForMaskedLMCustom'}:
            check_bert_attention_weights(lm_model, inspect_results_dir)
            plot_attention_bert(lm_model, batches, plot_save_dir)
        elif args['lm']['lm_type'] in {'GPT2LMHeadModel', 'GPT2LMHeadModelCustom'}:
            if args['lm']['embedding_type'] == 'z':
                better_balance_gpt(lm_model, plot_save_dir)
                calculate_balance_gpt(lm_model, plot_save_dir)
                visualize_attention_map_gpt(lm_model, plot_save_dir)
            plot_attention_gpt(lm_model, batches, plot_save_dir)
        else:
            raise NotImplementedError('Model not supported.')
