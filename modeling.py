import torch
import torch.nn as nn
from transformers import LlamaTokenizer, LlamaForCausalLM, XLMRobertaTokenizer
from torchscale.architecture.config import EncoderConfig
from ofa.tokenization_ofa import OFATokenizer
from ofa.modeling_ofa import OFAModel
from BEiT3 import BEiT3Wrapper

VICUNA_DIALOG_PROMPT = "###Human: <Img>{}</Img> {} {}\n###Assistant:"

class OFAMultiBLIP(nn.Module):
    def __init__(self, args):
        super(OFAMultiBLIP, self).__init__()
        self.args = args
        # init encoder
        ofa_model = OFAModel.from_pretrained(self.args.encoder_ckpt)
        ofa_model.decoder = None
        self.encoder = ofa_model.encoder
        if args.train_mask:
            for name, param in self.encoder.named_parameters():
                if 'mask_embed_increment' not in name:
                    param.requires_grad = False
        else:
            for name, param in self.encoder.named_parameters():
                param.requires_grad = False
        self.encoder_tokenizer = OFATokenizer.from_pretrained(self.args.encoder_ckpt)   
        # init projection layer
        self.query_projection = nn.Linear(self.get_ofa_hidden_size(self.args.arch), 4096) #4096是llama-7b的hidden size
        # init decoder 
        self.decoder = LlamaForCausalLM.from_pretrained(args.llama_ckpt, load_in_8bit=False, torch_dtype=torch.float16)
        self.decoder_tokenizer = LlamaTokenizer.from_pretrained(self.args.llama_ckpt)
        self.decoder_tokenizer.pad_token = self.decoder_tokenizer.unk_token
        self.end_sym = self.decoder_tokenizer.eos_token

        for name, param in self.decoder.named_parameters():
            param.requires_grad = False
        self.decoder.train = disabled_train

    
    def get_image_queries(self, patch_images, encoder_input_text):
        encoder_input_ids = self.encoder_tokenizer(encoder_input_text, padding='longest', truncation=True, max_length=self.args.max_src_len, return_tensors='pt').input_ids.to(patch_images.device)
        patch_masks = torch.full((encoder_input_ids.shape[0], 1), True).to(patch_images.device)
        encoder_output = self.encoder(encoder_input_ids, patch_images=patch_images, patch_masks=patch_masks)
        queries_idx = torch.where(encoder_input_ids == self.encoder_tokenizer.mask_token_id, 1, 0)
        queries_idx = queries_idx.nonzero()
        queries_idx[:,1] += 900 # 900 is the size of image feature sequence
        queries = encoder_output.last_hidden_state[queries_idx[:,0], queries_idx[:,1]] 
        queries = queries.reshape(-1, self.args.query_num, self.get_ofa_hidden_size(self.args.arch))
        atts_queries = torch.ones(queries.size()[:-1], dtype=torch.long).to(queries.device)

        return queries, atts_queries


    def prompt_wrap(self, queries, atts_queries, prompt='###Human: <Img><ImageHere></Img>'):
        if prompt:
            batch_size = queries.shape[0]
            p_before, p_after = prompt.split('<ImageHere>')
            p_before_tokens = self.decoder_tokenizer(p_before, return_tensors='pt', add_special_tokens=False).to(queries.device)
            p_after_tokens = self.decoder_tokenizer(p_after, return_tensors='pt', add_special_tokens=False).to(queries.device)
            p_before_embeds = self.decoder.model.embed_tokens(p_before_tokens.input_ids).expand(batch_size, -1, -1)
            p_after_embeds = self.decoder.model.embed_tokens(p_after_tokens.input_ids).expand(batch_size, -1, -1)
            wrapped_queries = torch.cat((p_before_embeds, queries, p_after_embeds), dim=1)
            wrapped_atts_queries = atts_queries[:, :1].expand(-1, wrapped_queries.shape[1])
        else:
            wrapped_queries, wrapped_atts_queries = queries, atts_queries

        return wrapped_queries, wrapped_atts_queries


    def generate_data_embeds(self, instruction, input, target=None, device='cpu', is_train=True):
        assert len(instruction) == len(input)
        if is_train :
            assert len(instruction) == len(target)
            user_prompt, full_prompt = [], []
            for i in range(len(instruction)):
                user_prompt.append(instruction[i] + input[i] + '###Assistant:')
                full_prompt.append(instruction[i] + input[i] + '###Assistant:' + target[i] + self.end_sym)
            user_prompt_token_ids = self.decoder_tokenizer(user_prompt, padding='longest', add_special_tokens=False, return_tensors='pt').input_ids.to(device)
            full_prompt_token_ids = self.decoder_tokenizer(full_prompt, padding='longest', add_special_tokens=False, return_tensors='pt').input_ids.to(device)
            data_embeds = self.decoder.model.embed_tokens(full_prompt_token_ids)
            atts = torch.ones_like(full_prompt_token_ids).masked_fill(full_prompt_token_ids == self.decoder_tokenizer.pad_token_id, -100)
            targets = full_prompt_token_ids.masked_fill(full_prompt_token_ids == self.decoder_tokenizer.pad_token_id, -100)
            # mask input
            for i in range(full_prompt_token_ids.shape[0]):
                for j, id in enumerate(user_prompt_token_ids[i]):
                    if id != self.decoder_tokenizer.pad_token_id:
                        targets[i][j] = -100
                    else:
                        break
            
            return data_embeds, targets, atts
        else:
            user_prompt = []
            for i in range(len(instruction)):
                user_prompt.append(instruction[i] + input[i] + '###Assistant:')
            user_prompt_token_ids = self.decoder_tokenizer(user_prompt, padding='longest', add_special_tokens=False, return_tensors='pt').input_ids.to(device)
            data_embeds = self.decoder.model.embed_tokens(user_prompt_token_ids)
            atts = torch.ones_like(user_prompt_token_ids).masked_fill(user_prompt_token_ids == self.decoder_tokenizer.pad_token_id, -100)
            
            return data_embeds, atts
        

    def forward(self, data, device):
        self.decoder_tokenizer.padding_side='right'
        patch_images, instruction, input, target = data['img'].half().to(device), data['instruction'], data['input'], data['target']
        queries_tag = ''.join([self.encoder_tokenizer.mask_token * self.args.query_num])
        encoder_input_text = [instruction[i] + input[i] + queries_tag for i in range(len(instruction))]
        queries, atts_queries = self.get_image_queries(patch_images, encoder_input_text)

        queries = self.query_projection(queries)
        queries, atts_queries = self.prompt_wrap(queries, atts_queries)

        data_embeds, targets, atts = self.generate_data_embeds(instruction, input, target, device)
        # concat data_embeds and queries
        data_embeds = torch.cat((queries, data_embeds), dim=1)
        empty_targets = torch.ones([atts_queries.shape[0], atts_queries.shape[1] + 1], dtype=torch.long).to(device).fill_(-100) # plus one for bos
        targets = torch.cat((empty_targets, targets), dim=1)
        atts = torch.cat((atts_queries, atts), dim=1)
        # add bos
        bos = torch.ones([queries.shape[0], 1], device=device, dtype=torch.long) * self.decoder_tokenizer.bos_token_id
        bos_embeds = self.decoder.model.embed_tokens(bos)
        atts_bos = atts_queries[:, :1]
        data_embeds = torch.cat((bos_embeds, data_embeds), dim=1)
        atts = torch.cat((atts_bos, atts), dim=1)

        loss = self.decoder(inputs_embeds=data_embeds, attention_mask=atts, labels=targets, return_dict=True)['loss']

        return loss


    def generate(self, data, device):
        self.decoder_tokenizer.padding_side='left'
        with torch.no_grad():
            patch_images, instruction, input = data['img'].half().to(device), data['instruction'], data['input']
            queries_tag = ''.join([self.encoder_tokenizer.mask_token * self.args.query_num])
            encoder_input_text = [instruction[i] + input[i] + queries_tag for i in range(len(instruction))]
            queries, atts_queries = self.get_image_queries(patch_images, encoder_input_text)

        queries = self.query_projection(queries)
        queries, atts_queries = self.prompt_wrap(queries, atts_queries)

        data_embeds, atts = self.generate_data_embeds(instruction, input, target=None, device=device, is_train=False)
        # concat data_embeds and queries
        data_embeds = torch.cat((queries, data_embeds), dim=1)

        atts = torch.cat((atts_queries, atts), dim=1)
        # add bos
        bos = torch.ones([queries.shape[0], 1], device=device, dtype=torch.long) * self.decoder_tokenizer.bos_token_id
        bos_embeds = self.decoder.model.embed_tokens(bos)
        atts_bos = atts_queries[:, :1]
        data_embeds = torch.cat((bos_embeds, data_embeds), dim=1)
        atts = torch.cat((atts_bos, atts), dim=1)

        decoder_inputs = {
            'inputs_embeds' : data_embeds,
            #'attention_mask': atts,
            'max_new_tokens': self.args.max_tgt_len,
        }
        generate_ids = self.decoder.generate(**decoder_inputs)
        # generate_ids = torch.where(generate_ids < 0, self.decoder_tokenizer.pad_token_id, generate_ids)
        generate_texts = self.decoder_tokenizer.batch_decode(generate_ids, skip_special_tokens=True, clean_up_tokenization_spaces=False)
        # generate_texts = self.batch_decode(generate_ids) # 为什么生成-1的id然后报错？

        return generate_texts


    def generate_with_config(self, data, config, device):
        self.decoder_tokenizer.padding_side='left'
        with torch.no_grad():
            patch_images, instruction, input = data['img'].half().to(device), data['instruction'], data['input']
            queries_tag = ''.join([self.encoder_tokenizer.mask_token * self.args.query_num])
            encoder_input_text = [instruction[i] + input[i] + queries_tag for i in range(len(instruction))]
            queries, atts_queries = self.get_image_queries(patch_images, encoder_input_text)

        queries = self.query_projection(queries)
        queries, atts_queries = self.prompt_wrap(queries, atts_queries)

        data_embeds, atts = self.generate_data_embeds(instruction, input, target=None, device=device, is_train=False)
        # concat data_embeds and queries
        data_embeds = torch.cat((queries, data_embeds), dim=1)

        atts = torch.cat((atts_queries, atts), dim=1)
        # add bos
        bos = torch.ones([queries.shape[0], 1], device=device, dtype=torch.long) * self.decoder_tokenizer.bos_token_id
        bos_embeds = self.decoder.model.embed_tokens(bos)
        atts_bos = atts_queries[:, :1]
        data_embeds = torch.cat((bos_embeds, data_embeds), dim=1)
        atts = torch.cat((atts_bos, atts), dim=1)

        config['inputs_embeds'] = data_embeds
        config['max_new_tokens'] = self.args.max_tgt_len
        generate_ids = self.decoder.generate(**config)
        # generate_ids = torch.where(generate_ids < 0, self.decoder_tokenizer.pad_token_id, generate_ids)
        generate_texts = self.decoder_tokenizer.batch_decode(generate_ids, skip_special_tokens=True, clean_up_tokenization_spaces=False)
        # generate_texts = self.batch_decode(generate_ids) # 为什么生成-1的id然后报错？

        return generate_texts


    def get_ofa_hidden_size(self, arch):
        arch = arch.lower().strip()
        arch2hidden_dim = {'tiny':256, 'medium':512, 'base':768, 'large':1024, 'huge':1280}
        return arch2hidden_dim[arch]


class BEiT3MultiBLIP(nn.Module):
    def __init__(self, args):
        super(BEiT3MultiBLIP, self).__init__()
        self.args = args
        # init encoder
        self.encoder = prepare_beit3_model(args)

        if args.train_mask:
            for name, param in self.encoder.named_parameters():
                if 'mask_embed_increment' not in name:
                    param.requires_grad = False
        else:
            for name, param in self.encoder.named_parameters():
                param.requires_grad = False
        self.encoder_tokenizer = XLMRobertaTokenizer("./pretrained_models/beit3.spm")   
        # init projection layer
        self.query_projection = nn.Linear(self.get_beit3_hidden_size(self.args.arch), 4096) # 4096是llama-7b的hidden size
        # init decoder 
        self.decoder = LlamaForCausalLM.from_pretrained(args.llama_ckpt, load_in_8bit=False).half()
        self.decoder_tokenizer = LlamaTokenizer.from_pretrained(self.args.llama_ckpt)
        self.decoder_tokenizer.pad_token = self.decoder_tokenizer.unk_token
        self.end_sym = self.decoder_tokenizer.eos_token

        for name, param in self.decoder.named_parameters():
            param.requires_grad = False
        self.decoder.train = disabled_train

    def get_image_queries(self, patch_images, encoder_input_text):
        encoder_input_ids = self.encoder_tokenizer(encoder_input_text, padding='longest', truncation=True, max_length=self.args.max_src_len, return_tensors='pt').input_ids.to(patch_images.device)
        queries = torch.full((len(encoder_input_text), self.args.query_num), self.encoder_tokenizer.mask_token_id).to(encoder_input_ids.device)
        encoder_input_ids = torch.cat((encoder_input_ids[:, 0].unsqueeze(1), queries, encoder_input_ids[:, 1:]), dim=1)
        encoder_output = self.encoder(encoder_input_ids, patch_images)
        split_idx = encoder_output['multiway_split_position']

        queries_idx = torch.where(encoder_input_ids == self.encoder_tokenizer.mask_token_id, 1, 0)
        queries_idx = queries_idx.nonzero()
        queries_idx[:,1] += split_idx # split_idx is the size of image feature sequence
        queries = encoder_output['encoder_out'][queries_idx[:,0], queries_idx[:,1]] 
        
        queries = queries.reshape(-1, self.args.query_num, self.get_beit3_hidden_size(self.args.arch))
        atts_queries = torch.ones(queries.size()[:-1], dtype=torch.long).to(queries.device)

        return queries, atts_queries


    def prompt_wrap(self, queries, atts_queries, prompt='###Human: <Img><ImageHere></Img>'):
        if prompt:
            batch_size = queries.shape[0]
            p_before, p_after = prompt.split('<ImageHere>')
            p_before_tokens = self.decoder_tokenizer(p_before, return_tensors='pt', add_special_tokens=False).to(queries.device)
            p_after_tokens = self.decoder_tokenizer(p_after, return_tensors='pt', add_special_tokens=False).to(queries.device)
            p_before_embeds = self.decoder.model.embed_tokens(p_before_tokens.input_ids).expand(batch_size, -1, -1)
            p_after_embeds = self.decoder.model.embed_tokens(p_after_tokens.input_ids).expand(batch_size, -1, -1)
            wrapped_queries = torch.cat((p_before_embeds, queries, p_after_embeds), dim=1)
            wrapped_atts_queries = atts_queries[:, :1].expand(-1, wrapped_queries.shape[1])
        else:
            wrapped_queries, wrapped_atts_queries = queries, atts_queries

        return wrapped_queries, wrapped_atts_queries


    def generate_data_embeds(self, instruction, input, target=None, device='cpu', is_train=True):
        assert len(instruction) == len(input)
        if is_train :
            assert len(instruction) == len(target)
            user_prompt, full_prompt = [], []
            for i in range(len(instruction)):
                user_prompt.append(instruction[i] + input[i] + '###Assistant:')
                full_prompt.append(instruction[i] + input[i] + '###Assistant:' + target[i] + self.end_sym)
            user_prompt_token_ids = self.decoder_tokenizer(user_prompt, padding='longest', add_special_tokens=False, return_tensors='pt').input_ids.to(device)
            full_prompt_token_ids = self.decoder_tokenizer(full_prompt, padding='longest', add_special_tokens=False, return_tensors='pt').input_ids.to(device)
            data_embeds = self.decoder.model.embed_tokens(full_prompt_token_ids)
            atts = torch.ones_like(full_prompt_token_ids).masked_fill(full_prompt_token_ids == self.decoder_tokenizer.pad_token_id, -100)
            targets = full_prompt_token_ids.masked_fill(full_prompt_token_ids == self.decoder_tokenizer.pad_token_id, -100)
            # mask input
            for i in range(full_prompt_token_ids.shape[0]):
                for j, id in enumerate(user_prompt_token_ids[i]):
                    if id != self.decoder_tokenizer.pad_token_id:
                        targets[i][j] = -100
                    else:
                        break
            
            return data_embeds, targets, atts
        else:
            user_prompt = []
            for i in range(len(instruction)):
                user_prompt.append(instruction[i] + input[i] + '###Assistant:')
            user_prompt_token_ids = self.decoder_tokenizer(user_prompt, padding='longest', add_special_tokens=False, return_tensors='pt').input_ids.to(device)
            data_embeds = self.decoder.model.embed_tokens(user_prompt_token_ids)
            atts = torch.ones_like(user_prompt_token_ids).masked_fill(user_prompt_token_ids == self.decoder_tokenizer.pad_token_id, -100)
            
            return data_embeds, atts
        

    def forward(self, data, device):
        self.decoder_tokenizer.padding_side='right'
        patch_images, instruction, input, target = data['img'].half().to(device), data['instruction'], data['input'], data['target']
        # queries_tag = ''.join([self.encoder_tokenizer.mask_token * self.args.query_num])
        encoder_input_text = [instruction[i] + input[i] for i in range(len(instruction))]
        queries, atts_queries = self.get_image_queries(patch_images, encoder_input_text)

        queries = self.query_projection(queries)
        queries, atts_queries = self.prompt_wrap(queries, atts_queries)

        data_embeds, targets, atts = self.generate_data_embeds(instruction, input, target, device)
        # concat data_embeds and queries
        data_embeds = torch.cat((queries, data_embeds), dim=1)
        empty_targets = torch.ones([atts_queries.shape[0], atts_queries.shape[1] + 1], dtype=torch.long).to(device).fill_(-100) # plus one for bos
        targets = torch.cat((empty_targets, targets), dim=1)
        atts = torch.cat((atts_queries, atts), dim=1)
        # add bos
        bos = torch.ones([queries.shape[0], 1], device=device, dtype=torch.long) * self.decoder_tokenizer.bos_token_id
        bos_embeds = self.decoder.model.embed_tokens(bos)
        atts_bos = atts_queries[:, :1]
        data_embeds = torch.cat((bos_embeds, data_embeds), dim=1)
        atts = torch.cat((atts_bos, atts), dim=1)

        loss = self.decoder(inputs_embeds=data_embeds, attention_mask=atts, labels=targets, return_dict=True)['loss']

        return loss


    def generate(self, data, device):
        self.decoder_tokenizer.padding_side='left'
        with torch.no_grad():
            patch_images, instruction, input = data['img'].half().to(device), data['instruction'], data['input']
            # queries_tag = ''.join([self.encoder_tokenizer.mask_token * self.args.query_num])
            encoder_input_text = [instruction[i] + input[i] for i in range(len(instruction))]
            queries, atts_queries = self.get_image_queries(patch_images, encoder_input_text)

        queries = self.query_projection(queries)
        queries, atts_queries = self.prompt_wrap(queries, atts_queries)

        data_embeds, atts = self.generate_data_embeds(instruction, input, target=None, device=device, is_train=False)
        # concat data_embeds and queries
        data_embeds = torch.cat((queries, data_embeds), dim=1)

        atts = torch.cat((atts_queries, atts), dim=1)
        # add bos
        bos = torch.ones([queries.shape[0], 1], device=device, dtype=torch.long) * self.decoder_tokenizer.bos_token_id
        bos_embeds = self.decoder.model.embed_tokens(bos)
        atts_bos = atts_queries[:, :1]
        data_embeds = torch.cat((bos_embeds, data_embeds), dim=1)
        atts = torch.cat((atts_bos, atts), dim=1)

        decoder_inputs = {
            'inputs_embeds' : data_embeds,
            #'attention_mask': atts,
            'max_new_tokens': self.args.max_tgt_len,
            'min_new_tokens': 5
        }
        generate_ids = self.decoder.generate(**decoder_inputs)
        # generate_ids = torch.where(generate_ids < 0, self.decoder_tokenizer.pad_token_id, generate_ids)
        generate_texts = self.decoder_tokenizer.batch_decode(generate_ids, skip_special_tokens=True, clean_up_tokenization_spaces=False)
        # generate_texts = self.batch_decode(generate_ids) # 为什么生成-1的id然后报错？

        return generate_texts


    def generate_with_config(self, data, config, device):
        self.decoder_tokenizer.padding_side='left'
        with torch.no_grad():
            patch_images, instruction, input = data['img'].half().to(device), data['instruction'], data['input']
            queries_tag = ''.join([self.encoder_tokenizer.mask_token * self.args.query_num])
            encoder_input_text = [instruction[i] + input[i] + queries_tag for i in range(len(instruction))]
            queries, atts_queries = self.get_image_queries(patch_images, encoder_input_text)

        queries = self.query_projection(queries)
        queries, atts_queries = self.prompt_wrap(queries, atts_queries)

        data_embeds, atts = self.generate_data_embeds(instruction, input, target=None, device=device, is_train=False)
        # concat data_embeds and queries
        data_embeds = torch.cat((queries, data_embeds), dim=1)

        atts = torch.cat((atts_queries, atts), dim=1)
        # add bos
        bos = torch.ones([queries.shape[0], 1], device=device, dtype=torch.long) * self.decoder_tokenizer.bos_token_id
        bos_embeds = self.decoder.model.embed_tokens(bos)
        atts_bos = atts_queries[:, :1]
        data_embeds = torch.cat((bos_embeds, data_embeds), dim=1)
        atts = torch.cat((atts_bos, atts), dim=1)

        config['inputs_embeds'] = data_embeds
        config['max_new_tokens'] = self.args.max_tgt_len
        generate_ids = self.decoder.generate(**config)
        # generate_ids = torch.where(generate_ids < 0, self.decoder_tokenizer.pad_token_id, generate_ids)
        generate_texts = self.decoder_tokenizer.batch_decode(generate_ids, skip_special_tokens=True, clean_up_tokenization_spaces=False)
        # generate_texts = self.batch_decode(generate_ids) # 为什么生成-1的id然后报错？

        return generate_texts
        

    def get_beit3_hidden_size(self, arch):
        arch = arch.lower().strip()
        arch2hidden_dim = {'base':768, 'large':1024}
        return arch2hidden_dim[arch]


def disabled_train(self, mode=True):
    """Overwrite model.train with this function to make sure train/eval mode
    does not change anymore.
    From MiniGPT-4"""

    return self

def get_beit3_config(arch):
    assert arch in ['base', 'large']
    if arch == 'base':
        return _get_base_config()
    elif arch == 'large':
        return _get_large_config()

def _get_base_config(
        img_size=224, patch_size=16, drop_path_rate=0, 
        checkpoint_activations=None, mlp_ratio=4, vocab_size=64010, **kwargs
):
    return EncoderConfig(
        img_size=img_size, patch_size=patch_size, vocab_size=vocab_size, multiway=True, 
        layernorm_embedding=False, normalize_output=True, no_output_layer=True, 
        drop_path_rate=drop_path_rate, encoder_embed_dim=768, encoder_attention_heads=12, 
        encoder_ffn_embed_dim=int(768 * mlp_ratio), encoder_layers=12, 
        checkpoint_activations=checkpoint_activations, 
    )


def _get_large_config(
        img_size=224, patch_size=16, drop_path_rate=0, 
        checkpoint_activations=None, mlp_ratio=4, vocab_size=64010, **kwargs
):
    return EncoderConfig(
        img_size=img_size, patch_size=patch_size, vocab_size=vocab_size, multiway=True, 
        layernorm_embedding=False, normalize_output=True, no_output_layer=True, 
        drop_path_rate=drop_path_rate, encoder_embed_dim=1024, encoder_attention_heads=16, 
        encoder_ffn_embed_dim=int(1024 * mlp_ratio), encoder_layers=24, 
        checkpoint_activations=checkpoint_activations, 
    )


def prepare_beit3_model(args):
    beit3_config = get_beit3_config(args.arch)
    removed_keys = ['mim_head.weight', 'mim_head.bias', 'mlm_head.weight', 'mlm_head.bias']
    model = BEiT3Wrapper(beit3_config)
    state_dict = torch.load(args.encoder_ckpt)['model']
    for key in removed_keys:
        del state_dict[key]
    model.load_state_dict(state_dict)

    return model.beit3

