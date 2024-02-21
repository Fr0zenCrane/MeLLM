import torch
import torch.nn as nn
from transformers import LlamaTokenizer, LlamaForCausalLM
from ofa.tokenization_ofa import OFATokenizer
from ofa.modeling_ofa import OFAModel


class OFAMultiBLIP(nn.Module):
    def __init__(self, args):
        super(OFAMultiBLIP, self).__init__()
        self.args = args
        # init encoder
        ofa_model = OFAModel.from_pretrained(self.args.ofa_ckpt)
        ofa_model.decoder = None
        self.encoder = ofa_model.encoder
        for name, param in self.encoder:
            param.requires_grad = False
        self.encoder = self.encoder.eval()
        self.encoder.train = disabled_train
        self.encoder_tokenizer = OFATokenizer.from_pretrained(self.args.ofa_ckpt)   
        # init projection layer
        self.query_projection = nn.Linear(self.get_ofa_hidden_size(self.args.arch), 4096) #4096是llama-7b的hidden size
        # init decoder 
        self.decoder = LlamaForCausalLM.from_pretrained(args.llama_ckpt, load_in_8bit=True, torch_dtype=torch.float16)
        self.decoder_tokenizer = LlamaTokenizer.from_pretrained(self.args.llama_ckpt)
        self.decoder_tokenizer.pad_token = self.decoder_tokenizer.eos_token

        for name, param in self.decoder.named_parameters():
            param.requires_grad = False
        self.decoder.train = disabled_train


    def get_image_queries(self, patch_images, encoder_input_text)
        encoder_input_ids = self.encoder_tokenizer(encoder_input_text, padding='longest', truncation=True, max_length=self.args.max_src_len, return_tensors='pt').input_ids.to(device)
        patch_masks = torch.full((encoder_input_ids.shape[0], 1), True).to(patch_images.device)
        encoder_out = self.encoder(encoder_input_ids, patch_images=patch_images, patch_masks=patch_masks)
        queries_idx = torch.where(encoder_input_ids == self.encoder_tokenizer.mask_token_id, 1, 0)
        queries_idx = queries_idx.nonzero()
        queries_idx[:,1] += 900 # 900 is the size of image feature sequence
        queries = encoder_output.last_hidden_state[queries_idx[:,0], queries_idx[:,1]] 
        queries = queries.reshape(-1, self.args.query_num, self.get_ofa_hidden_size(self.args.arch))
        atts_queries = torch.ones(queries.size()[:-1], dtype=torch.long).to(queries.device)

        return queries, atts_queries


    def prompt_wrap(self, queries, atts_queries, prompt):
        if prompt:
            batch_size = queries.shape[0]
            p_before, p_after = prompt.split('<ImageHere>')
            p_before_tokens = self.decoder_tokenizer(p_before, return_tensors='pt', add_special_tokens=False).to(queries.device)
            p_after_tokens = self.decoder_tokenizer(p_after, return_tensors='pt', add_special_tokens=False).to(queries.device)
            p_before_embeds = self.decoder.model.embed_tokens(p_before_tokens.input_ids).expand(batch_size, -1, -1)
            p_after_embeds = self.decoder.model.embed_tokens(p_after_tokens.input_ids).expand(batch_size, -1, -1)
            wrapped_queries = torch.cat([p_before_embeds, queries, p_after_embeds], dim=1)
            wrapped_atts_queries = atts_queries[:, :1].expand(-1, wrapped_queries.shape[1])
        else:
            wrapped_queries, wrapped_atts_queries = queries, atts_queries

        return wrapped_queries, wrapped_atts_queries

    def forward(self, data, device):
        with torch.no_grad():
            patch_images, input, target = data['img'].half().to(device), data['question'], data['main_answer']
            queries_tag = ''.join([self.encoder_tokenizer.mask_token * self.args.query_num])
            encoder_input_text = [queries_tag + question for question in questions]
            queries = self.get_image_queries(patch_images, encoder_input_text)
        queries, atts_queries = self.query_projection(queries)
        prompt = '###Human: <Img><ImageHere></Img> Briefly answer the questions based on the content of the given image:'
        queries, atts_queries = self.prompt_wrap(queries, atts_queries, prompt)
        self.decoder_tokenizer.padding_side = 'right'
        input = [item +' ###Assistant:' for item in input]
        assert len(input) == len(target)
        target = [input[i] + target +'\n' for i in len(target)]
        input_tokens = self.decoder_tokenizer(input, return_tensors='pt', add_special_tokens=False)
        target_tokens = self.decoder_tokenizer(
            target, 
            return_tensors='pt', 
            padding='longest', 
            truncation=True,
            max_length=self.args.max_src_len, 
            add_special_tokens=False)
        labels = 
        decoder_input, labels = self.generate_embeds(data, queries, device)
        loss = self.decoder(inputs_embeds=decoder_input, labels=labels, return_dict=True)['loss']


        return loss

    def generate(self, data, device):
        #patch_images, questions = data['img'].half().to(device), data['question']
        questions = data['question']
        batch_size = len(questions)
        patch_images = data['img'].to(device).half()
        queries_tag = ''.join([self.encoder_tokenizer.mask_token * self.args.query_num])
        encoder_input_text = [queries_tag + question for question in questions]
        encoder_input_ids = self.encoder_tokenizer(encoder_input_text, padding='longest', truncation=True, max_length=self.args.max_src_len, return_tensors='pt').input_ids.to(device)
        patch_masks = torch.full((batch_size, 1), True).to(device)
        encoder_out = self.encoder(encoder_input_ids, patch_images=patch_images, patch_masks=patch_masks)
        queries = self.get_queries(encoder_input_ids, encoder_out)
        queries = self.query_projection(queries)
        decoder_inputs = {
            'past_key_values': None,
            'inputs_embeds' : self.generate_embeds(data, queries, device, train=False),
            'max_new_tokens': self.args.max_tgt_len,
            'min_new_tokens': 1
        }
        retry_cnt = 0 
        while retry_cnt < 5:
            try:
                generate_ids = self.decoder.generate(**decoder_inputs)
                generate_texts = self.decoder_tokenizer.batch_decode(generate_ids, skip_special_tokens=True, clean_up_tokenization_spaces=False)
                break
            except:
                retry_cnt += 1
        if retry_cnt < 5:
            return generate_texts
        else:
            return ['' * batch_size]

    def get_ofa_hidden_size(self, arch):
        arch = arch.lower().strip()
        arch2hidden_dim = {'tiny':256, 'medium':512, 'base':768, 'large':1024, 'huge':1280}
        return arch2hidden_dim[arch]


def disabled_train(self, mode=True):
    """Overwrite model.train with this function to make sure train/eval mode
    does not change anymore.
    From MiniGPT-4"""

    return self