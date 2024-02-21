import argparse
from PIL import Image
from torch.utils.data import Dataset


def prepare_args():
    parser = argparse.ArgumentParser(description='Hyperparameters for Multimodal Encoder BLIP2')
    parser.add_argument('--do_train', action='store_true')
    parser.add_argument('--do_eval', action='store_true')
    parser.add_argument("--resume_from_ckpt", action="store_true")
    parser.add_argument("--eval_before_train", action="store_true")
    parser.add_argument('--data_path', type=str, default='./data/VQAv2/')
    parser.add_argument('--vqa_test_choice', type=str, default='val')
    parser.add_argument('--train_batch_size', type=int, default=16)
    parser.add_argument('--eval_batch_size', type=int, default=32)
    parser.add_argument('--train_epoch_num', type=int, default=20)
    parser.add_argument('--gradient_accumulation_num', type=int, default = 1)
    parser.add_argument('--eval_interval_step', type=int, default=1000)
    parser.add_argument('--logging_step', type=int, default=100)
    parser.add_argument('--output_dir', type=str, default='vqav2_output')
    parser.add_argument('--checkpoint', type=str, default='your_checkpoint')
    parser.add_argument("--need_shuffle", type=bool, default=True)
    parser.add_argument("--num_workers", type=int ,default=4)
    parser.add_argument("--train_mask", type=bool, default=True)
    parser.add_argument("--seed",  type=int, default=42)
    # Optimizer Argument
    parser.add_argument("--weight_decay", type=float, default=0.1)
    parser.add_argument("--lr", type=float, default=2e-5)
    parser.add_argument("--warmup_schedule", type=str, default="warmup_cosine",
                        help="Warm up schedule, choose from below: warmup_cosine,warmup_linear,warmup_constant")
    parser.add_argument("--warmup_proportion", type=float, default=0.1,
                        help="Proportion of training to perform linear learning rate warmup for. E.g., 0.1 = 10%% of training.")
    parser.add_argument("--adam_beta1", type=float, default=0.9,
                        help="Bert implementation version of optimizer adam's parameter, beta1")
    parser.add_argument("--adam_beta2", type=float, default=0.98,
                        help="Bert implementation version of optimizer adam's parameter, beta2")
    parser.add_argument("--adam_eps", type=float, default=1e-6,
                        help="Bert implemetation version of optimizer adam's parameter, epslion")
    # Model Argument
    parser.add_argument('--encoder_ckpt', type=str, default='./pretrained_models/OFA-base')
    parser.add_argument('--encoder_type', type=str, default='OFA')
    parser.add_argument('--llama_ckpt', type=str, default='./pretrained_models/vicuna')
    parser.add_argument('--resolution', type=int, default=480)
    parser.add_argument('--query_num', type=int, default=16)
    parser.add_argument('--arch', type=str, default='base')
    parser.add_argument('--max_src_len', type=int, default=128)
    parser.add_argument('--max_tgt_len', type=int, default=20)
    parser.add_argument('--beam_num', type=int, default=3)
    # Distributed Parallel Argument
    parser.add_argument("--local-rank", default=0, type=int)

    return parser.parse_args()