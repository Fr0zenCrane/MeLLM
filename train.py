import os
import torch
import json
import logging
import time
import numpy as np
import files2rouge

from PIL import Image
from pathlib import Path
from utils.utils import prepare_args
from utils.vqa_eval_utils import cal_VQA_metric
from dataset.vqav2_dataset import VQAv2Dataset
from dataset.pretrain_dataset import PretrainDataset
from dataset.mixed_dataset import MixedDataset
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
from modeling import OFAMultiBLIP, BEiT3MultiBLIP
from bert_adam import BertAdam 


def set_seed(seed=42):
    # Set the seed of the entire experiment so results are the same every time we run. 
    # THIS IS SET FOR REPRODUCIBILITY
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed) # Set this if you are using multi-GPU
    
    # When running on the CuDNN backend, two further option must be set
    os.environ['PYTHONHASHSEED'] = str(seed) # 禁止Python的hash随机化，对实验结果的可复现很重要
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True


def prepare_device(logger, local_rank, args):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu", local_rank)
    n_gpus = torch.cuda.device_count()

    return device, n_gpus


def prepare_data(logger, local_rank, args):
    train_dataset = MixedDataset(args, split='train')
    eval_dataset = PretrainDataset(args, split='val') 

    world_size = torch.distributed.get_world_size()
    global_rank = torch.distributed.get_rank()
    train_sampler = torch.utils.data.DistributedSampler(
                                    dataset = train_dataset, 
                                    num_replicas = world_size, 
                                    shuffle = args.need_shuffle,
                                    rank = global_rank, 
                                    seed = args.seed
    )
    train_dataloader = DataLoader(
                        train_dataset,
                        batch_size = args.train_batch_size,
                        shuffle = False,
                        sampler = train_sampler,
                        num_workers = args.num_workers,
                        collate_fn = None
    )
    eval_dataloader = DataLoader(
                        eval_dataset,
                        batch_size = args.eval_batch_size,
                        shuffle = False,
                        num_workers = args.num_workers,
                        collate_fn = None
    )

    return train_dataloader, eval_dataloader, len(train_dataset), len(eval_dataset)


def prepare_model(logger, local_rank, args, device):
    if args.resume_from_ckpt:
        model  = torch.load(args.checkpoint, map_location='cpu')
        if local_rank == 0:
            logger.info("Loading Checkpoint From {}".format(args.checkpoint))
    else:
        if args.encoder_type == 'OFA':
            model = OFAMultiBLIP(args)
        elif args.encoder_type == 'beit3':
            model = BEiT3MultiBLIP(args)
    model = model.to(device).half()    # Move model from cpu to gpu
    model = torch.nn.parallel.DistributedDataParallel(model, device_ids=[local_rank], output_device=local_rank)

    return model


def prepare_optimizer(logger, local_rank, args, model, total_optim_step):
    if hasattr(model, 'module'):
        model = model.module
    param = list(model.named_parameters())
    no_decay = ['bias', 'LayerNorm.bias', 'LayerNorm.weight']
    no_upgrade = ['encoder', 'decoder']
    decay_param = [(n,p) for n, p in param if not any(nd in n for nd in no_decay)]
    no_decay_param = [(n,p) for n, p in param if any(nd in n for nd in no_decay)]
    
    if not args.train_mask:
        optimizer_grouped_param = [
            {'params':[p for n, p in decay_param if not any (nd in n for nd in no_upgrade)], 'weight_decay':args.weight_decay, 'lr': args.lr},
            {'params':[p for n, p in no_decay_param if not any(nd in n for nd in no_upgrade)], 'weight_decay':0.0 , 'lr': args.lr}
        ]
    else:
        optimizer_grouped_param = [
            {'params':[p for n, p in decay_param if 'mask_embed_increment' in n or not any (nd in n for nd in no_upgrade)], 'weight_decay':args.weight_decay, 'lr': args.lr},
            {'params':[p for n, p in no_decay_param if 'mask_embed_increment' in n or not any(nd in n for nd in no_upgrade)], 'weight_decay':0.0 , 'lr': args.lr}
        ]        
    optimizer = BertAdam(optimizer_grouped_param, lr=args.lr, warmup=args.warmup_proportion,
                        schedule=args.warmup_schedule, b1=args.adam_beta1, b2=args.adam_beta2,
                        e=args.adam_eps, t_total=total_optim_step, max_grad_norm=1.0 )
    if local_rank == 0:
        logger.info("** BertAdam Optimizer Setting **")
        logger.info("Learning Rate:{} ".format(args.lr))
        logger.info("Warm-up Proportion:{}".format(args.warmup_proportion))
        logger.info("Warm-up Schedule Strategy:{}".format(args.warmup_schedule))
        logger.info("Adam Beta1:{}".format(args.adam_beta1))
        logger.info("Adam Beta2:{}".format(args.adam_beta2))
        logger.info("Adam Epslion:{}".format(args.adam_eps))
        logger.info("Total Optimization Step to Perform per Epoch:{}".format(total_optim_step))

    return optimizer


def get_logger(args, filename=None):
    # log.txt logger
    logger = logging.getLogger('logger')
    logger.setLevel(logging.DEBUG)
    logging.basicConfig(format='%(asctime)s - %(levelname)s -   %(message)s',
                    datefmt='%m/%d/%Y %H:%M:%S',
                    level=logging.INFO)
    if filename is not None:
        handler = logging.FileHandler(filename)
        handler.setLevel(logging.DEBUG)
        handler.setFormatter(logging.Formatter('%(asctime)s:%(levelname)s: %(message)s'))
        logging.getLogger().addHandler(handler)   
   # tensorboard logger
    tensorboard_writer = SummaryWriter(log_dir=args.output_dir, purge_step=None, max_queue=20, flush_secs=60)
    
    return logger, tensorboard_writer


def train_per_epoch(writer, logger, local_rank, args, epoch, model, optimizer, train_dataloader, eval_dataloader, device):
    model.train()
    total_loss = 0.0
    start_time = time.time()

    for step, data in enumerate(train_dataloader):
        if step  % (args.eval_interval_step * args.gradient_accumulation_num) == 0 and step:
            # eval_acc = evaluate(logger, local_rank, args, model, eval_dataloader, device)
            # if args.local_rank == 0:
            #     logger.info("Eval in Step {:>6d}: Accuarcy:{:.4f}."
            #     .format(epoch * len(train_dataloader) + step, eval_acc))
            #     writer.add_scalar("eval_acc_per_eval_interval", eval_acc, 
            #     global_step= epoch * (len(train_dataloader) / (args.eval_interval_step * args.gradient_accumulation_num)) + step /(args.eval_interval_step * args.gradient_accumulation_num))
            res = evaluate_instruction(logger, local_rank, args, model, eval_dataloader, device)
            if args.local_rank == 0:
                json.dump(res, open(os.path.join(args.output_dir, "{}-step_{}.json".format('pretrain_result', epoch * len(train_dataloader) + step)), mode='w', encoding='utf-8'), indent=4)
                ref = open(os.path.join(args.output_dir, "ref.txt"), mode='w', encoding='utf-8')
                hyp = open(os.path.join(args.output_dir, "hyp.txt"), mode='w', encoding='utf-8')
                for item in res:
                    ref.write(item['target'] + '\n')
                    ans = item['answer'].replace('\n', '').strip()
                    if len(ans) == 0:
                        ans += '.'
                    hyp.write(ans + '\n')
                ref.close()
                hyp.close()
                files2rouge.run(os.path.join(args.output_dir, 'hyp.txt'), os.path.join(args.output_dir, 'ref.txt'))
                

            torch.save(model.module.state_dict(), 
                    os.path.join(args.output_dir, "{}-step_{}.pth".format('pretrain', epoch * len(train_dataloader) + step)))
        loss_step = model(data, device)
        loss_step = loss_step.mean() # to average losses on multi_gpu
        loss_step /= args.gradient_accumulation_num

        loss_step.backward()
        if args.local_rank == 0:
            writer.add_scalar("train_loss_per_step", loss_step.item() * args.gradient_accumulation_num, 
                        global_step = len(train_dataloader) * epoch + step + 1 )
        total_loss += loss_step.item()
        if (step + 1) % args.gradient_accumulation_num == 0:
            optimizer.step()
            optimizer.zero_grad()
        if step % args.logging_step == 0 and local_rank == 0 :
            logger.info("Epoch[{}]:  Step: {}/{}, Lr: {}, Loss: {:.6f}   Averaged_Time_per_Step: {:.4f}".format(
                epoch + 1,
                step + 1, len(train_dataloader),
                "-".join([str('%.9f'%item) for item in sorted(list(set(optimizer.get_lr())))]),
                loss_step.item(),
                (time.time() - start_time) / args.logging_step))
            start_time = time.time()

    total_loss /= len(train_dataloader)
    return total_loss


@torch.no_grad()
def evaluate(logger, local_rank, args, model, eval_dataloader, device):
    model.eval()
    total_eval_loss = 0.0
    start_time = time.time()
    if hasattr(model, 'module'):
        model = model.module
    res = []
    for step, data in enumerate(eval_dataloader):
        target = list(model.generate(data, device))
        for i in range(len(target)):
            res.append({'question_id':int(data['question_id'][i]), 'answer':target[i]})

        if step % args.logging_step == 0 and local_rank == 0 :
            logger.info("Evaluation Procedure:  Step: {}/{}, Averaged_Time_per_Step: {:.4f}".format(
                step + 1, len(eval_dataloader),
                (time.time() - start_time) / args.logging_step))
            start_time = time.time()

    acc = cal_VQA_metric(args, res)
    return acc


@torch.no_grad()
def evaluate_instruction(logger, local_rank, args, model, eval_dataloader, device):
    model.eval()
    total_eval_loss = 0.0
    start_time = time.time()
    if hasattr(model, 'module'):
        model = model.module
    res = []
    for step, data in enumerate(eval_dataloader):
        target = list(model.generate(data, device))
        for i in range(len(target)):
            res.append({'conversation_id':int(data['conversation_id'][i]), 'instruction':data['instruction'][i], 'answer':target[i], 'target':data['target'][i]})

        if step % args.logging_step == 0 and local_rank == 0 :
            logger.info("Evaluation Procedure:  Step: {}/{}, Averaged_Time_per_Step: {:.4f}".format(
                step + 1, len(eval_dataloader),
                (time.time() - start_time) / args.logging_step))
            start_time = time.time()
    
    return res 


def main():
    args = prepare_args()
    Path(args.output_dir).mkdir(parents=True, exist_ok=True)
    logger, writer = get_logger(args, os.path.join(args.output_dir, 'log.txt'))
    torch.distributed.init_process_group("nccl")
    torch.cuda.set_device(args.local_rank)

    # Set Random Seed
    set_seed(args.seed)
    if args.local_rank == 0 :
        logger.info("Setting Seed = {} for Reproducibility".format(args.seed))
    
    # Prepare Device
    device, n_gpus = prepare_device(logger, args.local_rank, args)
    
    # Prepare Dataset
    if args.local_rank == 0:
        logger.info("Preparing Dataset......")
    train_dataloader, eval_dataloader, train_len, eval_len = prepare_data(logger, args.local_rank, args)
    if args.local_rank == 0:
        logger.info("Dataset Prepared, Basic Statistic is as followed:")
        logger.info("****** Train Dataset Statistic ******")
        logger.info("Train Step Num:{}".format(len(train_dataloader)))
        logger.info("Train Example Num:{}".format(train_len))
        logger.info("Train Batch Size:{}".format(args.train_batch_size))
        logger.info("Total Train Batch Size:{}".format(args.train_batch_size * args.gradient_accumulation_num * n_gpus))
        logger.info("****** Eval Dataset Statistic ******")
        logger.info("Eval Step Num:{}".format(len(eval_dataloader)))
        logger.info("Eval Example Num:{}".format(eval_len))
        logger.info("Eval Batch Size:{}".format(args.eval_batch_size))
    
    # Prepare Model
    # TODO: Resume
    if args.local_rank == 0:
        logger.info("Preparing Model......")
    model = prepare_model(logger, args.local_rank, args, device)
    
    # Prepare Optimizer
    if args.local_rank == 0:
        logger.info("Preparing Optimizer......")
    optimizer = prepare_optimizer(logger, args.local_rank, args, model,
                            len(train_dataloader) / args.gradient_accumulation_num)
    if not args.do_train and args.eval_before_train:
        raise ValueError("Can't eval before train while no training procedure will be performed.")
    if args.do_train:
        args.global_step = 0
        if args.eval_before_train:
            if args.local_rank == 0:
                logger.info("Starting Evaluation before Training Procedure.")
            eval_loss_per_epoch, _ = evaluate(logger, args.local_rank, args, model, eval_dataloader, device)
            if args.local_rank == 0:
                writer.add_scalar("eval_loss_per_epoch", eval_loss_per_epoch, global_step=0)            
        # Train Procedure
        for epoch in range(0, args.train_epoch_num):
            loss_per_epoch = train_per_epoch(writer, logger, args.local_rank, args, epoch, model, optimizer, train_dataloader, eval_dataloader, device)
            if args.local_rank == 0 :
                logger.info("--------------------------------------------------------------------")
                logger.info("Epoch {}/{} Completed. Total Loss : {:.6f}".format(epoch + 1, args.train_epoch_num, loss_per_epoch))
                writer.add_scalar("train_loss_per_epoch", loss_per_epoch, global_step=epoch+1)
                if args.do_eval:
                    logger.info("Starting Evaluation after another round for a completed epoch.")
                    res = evaluate_instruction(logger, args.local_rank, args, model, eval_dataloader, device)
                    json.dump(res, open(os.path.join(args.output_dir, "{}-epoch={}.pth".format('pretrain_result', epoch)), mode='w', encoding='utf-8'), indent=4)
                    ref = open(os.path.join(args.output_dir, "ref.txt"), mode='w', encoding='utf-8')
                    hyp = open(os.path.join(args.output_dir, "hyp.txt"), mode='w', encoding='utf-8')
                    for item in res:
                        ref.write(item['target'] + '\n')
                        ans = item['answer'].replace('\n', '').strip()
                        if len(ans) == 0:
                            ans += '.'
                        hyp.write(ans + '\n')
                    ref.close()
                    hyp.close()
                    files2rouge.run(os.path.join(args.output_dir, 'hyp.txt'), os.path.join(args.output_dir, 'ref.txt'))
                    torch.save(model.module, 
                        os.path.join(args.output_dir, "{}-epoch={}.pth".format('pretrain', epoch)))
        # Eval after each Train epoch
            # if args.do_eval:
            #     if args.local_rank == 0:
            #         logger.info("Starting Evaluation after another round for a completed epoch.")
            #     eval_loss_per_epoch, (acc, pre, rec, f1) = \
            #         evaluate(logger, args.local_rank, args, model, eval_dataloader, device)
            #     if args.local_rank == 0 :
            #         logger.info("--------------------------------------------------------------------")
            #         logger.info("Evaluation Completed. Eval Loss:{:.6f}, Accuracy:{:.4f}, Precision:{:.4f}, Recall:{:.4f}, F1:{:.4f}"\
            #             .format(eval_loss_per_epoch, acc, pre, rec, f1))
            #         writer.add_scalar("eval_loss_per_epoch", eval_loss_per_epoch, global_step=epoch+1)
            #         writer.add_scalar("eval_acc_per_epoch", acc, global_step=epoch+1)
            #         writer.add_scalar("eval_pre_per_epoch", pre, global_step=epoch+1)
            #         writer.add_scalar("eval_rec_per_epoch", rec, global_step=epoch+1)
            #         writer.add_scalar("eval_f1_per_epoch", f1, global_step=epoch+1)
            #     torch.save(model.module, 
            #         os.path.join(args.output_dir, args.bert_config.split('/')[-1], "{}-epoch={}.pth".format(args.bert_config.split('/')[-1], epoch)))
    

    if args.local_rank == 0:
        logger.info("Complete Train/Eval, Exiting...")
        writer.close()


if __name__ == "__main__":
    main()