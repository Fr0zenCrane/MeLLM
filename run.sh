export CUDA_VISIBLE_DEVICES=4,5,6,7
export OMP_NUM_THREADS=16
python -m torch.distributed.launch --nproc_per_node 4  --master_port='12134' train.py \
--do_train \
--do_eval \
--train_epoch_num 5 \
--encoder_type OFA \
--arch huge \
--encoder_ckpt ./pretrained_models/OFA-huge \
--llama_ckpt ./pretrained_models/vicuna-7b-v1.3 \
--output_dir output_ofa_vicuna \
--lr 2e-5 \
--do_eval \
--train_mask false \
--vqa_test_choice dev \
--train_batch_size 1 \
--gradient_accumulation_num  1 \
--eval_batch_size 16 \
--query_num 16 \
--max_src_len 512 \
--max_tgt_len 128 \
--logging_step 10 \
--eval_interval_step 500 \
