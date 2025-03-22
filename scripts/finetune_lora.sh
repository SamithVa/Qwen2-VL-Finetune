#!/bin/bash

# You can use 2B instead of 7B
# MODEL_NAME="Qwen/Qwen2-VL-7B-Instruct"
# MODEL_NAME="/data/data1/syc/intern/wanshan/models/Qwen2-VL-2B-Instruct"
MODEL_NAME="/data/data1/syc/intern/wanshan/models/Qwen2-VL-2B-Instruct-Mind2Web"
# MODEL_NAME="Qwen/Qwen2.5-VL-3B-Instruct"
# MODEL_NAME="Qwen/Qwen2.5-VL-7B-Instruct"

export PYTHONPATH=src:$PYTHONPATH

GLOBAL_BATCH_SIZE=64
BATCH_PER_DEVICE=1
NUM_DEVICES=4
GRAD_ACCUM_STEPS=$((GLOBAL_BATCH_SIZE / (BATCH_PER_DEVICE * NUM_DEVICES)))

# If you want to tune the `embed_token` with LoRA, You need to tune `lm_head` together

CUDA_VISIBLE_DEVICES=0,1,2,3 deepspeed src/training/train.py \
    --use_liger True \
    --lora_enable True \
    --use_dora False \
    --lora_namespan_exclude "['lm_head', 'embed_tokens']" \
    --lora_rank 8 \
    --lora_alpha 16 \
    --lora_dropout 0 \
    --num_lora_modules -1 \
    --deepspeed scripts/zero2.json \
    --model_id $MODEL_NAME \
    --data_path /home/syc/intern/wanshan/Qwen2VL-UI-Graph-Finetune/data/mind2web_train_sft_history.json \
    --image_folder /data/data1/syc/intern/wanshan/mind2map_dataset/mind2web_images \
    --remove_unused_columns False \
    --freeze_vision_tower True \
    --freeze_llm True \
    --vision_lora False \
    --tune_merger False \
    --bf16 True \
    --fp16 False \
    --disable_flash_attn2 False \
    --output_dir output/qwen2_vl_0.3_train_vit_from_llamafactory_ckpt \
    --num_train_epochs 3 \
    --per_device_train_batch_size $BATCH_PER_DEVICE \
    --per_device_eval_batch_size 1 \
    --gradient_accumulation_steps $GRAD_ACCUM_STEPS \
    --image_min_pixels $((256 * 28 * 28)) \
    --image_max_pixels $((1280 * 28 * 28)) \
    --learning_rate 5e-5 \
    --merger_lr 1e-5 \
    --vision_lr 2e-6 \
    --weight_decay 0.1 \
    --warmup_ratio 0.03 \
    --lr_scheduler_type "cosine" \
    --logging_steps 5 \
    --tf32 True \
    --gradient_checkpointing True \
    --report_to wandb \
    --lazy_preprocess True \
    --save_strategy "no" \
    --save_steps 100 \
    --save_total_limit 2 \
    --dataloader_num_workers 2 \
    --max_seq_length 2048 \
    --eval_ratio 0.1 \
    --eval_strategy "steps" \
    --eval_step 100 \
    --lm_skip_ratio 0.3 \
    --lm_skip_layer "[1,28,1]" \
    --uigraph_train True \
    --uigraph_test True \
    --uimask_pre True \
    