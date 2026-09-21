export WANDB_API_KEY='Your_WandB_API_Key_Here'
export CUDA_VISIBLE_DEVICES=0,1,2,3
torchrun --nproc_per_node=4 --master_port=20119 --master_addr=localhost torchrun_main_HTMuon.py \
    --model_config configs/qwen_600m.json \
    --model_name_or_path Qwen/Qwen3-0.6B \
    --tokenizer_name_or_path Qwen/Qwen3-0.6B \
    --use_hf_model \
    --trust_remote_code \
    --optimizer adamw \
    --seed 5 \
    --lr 0.001 \
    --batch_size 128 \
    --total_batch_size 512 \
    --num_training_steps 5000 \
    --warmup_steps 500 \
    --weight_decay 0.1 \
    --dtype bfloat16 \
    --eval_every 500 \
    --wandb_name 'Your_WandB_Name_Here' \
    --target_eval_tokens 10_000_000 \
    --save_dir outputs/qwen3_600m_adamw \
    --save_every 5000
