accelerate launch sft_correctionlm.py \
    --model_name="meta-llama/Meta-Llama-3-8B-Instruct" \
    --output_dir="models/mwoz/sft_llama3_on_train5p_zeroshot/" \
    --train_data_file="data/llama3_on_train5p_zeroshot_ICL_prompt.json" \
    --valid_data_file="data/llama3_on_train5p_zeroshot_ICL_prompt.json" \
    --do_train \
    --do_eval \
    --evaluation_strategy="steps" \
    --eval_steps=1000 \
    --max_steps=5000 \
    --logging_steps=1000 \
    --save_steps=1000 \
    --per_device_train_batch_size=1 \
    --per_device_eval_batch_size=4 \
    --gradient_accumulation_steps=1 \
    --gradient_checkpointing=False \
    --group_by_length=False \
    --lr_scheduler_type="cosine" \
    --warmup_steps=100 \
    --learning_rate=3e-5 \
    --weight_decay=0.1 \
    --max_grad_norm=0.3 \
    --bf16=True \
    --optim="paged_adamw_32bit" \
    --remove_unused_columns=True \
    --run_name="sft_llama3_on_train5p_zeroshot" \
    --WANDB_PROJECT="sft_llama_mwoz_correction" \
    --report_to="wandb" 