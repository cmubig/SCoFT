export MODEL_NAME="CompVis/stable-diffusion-v1-4"
export TRAIN_DIR="/scratch/zhixuan2/m3c/culture_data/cvpr/china"

accelerate launch --mixed_precision="fp16"  ../train.py \
  --pretrained_model_name_or_path=$MODEL_NAME \
  --train_data_dir=$TRAIN_DIR \
  --caption_column="text" \
  --blip_caption_column="blip_text" \
  --negative_example_column="negative_imgpath" \
  --dataloader_num_workers=8 \
  --resolution=512 --center_crop --random_flip \
  --train_batch_size=1 \
  --gradient_accumulation_steps=4 \
  --num_train_epochs=100 \
  --learning_rate=1e-04 \
  --max_grad_norm=1 \
  --lr_scheduler="cosine" --lr_warmup_steps=0 \
  --output_dir="../output/scoft-Chinese-culture" \
  --checkpointing_steps=1000 \
  --validation_prompt="family is eating together in China" \
  --perceptualloss \
  --dreamsimloss \
  --recordfirstgradient \
  --rank=64 \
  --seed=1024 \
  --report_to=wandb