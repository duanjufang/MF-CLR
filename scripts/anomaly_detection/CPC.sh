python -u EXP_AD_PUBLIC_DATASETS.py \
    --is_training True \
    --dataset  SML \
    --method CPC \
    --batch_size 32 \
    --epoch 50 \
    --enc_len 32 \
    --use_gpu False \

python -u EXP_AD_PUBLIC_DATASETS.py \
    --is_training True \
    --dataset  SMD \
    --method CPC  \
    --batch_size 32 \
    --epoch 50 \
    --enc_len 32 \
    --use_gpu False \


python -u EXP_AD_PUBLIC_DATASETS.py \
    --is_training True \
    --dataset  SMAP \
    --method CPC \
    --batch_size 32 \
    --epoch 50 \
    --enc_len 32 \
    --use_gpu False \



python -u EXP_AD_PUBLIC_DATASETS.py \
    --is_training True \
    --dataset  SWaT \
    --method CPC  \
    --batch_size 32 \
    --epoch 50 \
    --enc_len 32 \
    --use_gpu False \