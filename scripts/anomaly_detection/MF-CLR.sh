python -u EXP_AD_PUBLIC_DATASETS.py \
    --is_training True \
    --dataset  SML \
    --method MF-CLR \
    --batch_size 32 \
    --epoch 50 \
    --enc_len 128 \
    --use_gpu False \

python -u EXP_AD_PUBLIC_DATASETS.py \
    --is_training True \
    --dataset  SMD \
    --method MF-CLR \
    --batch_size 32 \
    --epoch 50 \
    --enc_len 128 \
    --use_gpu False \


python -u EXP_AD_PUBLIC_DATASETS.py \
    --is_training True \
    --dataset  SMAP \
    --method MF-CLR \
    --batch_size 32 \
    --epoch 50 \
    --enc_len 128 \
    --use_gpu False \



python -u EXP_AD_PUBLIC_DATASETS.py \
    --is_training True \
    --dataset  SWaT \
    --method MF-CLR \
    --batch_size 32 \
    --epoch 50 \
    --enc_len 128 \
    --use_gpu False \