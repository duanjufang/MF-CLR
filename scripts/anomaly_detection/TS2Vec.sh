python -u EXP_AD_PUBLIC_DATASETS.py \
    --is_training True \
    --dataset  SML \
    --method TS2Vec \
    --batch_size 32 \
    --epoch 50 \
    --enc_len 128 \
    --use_gpu False \

python -u EXP_AD_PUBLIC_DATASETS.py \
    --is_training True \
    --dataset  SMD \
    --method TS2Vec \
    --batch_size 32 \
    --epoch 50 \
    --enc_len 128 \
    --use_gpu False \


python -u EXP_AD_PUBLIC_DATASETS.py \
    --is_training True \
    --dataset  SMAP \
    --method TS2Vec \
    --batch_size 32 \
    --epoch 50 \
    --enc_len 128 \
    --use_gpu False \



python -u EXP_AD_PUBLIC_DATASETS.py \
    --is_training True \
    --dataset  SWaT \
    --method TS2Vec \
    --batch_size 32 \
    --epoch 50 \
    --enc_len 128 \
    --use_gpu False \