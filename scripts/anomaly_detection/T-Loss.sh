python -u EXP_AD_PUBLIC_DATASETS.py \
    --is_training True \
    --dataset  SML \
    --method T-Loss \
    --batch_size 32 \
    --epoch 50 \
    --enc_len 128 \
    --use_gpu False \

python -u EXP_AD_PUBLIC_DATASETS.py \
    --is_training True \
    --dataset  T-Loss \
    --method CoST \
    --batch_size 32 \
    --epoch 50 \
    --enc_len 128 \
    --use_gpu False \


python -u EXP_AD_PUBLIC_DATASETS.py \
    --is_training True \
    --dataset  T-Loss \
    --method CoST \
    --batch_size 32 \
    --epoch 50 \
    --enc_len 128 \
    --use_gpu False \



python -u EXP_AD_PUBLIC_DATASETS.py \
    --is_training True \
    --dataset  SWaT \
    --method T-Loss \
    --batch_size 32 \
    --epoch 50 \
    --enc_len 128 \
    --use_gpu False \