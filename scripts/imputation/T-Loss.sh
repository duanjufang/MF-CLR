python -u EXP_IMPE_PUBLIC_DATASETS.py \
    --is_training True \
    --dataset  ETTm1 \
    --method T-Loss \
    --batch_size 32 \
    --epoch 50 \
    --enc_len 128 \
    --use_gpu False \



python -u EXP_IMPE_PUBLIC_DATASETS.py \
    --is_training True \
    --dataset  ETTm2 \
    --method T-Loss \
    --batch_size 32 \
    --epoch 50 \
    --enc_len 128 \
    --use_gpu False \


python -u EXP_IMPE_PUBLIC_DATASETS.py \
    --is_training True \
    --dataset  weather \
    --method T-Loss \
    --batch_size 32 \
    --epoch 50 \
    --enc_len 128 \
    --use_gpu False \



python -u EXP_IMPE_PUBLIC_DATASETS.py \
    --is_training True \
    --dataset  traffic \
    --method T-Loss \
    --batch_size 32 \
    --epoch 50 \
    --enc_len 128 \
    --use_gpu False \