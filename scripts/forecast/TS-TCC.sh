python -u EXP_FCST_PUBLIC_DATASETS.py \
    --is_training True \
    --dataset  ETTm1 \
    --method TS-TCC \
    --batch_size 32 \
    --epoch 50 \
    --enc_len 48 \
    --ot_granu quarterly \
    --use_gpu False \


python -u EXP_FCST_PUBLIC_DATASETS.py \
    --is_training True \
    --dataset  ETTm1 \
    --method TS-TCC \
    --batch_size 32 \
    --epoch 50 \
    --enc_len 48 \
    --ot_granu hourly \
    --use_gpu False \


python -u EXP_FCST_PUBLIC_DATASETS.py \
    --is_training True \
    --dataset  ETTm2 \
    --method TS-TCC \
    --batch_size 32 \
    --epoch 50 \
    --enc_len 48 \
    --ot_granu daily \
    --use_gpu False \


python -u EXP_FCST_PUBLIC_DATASETS.py \
    --is_training True \
    --dataset  ETTm2 \
    --method TS-TCC \
    --batch_size 32 \
    --epoch 50 \
    --enc_len 48 \
    --ot_granu quarterly \
    --use_gpu False \


python -u EXP_FCST_PUBLIC_DATASETS.py \
    --is_training True \
    --dataset  ETTm2 \
    --method TS-TCC \
    --batch_size 32 \
    --epoch 50 \
    --enc_len 48 \
    --ot_granu hourly \
    --use_gpu False \

python -u EXP_FCST_PUBLIC_DATASETS.py \
    --is_training True \
    --dataset  ETTm2 \
    --method TS-TCC \
    --batch_size 32 \
    --epoch 50 \
    --enc_len 48 \
    --ot_granu daily \
    --use_gpu False \

python -u EXP_FCST_PUBLIC_DATASETS.py \
    --is_training True \
    --dataset  weather \
    --method TS-TCC \
    --batch_size 32 \
    --epoch 50 \
    --enc_len 128 \
    --ot_granu 10-minute \
    --use_gpu False \


python -u EXP_FCST_PUBLIC_DATASETS.py \
    --is_training True \
    --dataset  weather \
    --method TS-TCC \
    --batch_size 32 \
    --epoch 50 \
    --enc_len 128 \
    --ot_granu hourly \
    --use_gpu False \

python -u EXP_FCST_PUBLIC_DATASETS.py \
    --is_training True \
    --dataset  weather \
    --method TS-TCC \
    --batch_size 32 \
    --epoch 50 \
    --enc_len 128 \
    --ot_granu daily \
    --use_gpu False \

python -u EXP_FCST_PUBLIC_DATASETS.py \
    --is_training True \
    --dataset  weather \
    --method TS-TCC \
    --batch_size 32 \
    --epoch 50 \
    --enc_len 128 \
    --ot_granu weekly \
    --use_gpu False \


python -u EXP_FCST_PUBLIC_DATASETS.py \
    --is_training True \
    --dataset  traffic \
    --method TS-TCC \
    --batch_size 32 \
    --epoch 50 \
    --enc_len 128 \
    --ot_granu hourly \
    --use_gpu False \

python -u EXP_FCST_PUBLIC_DATASETS.py \
    --is_training True \
    --dataset  traffic \
    --method TS-TCC \
    --batch_size 32 \
    --epoch 50 \
    --enc_len 128 \
    --ot_granu daily \
    --use_gpu False \