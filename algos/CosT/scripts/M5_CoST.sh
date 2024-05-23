for level in $(seq 1 10); do
  for seed in $(seq 0 4); do
    # multivar
    python -u train.py M5-l${level} forecast_multivar --alpha 0.0005 --kernels 1 2 4 8 16 32 64 128 --max-train-length 201 --batch-size 128 --archive forecast_csv --repr-dims 320 --max-threads 8 --seed ${seed} --eval
    # univar
    python -u train.py M5-l${level} forecast_univar --alpha 0.0005 --kernels 1 2 4 8 16 32 64 128 --max-train-length 201 --batch-size 128 --archive forecast_csv_univar --repr-dims 320 --max-threads 8 --seed ${seed} --eval
  done
done