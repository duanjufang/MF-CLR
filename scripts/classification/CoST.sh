UEA_config=("ArticularyWordRecognition" "AtrialFibrillation" "BasicMotions" "CharacterTrajectories" "Cricket" "DuckDuckGeese" "EigenWorms" "Epilepsy" "ERing" "EthanolConcentration" "FaceDetection" "FingerMovements" "HandMovementDirection" "Handwriting" "Heartbeat" "InsectWingbeat" "JapaneseVowels" "Libras" "LSST" "MotorImagery" "NATOPS" "PEMS-SF" "PenDigits" "PhonemeSpectra" "RacketSports" "SelfRegulationSCP1" "SelfRegulationSCP2" "SpokenArabicDigits" "StandWalkJump" "UWaveGestureLibrary")


for i in "${UEA_config[@]}"
do

    python -u EXP_CLSF_PUBLIC_DATASETS.py \
        --is_training True \
        --dataset  $i \
        --method CoST \
        --batch_size 32 \
        --epoch 50 \
        --enc_len 128 \
        --use_gpu False \
        
done