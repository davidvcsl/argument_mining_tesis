ATTENTION_MODEL=$1
ATTENTION_ACTIVATION=$2

for i in 1 2 3 4 5 6 7 8 9 10; do
    echo "******************* EXPLORING SETTING $i ***************************"
    ATT_ACT=(relu softmax sigmoid)
    rand_att_act=${ATT_ACT[$[$RANDOM % ${#ATT_ACT[@]}]]}
    echo "Attention activation function" $rand_att_act

    LSTM_UNITS=(50 100 150 200)
    rand_lstm_units=${LSTM_UNITS[$[$RANDOM % ${#LSTM_UNITS[@]}]]}
    echo "LSTM units" $rand_lstm_units

    DROPOUT=(0.1 0.2 0.3 0.4 0.5)
    rand_dropout=${DROPOUT[$[$RANDOM % ${#DROPOUT[@]}]]}
    echo "Dropout" $rand_dropout

    BATCH_SIZE=(8 16 32 64 128 512)
    rand_batch_size=${BATCH_SIZE[$[$RANDOM % ${#BATCH_SIZE[@]}]]}
    echo "Batch size" $rand_batch_size

    bash ../argument_mining_tesis/train.sh $rand_att_act $rand_dropout $rand_batch_size $rand_lstm_units
done
