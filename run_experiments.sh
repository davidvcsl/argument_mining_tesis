ATTENTION_MODEL=$1
ATTENTION_ACTIVATION=$2

for i in 1; do
    echo "******************* EXPLORING SETTING $i ***************************"
    ATT_ACT=(relu softmax sigmoid)
    rand_att_act=${ATT_ACT[$[$RANDOM % ${#ATT_ACT[@]}]]}
    echo "Attention activation function" $rand_att_act

    LSTM_UNITS=(30 50 100 200)
    rand_lstm_units=${LSTM_UNITS[$[$RANDOM % ${#LSTM_UNITS[@]}]]}
    echo "LSTM units" $rand_lstm_units

    DROPOUT=(0.1 0.2 0.3 0.4 0.5)
    rand_dropout=${DROPOUT[$[$RANDOM % ${#DROPOUT[@]}]]}
    echo "Dropout" $rand_dropout

    BATCH_SIZE=(16 32 64 128 512)
    rand_batch_size=${BATCH_SIZE[$[$RANDOM % ${#BATCH_SIZE[@]}]]}
    echo "Batch size" $rand_batch_size

    bash train.sh $rand_att_act $rand_dropout $rand_batch_size $rand_lstm_units
done
