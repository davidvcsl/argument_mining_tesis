# Script to run the training in all configurations. Must be run from the
# argument mining folder

DATE=$(date +%y-%m-%d-%H-%M)
echo "******** Starting experiment $DATE"

python -u Train_AM.py am_simplest glove \
    --experimentDate $DATE \
    --miniBatchSize $3 \
    --lstmSize $4 \
    --dropout $2 \
    --attentionActivation $1

echo "********** Finished experiment"
