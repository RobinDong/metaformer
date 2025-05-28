DATA_PATH=/data/cifar100/
LOCAL_RANK=0

ALL_BATCH_SIZE=8192
NUM_GPU=1
GRAD_ACCUM_STEPS=1 # Adjust according to your GPU numbers and memory size.
BATCH_SIZE=$(expr $ALL_BATCH_SIZE / $NUM_GPU / $GRAD_ACCUM_STEPS )

torchrun --nproc_per_node=1 train.py $DATA_PATH \
--dataset cifar100 \
--input-size 3 56 56 \
--val-split test \
--model poolformerv2_s12 \
--opt adamw \
--lr 1e-3 \
--min-lr 1e-7 \
--warmup-epochs 0 \
--batch-size $BATCH_SIZE \
--grad-accum-steps $GRAD_ACCUM_STEPS \
--drop-path 0.0 \
--head-dropout 0.0 \
--epochs 100 \
--start-epoch 0 \
--no-resume-opt \
--resume poolformerv2_s12.pth
