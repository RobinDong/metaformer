DATA_PATH=/root/imagenet
LOCAL_RANK=0

ALL_BATCH_SIZE=4096
NUM_GPU=2
GRAD_ACCUM_STEPS=4 # Adjust according to your GPU numbers and memory size.
BATCH_SIZE=$(expr $ALL_BATCH_SIZE / $NUM_GPU / $GRAD_ACCUM_STEPS )

torchrun --nproc_per_node=2 train.py $DATA_PATH \
--model poolformerv2_s12 \
--opt adamw \
--lr 1e-2 \
--min-lr 1e-6 \
--warmup-epochs 5 \
--batch-size $BATCH_SIZE \
--grad-accum-steps $GRAD_ACCUM_STEPS \
--drop-path 0.0 \
--head-dropout 0.0 \
--workers 64 \
--pin-mem \
--log-interval 1000 \
--epochs 300 \
--weight-decay 0.02
