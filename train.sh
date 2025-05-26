DATA_PATH=/data/robin/imagenet
LOCAL_RANK=0

ALL_BATCH_SIZE=4096
NUM_GPU=1
GRAD_ACCUM_STEPS=8 # Adjust according to your GPU numbers and memory size.
BATCH_SIZE=$(expr $ALL_BATCH_SIZE / $NUM_GPU / $GRAD_ACCUM_STEPS )

torchrun --nproc_per_node=1 train.py $DATA_PATH \
--model poolformerv2_s12 \
--opt adamw \
--lr 4e-4 \
--warmup-epochs 5 \
--batch-size $BATCH_SIZE \
--grad-accum-steps $GRAD_ACCUM_STEPS \
--drop-path 0.2 \
--head-dropout 0.0 \
--log-interval 2000 \
--epochs 100 \
--start-epoch 0 \
--no-resume-opt \
--resume poolformerv2_s12.pth
