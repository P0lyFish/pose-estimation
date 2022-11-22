python train_mpii.py \
    --image-path=mpii_human_pose_v1/images/ \
    --checkpoint=checkpoint/pilot \
    --train-batch=24 \
    --workers=8 \
    --test-batch=24 \
    --cfg configs/baseline_gelu.yml
