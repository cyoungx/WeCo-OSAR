#!/bin/bash

# ------------------------------
# -------- 大规模数据集 ---------
# ------------------------------

# 生成时间戳
file_path='./results'
experiment_id='XXXX'
timestamp=$(date +"%Y%m%d%H%M%S")

CUDA_VISIBLE_DEVICES=0,1 python main.py --config config/nturgbd120-cross-set/ctrgcn_default.yaml \
                                            --work-dir ${file_path}/${experiment_id}_${timestamp}/work_dir/ctrgcn/ntu120/cset/default \
                                            --device 0 \
                                            --phase train \
                                            --file_path $file_path \
                                            --experiment_id $experiment_id \
                                            --timestamp $timestamp \
                                            --step 25 45 60 \
                                            --num_epoch 60 \
                                            --eval_epoch 55 \
                                            --save_epoch 10 \
                                            --eval_interval 10 \
                                            --run 2 \
                                            --flag_loss_ce \
                                            --flag_loss_align \
                                            --flag_loss_mwcl \
                                            --flag_loss_dwccl \
                                            --num_class 30 \

CUDA_VISIBLE_DEVICES=0,1 python main.py --config config/nturgbd120-cross-set/ctrgcn_default.yaml \
                                            --device 0 \
                                            --phase test \
                                            --file_path $file_path \
                                            --experiment_id $experiment_id \
                                            --timestamp $timestamp \
                                            --weights ${file_path}/${experiment_id}_${timestamp}/work_dir/ctrgcn/ntu120/cset/default/runs-60*.pt \
                                            --weights_velocity ${file_path}/${experiment_id}_${timestamp}/work_dir/ctrgcn/ntu120/cset/default/runs-velocity60*.pt \
                                            --weights_bone ${file_path}/${experiment_id}_${timestamp}/work_dir/ctrgcn/ntu120/cset/default/runs-bone60*.pt \
                                            --num_class 30 \
                                            --run 2 \
