#!/bin/bash

shopt -s nullglob

while true
do
    for out_dir in "/idiap/temp/lluevano/FaceX-Zoo/training_mode/conventional_training/ft_survface_verification/IResNet50/ArcFace_nf72_lr_0.001/"; do
      echo "Folder $out_dir"
      python load_facexzoo_model.py --backbone_type IResNet --out_dir "$out_dir" --eval_set survface
    done
done
