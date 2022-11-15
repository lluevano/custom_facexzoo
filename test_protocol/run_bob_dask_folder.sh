#!/bin/bash

shopt -s nullglob
for out_dir in "/idiap/temp/lluevano/FaceX-Zoo/training_mode/conventional_training/ft_survface_verification/MobileFaceNet/"*"/"; do
  echo "Folder $out_dir"
  python load_facexzoo_model.py --backbone_type MobileFaceNet --out_dir "$out_dir" --eval_set survface
done


