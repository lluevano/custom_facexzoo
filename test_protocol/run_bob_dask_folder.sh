#SETSHELL grid


source /idiap/home/lluevano/mambaforge-pypy3/bin/activate
conda activate bob_env1


shopt -s nullglob


for out_dir in "/idiap/home/lluevano/my_temp_folder/FaceX-Zoo/training_mode/conventional_training/ft_tinyface_verification2/MobileFaceNet/ArcFace_nf72_lr_0.0001/"; do
  echo "Folder $out_dir"
  python load_facexzoo_model.py --backbone_type MobileFaceNet --out_dir "$out_dir" --eval_set tinyface
done

