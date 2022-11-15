bob bio roc -vvv --eval --output 'mobilefacenet_tinyface_ft.pdf' \
	/idiap/home/lluevano/my_temp_folder/tinyface/finetuned/mobilefacenet/baseline/scores-{dev,eval}.csv \
	/idiap/home/lluevano/my_temp_folder/FaceX-Zoo/training_mode/conventional_training/ft_tinyface_verification/MobileFaceNet/ArcFace_nf72_lr_0.0001/scores-{dev,eval}-3.csv \
	/idiap/home/lluevano/my_temp_folder/FaceX-Zoo/training_mode/conventional_training/ft_tinyface_verification/MobileFaceNet/ContrastiveLoss_nf36_lr_0.0001/scores-{dev,eval}-13.csv \
	/idiap/home/lluevano/my_temp_folder/FaceX-Zoo/training_mode/conventional_training/ft_tinyface_verification/MobileFaceNet/ContrastiveLoss_nf18_lr_0.0001/scores-{dev,eval}-18.csv \
	--legends baseline,AF-72-3-1e-4,CL-36-13-1e-4,CL-18-18-1e-4
