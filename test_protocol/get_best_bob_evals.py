import glob
import pickle
pickle_jar_dev = "/idiap/home/lluevano/my_temp_folder/FaceX-Zoo/training_mode/conventional_training/pretrain_emore_RRDBNet-prelu_SR_msceleb_ft_tinyface/MobileFaceNet/*/best_dev.pickle"
pickle_jar_eval = "/idiap/home/lluevano/my_temp_folder/FaceX-Zoo/training_mode/conventional_training/pretrain_emore_RRDBNet-prelu_SR_msceleb_ft_tinyface/MobileFaceNet/*/*_eval.pickle"
k = 5

for pickle_jar in [('dev',pickle_jar_dev), ('eval',pickle_jar_eval)]:
    pickle_iter = glob.iglob(pickle_jar[1])

    all_scores = []
    for pickle_path in pickle_iter:
        with open(pickle_path,'rb') as handle:
            saved_scores = pickle.load(handle)
            all_scores.append(saved_scores)

    scores_sorted = sorted(all_scores,key=lambda x: x['scores'][0]['EER'])
    print(f"Min-{k} {pickle_jar[0]} EER scores")
    print("| Loss | LR | #train layers | epoch | R1 | EER | AUC |")
    for i in range(k):
        params = ['head_type', 'lr', 'n_unfrozen_layers']
        str_score ="| "
        for param in params:
            str_score += f"{scores_sorted[i]['conf'][param]} | "
        str_score += f"{scores_sorted[i]['epoch']} | "
        scores_arr = scores_sorted[i]['scores']
        for _, v in scores_arr[0].items():
            str_score+=f"{round(v,2)} | "
        print(str_score)
