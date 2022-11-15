import glob
import pickle
pickle_jar_dev = "/idiap/temp/lluevano/FaceX-Zoo/training_mode/conventional_training/PDT_ft_tinyface/IResNet50/*/best_dev.pickle"
pickle_jar_eval = "/idiap/temp/lluevano/FaceX-Zoo/training_mode/conventional_training/PDT_ft_tinyface/IResNet50/*/best_dev_eval.pickle"
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
    for i in range(k):
        params = ['head_type', 'lr', 'n_unfrozen_layers']
        str_score =""
        for param in params:
            str_score += f"{scores_sorted[i]['conf'][param]} "
        str_score+=f"{scores_sorted[i]['scores']} "
        str_score += f"{scores_sorted[i]['epoch']} "
        print(str_score)
