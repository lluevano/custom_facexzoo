import scipy.io
import os
import numpy as np
from sklearn.metrics import pairwise_distances
from compute_ap import compute_AP

# load the corresponding probe&gallery ids
gallery_match_img_id_pairs = scipy.io.loadmat('gallery_match_img_ID_pairs.mat')
probe_img_ID_pairs = scipy.io.loadmat('probe_img_ID_pairs.mat')

gallery_ids = gallery_match_img_id_pairs['gallery_ids']
probe_ids = probe_img_ID_pairs['probe_ids']

# load your features, please name your feature matrix as
# "gallery_feature_map" & "probe_feature_map" & "distractor_feature_map",
# where each feature map's dimension should be
# [image_number]_by_[feature_dimension]

feature_folder = os.path.join('..')

gallery_feature_map = scipy.io.loadmat(os.path.join(feature_folder,'gallery.mat'))['gallery_feature_map']
probe_feature_map = scipy.io.loadmat(os.path.join(feature_folder,'probe.mat'))['probe_feature_map']
distractor_feature_map = scipy.io.loadmat(os.path.join(feature_folder,'distractor.mat'))['distractor_feature_map']

# gallery/query construction
# building the all the gallery set by combining the match gallery and distractor features

gallery_feature_map = np.concatenate((gallery_feature_map, distractor_feature_map))

# buidling the corresponding gallery ids, all the distractors are labeled as
# -100, while mated gallery are labeled by corresponding ids (to-be-matched to probe)

distractor_ids = -100 * np.ones((np.shape(distractor_feature_map,)[0], 1));
gallery_ids = np.concatenate((gallery_ids, distractor_ids))

dist = pairwise_distances(gallery_feature_map, probe_feature_map, 'euclidean')
CMC = np.zeros((probe_feature_map.shape[0], gallery_ids.shape[0]))
ap = np.zeros((probe_feature_map.shape[0],))

for p in range(probe_feature_map.shape[0]):
    probe_id = probe_ids[p]
    good_index = np.argwhere(gallery_ids == probe_id)[:, 0]
    score = dist[:, p]
    index = np.argsort(score)
    ap_t, CMC_t = compute_AP(good_index, index)
    ap[p] = ap_t
    CMC[p, :] = CMC_t[:]

CMC = np.mean(CMC,axis=0)
mean_ap = np.mean(ap)

print(f"mAP = {mean_ap}, r1 precision = {CMC[0]}, r5 precision = {CMC[4]}, r10 precision = {CMC[9]}, r20 precision = {CMC[19]}, r50 precision = {CMC[49]}")
