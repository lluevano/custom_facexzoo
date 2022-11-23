# CLI Command that crops faces using the eyes annotations (FaceCrop)
# Taken from https://gitlab.idiap.ch/bob/bob.bio.face/-/blob/fb8ffece2423465fdbe6325c75845817d4b53a92/bob/bio/face/script/crop_face_106_landmarks.py
###########################

import os

import cv2
import dask.bag
import numpy as np

from skimage import transform as trans

import bob.bio.face
import bob.io.base

from faceX_106landmarks import FaceX106Landmarks
from bob.extension.scripts.click_helper import ResourceOption
from bob.io.image import bob_to_opencvbgr, opencvbgr_to_bob
from bob.pipelines.distributed import VALID_DASK_CLIENT_STRINGS

# Taken from here: https://github.com/JDAI-CV/FaceX-Zoo/blob/db0b087e4f4d28152e172d6c8d3767a8870733b4/face_sdk/utils/lms_trans.py

lms5_2_lms106 = {1: 105, 2: 106, 3: 55, 4: 85, 5: 91}


def lms106_2_lms5(lms_106):
    lms5 = []
    for cur_point_index in range(5):
        cur_point_id = cur_point_index + 1
        point_id_106 = lms5_2_lms106[cur_point_id]
        cur_point_index_106 = point_id_106 - 1
        cur_point_x = lms_106[cur_point_index_106 * 2]
        cur_point_y = lms_106[cur_point_index_106 * 2 + 1]
        lms5.append(cur_point_x)
        lms5.append(cur_point_y)
    return lms5


# Taken from here: https://github.com/JDAI-CV/FaceX-Zoo/blob/db0b087e4f4d28152e172d6c8d3767a8870733b4/face_sdk/utils/lms_trans.py
def estimate_norm(lmk, image_size=112):

    # Arcface reference points for aligment
    arcface_reference_lmk = np.array(
        [
            [38.2946, 51.6963],
            [73.5318, 51.5014],
            [56.0252, 71.7366],
            [41.5493, 92.3655],
            [70.7299, 92.2041],
        ],
        dtype=np.float32,
    )

    arcface_reference_lmk = np.expand_dims(arcface_reference_lmk, axis=0)

    ################

    assert lmk.shape == (5, 2)
    tform = trans.SimilarityTransform()
    lmk_tran = np.insert(lmk, 2, values=np.ones(5), axis=1)
    min_M = []
    min_index = []
    min_error = float("inf")
    for i in np.arange(arcface_reference_lmk.shape[0]):
        tform.estimate(lmk, arcface_reference_lmk[i])
        M = tform.params[0:2, :]
        results = np.dot(M, lmk_tran.T)
        results = results.T
        error = np.sum(
            np.sqrt(np.sum((results - arcface_reference_lmk[i]) ** 2, axis=1))
        )
        #         print(error)
        if error < min_error:
            min_error = error
            min_M = M
            min_index = i
    return min_M, min_index

def preprocess_insightface(inputs, shape=(112,112), interpolation=cv2.INTER_AREA):
    #bob_img = bob.io.base.load(inputs)
    cv2_img = bob_to_opencvbgr(inputs)

    #1: manage grayscale
    if cv2_img.shape[-1] != 3:
        cv2_img = cv2.cvtColor(cv2_img, cv2.COLOR_GRAY2BGR)

    #2: resize to 112x112, control interpolation
    cv2_resized = cv2.resize(cv2_img, shape, interpolation=interpolation)

    bob_resized = opencvbgr_to_bob(cv2_resized)

    return bob_resized

def faceX_cropper(
    files,
    database_path,
    output_path,
):
    annotator = FaceX106Landmarks(min_size=6, factor=0.79, thresholds=(0.5, 0.5, 0.3))
    image_size = 112
    total_cropped=0
    total_imgs=0
    save_not_aligned = True
    # Load
    for f in files:
        f = f.rstrip("\n")

        output_filename = os.path.join(output_path, f)
        if os.path.exists(output_filename):
            continue

        image = bob.io.base.load(os.path.join(database_path, f))
        total_imgs+=1

        # If it's grayscaled, expand dims
        if image.ndim == 2:
            image = np.repeat(np.expand_dims(image, 0), 3, axis=0)

        # DEtect landmarks

        annot, non_aligned = annotator.annotate(image.copy())

        if annot is None:
            print(f"Face on {f} was not detected. {'Saving non-aligned version' if save_not_aligned else ''}")
            print(output_filename)
            if save_not_aligned:
                non_aligned_output = os.path.join(output_path, f)
                os.makedirs(os.path.dirname(non_aligned_output), exist_ok=True)
                bob.io.base.save(preprocess_insightface(image), non_aligned_output)
        else:
            # if save_not_aligned:
            #     non_aligned -= non_aligned.mean()
            #     non_aligned /= non_aligned.std()
            #     non_aligned *= 64
            #     non_aligned += 128
            #     non_aligned = np.clip(non_aligned.detach().numpy(), 0, 255).astype('uint8')
            #     non_aligned = non_aligned.reshape((3, 112, 112))
            #     non_aligned = cv2.cvtColor(non_aligned.transpose((1,2,0)),cv2.COLOR_BGR2RGB)
            #     non_aligned = non_aligned.transpose((2,0,1))
            #     #print(non_aligned)
            #     non_aligned_output = os.path.join(output_path, f)
            #     os.makedirs(os.path.dirname(non_aligned_output), exist_ok=True)
            #     bob.io.base.save(non_aligned, non_aligned_output)

            annot = annot.flatten()

            landmarks = np.array(lms106_2_lms5(annot))
            landmarks = landmarks.reshape((5, 2))

            M, pose_index = estimate_norm(landmarks, image_size=image_size)

            # bob_to_opencvbgr, opencvbgr_to_bob
            image = bob_to_opencvbgr(image)

            cropped_image = cv2.warpAffine(
                image.copy(), M, (image_size, image_size), borderValue=0.0
            )

            cropped_image = opencvbgr_to_bob(cropped_image)

            #os.makedirs(os.path.dirname(output_filename), exist_ok=True)
            #bob.io.base.save(cropped_image, output_filename)
            total_cropped+=1
            pass
    print(f"Cropped {round(total_cropped/float(total_imgs),2) if total_imgs else 0}% of the total images ({total_cropped}/{total_imgs})")



import click


@click.command()
@click.argument("file_list")
@click.argument("database_path")
@click.argument("output_path")
@click.option(
    "--dask-client",
    "-l",
    entry_point_group="dask.client",
    string_exceptions=VALID_DASK_CLIENT_STRINGS,
    default="single-threaded",
    help="Dask client for the execution of the pipeline.",
    cls=ResourceOption,
)
def crop_faces_faceX(file_list, database_path, output_path, dask_client):

    files = open(file_list).readlines()

    files = dask.bag.from_sequence(files)
    files.map_partitions(
        faceX_cropper,
        database_path,
        output_path,
    ).compute(scheduler=dask_client)

    print("##############################################")
    print("#################### DONE ####################")
    print("##############################################")

    pass


if __name__ == "__main__":
    crop_faces_faceX()
