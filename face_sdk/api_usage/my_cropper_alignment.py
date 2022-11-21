import sys
sys.path.append('.')

import yaml
import cv2
import numpy as np
from face_sdk.core.model_loader.face_detection.FaceDetModelLoader import FaceDetModelLoader
from face_sdk.core.model_handler.face_detection.FaceDetModelHandler import FaceDetModelHandler
from face_sdk.core.model_loader.face_alignment.FaceAlignModelLoader import FaceAlignModelLoader
from face_sdk.core.model_handler.face_alignment.FaceAlignModelHandler import FaceAlignModelHandler
from face_sdk.core.image_cropper.arcface_cropper.FaceRecImageCropper import FaceRecImageCropper

import os

import pandas as pd

def crop_and_align_faces(faceDetModelHandler, faceAlignModelHandler, face_cropper, DATA_ROOT):

    BOB_TRAIN_WORLD = "./train_world_scface.csv"

    train_df = pd.read_csv(filepath_or_buffer=BOB_TRAIN_WORLD)

    for img_to_load in train_df.itertuples(index=False):
        img = cv2.imread(os.path.join(DATA_ROOT, img_to_load.PATH))
        # handle grayscale if any
        if img.shape[-1] != 3:
            img = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)

        dets = faceDetModelHandler.inference_on_image(img)
        # face_nums = dets.shape[0]
        landmarks = faceAlignModelHandler.inference_on_image(img, dets[0])
        landmarks_list = []
        for (x, y) in landmarks.astype(np.int32):
            landmarks_list.extend((x, y))
        cropped_image = face_cropper.crop_image_by_mat(img, landmarks_list)
        cropped_image

def main():
    # common setting for all models, need not modify.
    DATA_ROOT = "/idiap/resource/database/scface/SCface_database/"

    model_path = '../models'

    with open('../config/model_conf.yaml') as f:
        model_conf = yaml.load(f, Loader=yaml.FullLoader)

    # face detection model setting.
    scene = 'non-mask'
    model_category = 'face_detection'
    model_name = model_conf[scene][model_category]
    #logger.info('Start to load the face detection model...')
    try:
        faceDetModelLoader = FaceDetModelLoader(model_path, model_category, model_name)
        model, cfg = faceDetModelLoader.load_model()
        faceDetModelHandler = FaceDetModelHandler(model, 'cpu', cfg)
    except Exception as e:
        #logger.error('Falied to load face detection Model.')
        #logger.error(e)
        sys.exit(-1)


    # face landmark model setting.
    model_category = 'face_alignment'
    model_name = model_conf[scene][model_category]
    #logger.info('Start to load the face landmark model...')
    try:
        faceAlignModelLoader = FaceAlignModelLoader(model_path, model_category, model_name)
        model, cfg = faceAlignModelLoader.load_model()
        faceAlignModelHandler = FaceAlignModelHandler(model, 'cpu', cfg)
    except Exception as e:
        #logger.error('Failed to load face landmark model.')
        #logger.error(e)
        sys.exit(-1)

    face_cropper = FaceRecImageCropper()

    crop_and_align_faces(faceDetModelHandler, faceAlignModelHandler, face_cropper, DATA_ROOT)


main()
