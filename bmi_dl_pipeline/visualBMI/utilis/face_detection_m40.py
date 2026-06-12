
import os
import cv2
import numpy as np
from PIL import Image
from facenet_pytorch import MTCNN
import torch
from tqdm import tqdm
import pandas as pd


def get_mtcnn_m40(device):
    return MTCNN(
        image_size   = 224,
        margin       = 40,        # increased from 20
        min_face_size= 40,
        thresholds   = [0.6, 0.7, 0.7],
        factor       = 0.709,
        keep_all     = False,
        device       = device
    )


def crop_and_save_face_m40(img_path, save_path, mtcnn):
    try:
        img  = Image.open(img_path).convert('RGB')
        face = mtcnn(img)

        if face is None:
            return False

        face_np  = face.permute(1, 2, 0).numpy()
        face_np  = ((face_np - face_np.min()) /
                    (face_np.max() - face_np.min()) * 255).astype(np.uint8)
        face_bgr = cv2.cvtColor(face_np, cv2.COLOR_RGB2BGR)
        cv2.imwrite(save_path, face_bgr)
        return True

    except Exception:
        return False


def process_all_faces_m40(df, crops_dir, device):
    mtcnn   = get_mtcnn_m40(device)
    success, failed, records = [], [], []

    for _, row in tqdm(df.iterrows(), total=len(df),
                       desc="Detecting faces (margin=40)"):
        img_path  = row['img_path']
        person_id = row['person_id']
        bmi       = row['bmi']

        img_name  = os.path.basename(img_path).replace(
                        '.jpg', '_m40_crop.jpg')
        save_path = os.path.join(crops_dir, img_name)

        if os.path.exists(save_path):
            records.append({'crop_path': save_path,
                            'person_id': person_id,
                            'bmi':       bmi})
            success.append(img_path)
            continue

        ok = crop_and_save_face_m40(img_path, save_path, mtcnn)

        if ok:
            records.append({'crop_path': save_path,
                            'person_id': person_id,
                            'bmi':       bmi})
            success.append(img_path)
        else:
            failed.append(img_path)

    print(f"\n✅ Face detection (margin=40) complete")
    print(f"   Success : {len(success)}")
    print(f"   Failed  : {len(failed)}")
    print(f"   Rate    : {len(success)/len(df)*100:.1f}%")

    if failed:
        pd.DataFrame({'failed_path': failed}).to_csv(
            '/kaggle/working/results/metrics_m40/'
            'failed_detections_m40.csv', index=False
        )

    return pd.DataFrame(records)
