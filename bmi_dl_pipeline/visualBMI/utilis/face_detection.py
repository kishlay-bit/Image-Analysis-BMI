
import os
import cv2
import numpy as np
from PIL import Image
from facenet_pytorch import MTCNN
import torch
from tqdm import tqdm
import pandas as pd

def get_mtcnn(device):
    return MTCNN(
        image_size=224,
        margin=20,
        min_face_size=40,
        thresholds=[0.6, 0.7, 0.7],
        factor=0.709,
        keep_all=False,
        device=device
    )

def crop_and_save_face(img_path, save_path, mtcnn):
    """
    Detect, align and save face crop.
    Returns True if successful, False if no face detected.
    """
    try:
        img = Image.open(img_path).convert('RGB')
        face = mtcnn(img)

        if face is None:
            return False

        # Convert tensor to numpy for saving
        face_np = face.permute(1, 2, 0).numpy()
        face_np = ((face_np - face_np.min()) /
                   (face_np.max() - face_np.min()) * 255).astype(np.uint8)
        face_bgr = cv2.cvtColor(face_np, cv2.COLOR_RGB2BGR)
        cv2.imwrite(save_path, face_bgr)
        return True

    except Exception as e:
        return False


def process_all_faces(df, raw_base, crops_dir, device):
    """
    Run MTCNN on all images, save crops, return clean dataframe.
    """
    mtcnn = get_mtcnn(device)
    
    success, failed = [], []
    records = []

    for _, row in tqdm(df.iterrows(), total=len(df), desc="Detecting faces"):
        img_path  = row['img_path']
        person_id = row['person_id']
        bmi       = row['bmi']

        # Unique save name per image
        img_name  = os.path.basename(img_path).replace('.jpg', '_crop.jpg')
        save_path = os.path.join(crops_dir, img_name)

        # Skip if already processed
        if os.path.exists(save_path):
            records.append({
                'crop_path': save_path,
                'person_id': person_id,
                'bmi':       bmi
            })
            success.append(img_path)
            continue

        ok = crop_and_save_face(img_path, save_path, mtcnn)

        if ok:
            records.append({
                'crop_path': save_path,
                'person_id': person_id,
                'bmi':       bmi
            })
            success.append(img_path)
        else:
            failed.append(img_path)

    print(f"\n✅ Face detection complete")
    print(f"   Success : {len(success)}")
    print(f"   Failed  : {len(failed)}")
    print(f"   Rate    : {len(success)/len(df)*100:.1f}%")

    if failed:
        pd.DataFrame({'failed_path': failed}).to_csv(
            '/kaggle/working/results/metrics/failed_detections.csv', index=False
        )

    return pd.DataFrame(records)
