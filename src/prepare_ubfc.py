from pathlib import Path

import cv2
import imageio.v3 as iio
import mediapipe as mp
import numpy as np
import rootutils
from omegaconf import OmegaConf

root = rootutils.setup_root(__file__, indicator="src", pythonpath=True)

from src.utils import make_dir  # noqa


# прочитаем BVP (Blood Volume Pulse) данные
def read_bvp(bvp_file):
    with open(bvp_file, "r") as f:
        s = f.read()
        s = s.split("\n")
        bvp = [float(x) for x in s[0].split()]

    return np.asarray(bvp)


def main(cfg):
    src_data_dir = Path(cfg.src_data_dir)
    video_files = {int(f.parent.stem[7:]): f for f in src_data_dir.glob("**/*.avi")}

    dst_data_dir = make_dir(cfg.dst_data_dir)

    for pid, src_video_file in video_files.items():
        subject_dir = make_dir(dst_data_dir / str(pid).zfill(2))
        print(f"{src_video_file} -> {subject_dir}")

        # читаем видео
        frames = iio.imread(src_video_file, plugin="pyav")
        gt_bvp = read_bvp(src_video_file.parent / "ground_truth.txt")

        num_frames, img_height, img_width = frames.shape[:3]

        # детектируем лицо на видео с помощью MediaPipe
        mp_face_detection = mp.solutions.face_detection
        face_bboxes = np.zeros((num_frames, 4), dtype=int)

        with mp_face_detection.FaceDetection(model_selection=1, min_detection_confidence=0.5) as face_detection:
            for n, frame in enumerate(frames):
                results = face_detection.process(frame)
                
                if not results.detections:
                    raise RuntimeError(f"frame: {n} face not detected")
                    
                for detection in results.detections:
                    box = detection.location_data.relative_bounding_box
                    x, y, w, h = map(int, [box.xmin * img_width, box.ymin * img_height, box.width * img_width, box.height * img_height])
                    face_bboxes[n] = x, y, x + w, y + h

        # записываем ground truth BVP
        np.savetxt(subject_dir / "gt_bvp.txt", gt_bvp, fmt="%.4f")

        #  записываем видео
        w = h = cfg.image_size
        faces = np.zeros((num_frames, h, w, 3), dtype=np.uint8)

        for n, (frame, (x1, y1, x2, y2)) in enumerate(zip(frames, face_bboxes)):
            crop = frame[y1: y2, x1: x2]
            faces[n] = cv2.resize(crop, (w, h), interpolation=cv2.INTER_AREA)

        np.savez(subject_dir / "faces.npz", faces=faces)
        

if __name__ == "__main__":
    cfg = OmegaConf.load("params.yaml").prepare_ubfc
    main(cfg)
