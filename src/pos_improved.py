from pathlib import Path

import numpy as np
import rootutils
from dvclive import Live
from omegaconf import OmegaConf

root = rootutils.setup_root(__file__, indicator="src", pythonpath=True)

from src.utils import calculate_fft_hr, make_dir  # noqa


def main(cfg):
    subject_dirs = sorted(Path(cfg.dataset_dir).iterdir())
    log_dir = make_dir("results/pos_improved")
        
    with Live(log_dir, save_dvc_exp=False) as live:
        params = OmegaConf.to_container(cfg, resolve=True)
        live.log_params(params)

        mae_errors = []
        pad = cfg.pad_size
        
        for subject_dir in subject_dirs:
            live.step = int(subject_dir.stem)

            faces = np.load(subject_dir / "faces.npz")["faces"]
            gt_bvp = np.loadtxt(subject_dir / "gt_bvp.txt")

            faces = faces[:, pad:-pad, pad:-pad, :]

            # ground truth HR
            gt_hr = calculate_fft_hr(gt_bvp, cfg.fps)
            
            # improved POS
            rgb = faces.mean(axis=(1, 2))

            N = rgb.shape[0]
            C = np.empty((0, 3))

            # statistics gathering
            for n in range(N):
                m = n - cfg.win_size

                if m < 0:
                    continue

                c = rgb[m:n, :] / np.mean(rgb[m:n, :], axis=0)
                C = np.vstack((C, c))

            cov_matrix = C.T @ C / len(C)
            _, eigenvectors = np.linalg.eig(cov_matrix)
            e1, e2 = eigenvectors[:, 1], eigenvectors[:, 2]

            bvp = np.zeros(N)

            # prediction
            for n in range(N):
                m = n - cfg.win_size

                if m < 0:
                    continue
                
                c = rgb[m:n, :] / np.mean(rgb[m:n, :], axis=0)

                y1 = e1 @ c.T
                y2 = e2 @ c.T

                y1 = (y1 - y1.mean()) / y1.std()
                y2 = (y2 - y2.mean()) / y2.std()

                bvp[m:n] = bvp[m:n] + y1 + y2

            hr = calculate_fft_hr(bvp, cfg.fps)

            # logging
            mae = np.abs(hr - gt_hr) 
            mae_errors.append(mae)

            print(f"FFT MAE {subject_dir}: {mae}")

            live.log_metric("MAE", mae)
            live.next_step()

        mean_mae = np.array(mae_errors).mean()

        print(f"Mean FFT MAE {mean_mae:.4f}")
        live.log_metric("MEAN_MAE", mean_mae)


if __name__ == "__main__":
    cfg = OmegaConf.load("params.yaml").pos
    main(cfg)
