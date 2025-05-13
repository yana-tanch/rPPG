from pathlib import Path

import numpy as np
import rootutils
from dvclive import Live
from omegaconf import OmegaConf

root = rootutils.setup_root(__file__, indicator="src", pythonpath=True)

from src.utils import calculate_fft_hr, make_dir  # noqa


def main(cfg):
    subject_dirs = sorted(Path(cfg.dataset_dir).iterdir())
    log_dir = make_dir("results/pos_baseline")
        
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

            # ground truth
            gt_hr = calculate_fft_hr(gt_bvp, cfg.fps)
            
            # POS prediction
            rgb = faces.mean(axis=(1, 2))
            N = rgb.shape[0]
            bvp = np.zeros(N)
            
            for n in range(N):
                m = n - cfg.win_size

                if m < 0:
                    continue
                
                c = rgb[m:n, :] / np.mean(rgb[m:n, :], axis=0)
                s = np.array([[0, 1, -1], [-2, 1, 1]]) @ c.T

                h = s[0, :] + s[1, :] * np.std(s[0, :]) / np.std(s[1, :])
                h = h - h.mean()

                bvp[m:n] = bvp[m:n] + h

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
