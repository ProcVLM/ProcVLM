"""
python evqa/tools/add_baseline_metrics_to_tensorboard.py --steps 0 1
"""
import os
import shutil
import argparse
from torch.utils.tensorboard import SummaryWriter

def log_baselines(log_root="./logs/qwen3vl", steps=[0, 1]):
    ######################## CONFIGS ########################
    baselines = { # Modify These Values Based on Your Baseline Results 
        "Baseline_Qwen3-VL-235B": {
            "indomain": {
                "average": 0.5735,
                "verification/MCC": 0.4224,
                "progress/GS": 0.2601,
                "deshuffling/KTS": 0.4629,
                "planning/SA": 0.7936,
                "segmentation/F1@50": 0.6399
            },
            "outdomain": {
                "average": 0.6480,
                "verification/MCC": 0.6027,
                "progress/GS": 0.3425,
                "deshuffling/KTS": 0.5868,
                "planning/SA": 0.8412,
                "segmentation/F1@50": 0.6683
            }
        },
        "Baseline_InternVL3_5-38B": {
            "indomain": {
                "average": 0.3558,
                "verification/MCC": 0.1921,
                "progress/GS": 0.2522,
                "deshuffling/KTS": 0.1481,
                "planning/SA": 0.6779,
                "segmentation/F1@50": 0.1048
            },
            "outdomain": {
                "average": 0.3558,
                "verification/MCC": 0.0772,
                "progress/GS": 0.2851,
                "deshuffling/KTS": 0.1249,
                "planning/SA": 0.7864,
                "segmentation/F1@50": 0.0437
            }
        }
    }
    ######################## END OF CONFIGS ######################

    for model_name, domains in baselines.items():
        log_dir = os.path.join(log_root, model_name)
        if os.path.exists(log_dir):
            print(f"Log directory {log_dir} already exists. Do you want to overwrite it? (y/n)")
            choice = input().lower()
            if choice != 'y':
                print(f"Skipping {model_name}...")
                continue
            else:
                # Remove existing log directory
                shutil.rmtree(log_dir)
        writer = SummaryWriter(log_dir=log_dir)
        
        print(f"Writing logs for {model_name}...")
        
        for domain_name, metrics in domains.items():
            for metric_key, value in metrics.items():
                tag = f"eval/{domain_name}_{metric_key}"
                for step in steps:
                    writer.add_scalar(tag, value, step)
        
        writer.close()
    
    print("Done. Please restart TensorBoard or refresh the page.")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Add baseline metrics to TensorBoard logs.")
    parser.add_argument("--log_root", type=str, default="./logs/qwen3vl", help="Root directory for TensorBoard logs")
    parser.add_argument("--steps", type=int, nargs="+", default=[0, 1], help="Training steps at which to log metrics")
    args = parser.parse_args()
    log_baselines(log_root=args.log_root, steps=args.steps)