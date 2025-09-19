import os
import subprocess
import argparse

CHECKPOINTS_ROOT = '/hdd/jcaicedo/projects/ICLR_Benchmarking/benchmarks'
FEATURES_ROOT    = '/hdd/jcaicedo/projects/ICLR_Benchmarking/benchmarks/features'

def get_checkpoints(base_dir: str):
    checkpoints = []
    for dirpath, dirnames, filenames in os.walk(base_dir):
        if len(filenames) > 0 and len(dirnames) == 0:
            model = dirpath.split('/')[-2]
            for filename in filenames:
                if filename.endswith('.pth') or filename.endswith('.pt'):
                    if model == 'simclr':
                        epoch = filename.split('_')[-1][:-3]
                        if int(epoch) % 5 != 0:
                            continue
                    elif model == 'mae':
                        if 'latest' in filename:
                            continue
                        if '_' in filename:
                            epoch = filename.split('_')[-1][:-4]
                            if int(epoch) % 5 != 0:
                                continue
                        elif '-' in filename:
                            epoch = filename.split('-')[-1][:-4]
                            if int(epoch) % 5 != 0:
                                continue
                    
                    files = os.listdir(dirpath)
                    small = list(filter(lambda x: x=='small.txt', files))
                    base = list(filter(lambda x: x=='base.txt', files))
                    large = list(filter(lambda x: x=='large.txt', files))
                    if len(small) > 0:
                        size = "small"
                    elif len(base) > 0:
                        size = "small"
                    elif len(large) > 0:
                        size = "small"
                    checkpoints.append((os.path.join(dirpath, filename), model, size, filename.split('.')[0], os.path.basename(dirpath)))

    return checkpoints

def main(dry_run: bool, base_dir: str):
    base_dir = os.path.abspath(os.path.expanduser(base_dir))   
    checkpoints_to_eval = get_checkpoints(base_dir)
    for (checkpoint, model, size, check_dir, model_name) in checkpoints_to_eval:   
        wandb_key =  os.environ.get('WANDB_API_KEY')
        
        output_dir = os.path.join(base_dir, 'features', model, model_name, check_dir)
        
        if "CHANViT" in model_name:
            if model == 'simclr':
                model_name = 'chanvit_simclr'
            elif model == 'mae':
                model_name = 'chanvit_mae'
        elif 'DINO' in model_name:
            model_name = 'vit'
        elif '0f6fa51' in model_name or '6c1aa1e' in model_name:
            model_name = 'channel_vit'
        elif 'SimCLR' in model_name:
            model_name = 'simclr'
        elif 'MAE100' in model_name:
            model_name = 'mae'
        
        if dry_run:
            print(check_dir, model_name)
        else:
            submit_cmd = f"condor_submit -i model_type={model_name} model_size={size} model_path={checkpoint} feature_out={output_dir} condor_eval.sh"
            result = subprocess.run(
                        submit_cmd,
                        shell=True,
                        # capture_output=True,
                        check=True,
                        text=True
                    )
            break

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("-d", "--dry-run", action="store_true", help="Perform a dry run without making actual changes")
    parser.add_argument("-b", "--base-dir", required=True)
    args = parser.parse_args()
    
    main(args.dry_run, args.base_dir)
