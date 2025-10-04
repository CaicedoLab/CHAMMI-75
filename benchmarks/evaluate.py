"""
Master file to run all evaluations at once
Run the commands using a config file (benchmark_config.yaml)
"""

import yaml
import subprocess
import os

# Function to write dinov2_idr17.yaml from benchmark_config.yaml
def write_dinov2_config(config, output_path):
    dinov2_config = {
        'out_folder': config['IDR_DATA_FOLDER'] + '/study_replication',
        'cache': config['IDR_CACHE'],
        'study': config['IDR_DATA_FOLDER'] + '/images',
        'metadata': config['IDR_DATA_FOLDER'] + '/metadata/idr0017_meta.csv',
        'feature_extraction': {
            'feature_agg': True,
            'model_mode': config.get('MODEL_MODE', None),
            'model_path': config.get('MODEL_PATH', None),
            'model': config.get('MODEL_TYPE', 'dinov2'),
            'subcell_channel_map': {
                'nucleus': 1,
                'protein': 2,
                'er': None,
                'mt': None
            },
            'crop': config.get('IDR_CROP', 100),
            'resize': config.get('IDR_RESIZE', 224),
            'resources': {
                'num_gpus': config.get('NUM_GPUS', 1),
                'batch_size': config.get('IDR_BATCH_SIZE', 2048),
                'num_workers': config.get('IDR_NUM_WORKERS', 8),
                'threads': config.get('IDR_THREADS', 16)
            }
        }
    }
    with open(output_path, 'w') as f:
        yaml.dump(dinov2_config, f, default_flow_style=False)

CONFIG_PATH = os.path.join(os.path.dirname(__file__), 'benchmark_config.yaml')
BENCHMARKS_DIR = os.path.dirname(__file__)

# Helper to run a command in a specific folder
def run_command(cmd, cwd=None):
    print(f"Running: {cmd} in {cwd if cwd else os.getcwd()}")
    subprocess.run(cmd, shell=True, check=True, cwd=cwd)

def load_config(path):
    with open(path, 'r') as f:
        return yaml.safe_load(f)


def main():

    config = load_config(CONFIG_PATH)

    # CHAMMI
    if config.get('CHAMMI', False):
        morphem_dir = os.path.join(BENCHMARKS_DIR, 'morphem')
        chammi_cmd = (
            f"python feature_extraction.py "
            f"--root_dir {config['CHAMMI_IMAGES_PATH']} "
            f"--feat_dir {config['CHAMMI_FEATURES_PATH']} "
            f"--model {config['MODEL_TYPE']} "
            f"--model_path {config['MODEL_PATH']} "
            f"--model_size {config['MODEL_SIZE']} "
            f"--batch_size {config['CHAMMI_BATCH_SIZE']} "
        )
        run_command(chammi_cmd, cwd=morphem_dir)

        # Run scoring from benchmarks folder (not morphem)
        benchmark_cmd = (
            f"python -c \"from morphem.benchmark import run_benchmark; "
            f"run_benchmark('{config['CHAMMI_IMAGES_PATH']}', '{config['CHAMMI_SCORE_PATH']}', "
            f"'{config['CHAMMI_FEATURES_PATH']}', f'pretrained_{config['MODEL_TYPE']}_features.npy')\""
        )
        run_command(benchmark_cmd, cwd=BENCHMARKS_DIR)

    # HPA
    if config.get('HPA', False):
        hpa_dir = os.path.join(BENCHMARKS_DIR, 'hpa')
        hpa_cmd = (
            f"accelerate launch --multi_gpu --num_processes={config['NUM_GPUS']} accelerate_hpa_features.py "
            f"--model {config['MODEL_TYPE']} "
            f"--model_path {config['MODEL_PATH']} "
            f"--image_folder {config['HPA_IMAGES_PATH']} "
            f"--batch_size {config['HPA_BATCH_SIZE']} "
            f"--num_workers {config['HPA_NUM_WORKERS']} "
            f"--output_folder {config['HPA_FEATURES_PATH']}"
        )
        # run_command(hpa_cmd, cwd=hpa_dir)

        train_cmd = (
            f"python train_classification.py -f {config['HPA_FEATURES_PATH']} -cc locations -uc challenge_cats"
        )
        run_command(train_cmd, cwd=hpa_dir)
        
        train_cmd = (
            f"python train_classification.py -f {config['HPA_FEATURES_PATH']} -cc locations -uc unique_cats"
        )
        run_command(train_cmd, cwd=hpa_dir)

    # Neuron Features
    if config.get('NEURON_FEATURES', False):
        neuron_dir = os.path.join(BENCHMARKS_DIR, 'neuron_features')
        neuron_cmd = (
            f"accelerate launch --multi_gpu --num_processes={config['NUM_GPUS']} extraction.py "
            f"--model {config['MODEL_TYPE']} "
            f"--image_folder {config['NEURON_IMAGES_PATH']} "
            f"--model_path {config['MODEL_PATH']} "
            f"--output_folder {config['NEURON_FEATURES_PATH']} "
            f"--num_workers {config['NEURON_NUM_WORKERS']}"
        )
        # run_command(neuron_cmd, cwd=neuron_dir)

        classifier_cmd = (
            f"python classifier.py --embedding_path {config['NEURON_FEATURES_PATH']}"
        )
        run_command(classifier_cmd, cwd=neuron_dir)

    # IDR-17 Benchmark
    if config.get('IDR-17', False):
        idr_config_path = os.path.join(BENCHMARKS_DIR, 'idr0017_benchmark/configs/idr0017/idr17_temp.yaml')
        write_dinov2_config(config, idr_config_path)
        workflow_dir = os.path.join(BENCHMARKS_DIR, 'idr0017_benchmark/workflow')
        snakemake_cmd = (
            f"snakemake --configfile ../configs/idr0017/idr17_temp.yaml --jobs {config['NUM_GPUS']} --cores {config['NUM_GPUS']*config['IDR_NUM_WORKERS']}"
        )
        run_command(snakemake_cmd, cwd=workflow_dir)

    # JUMPCP Features
    if config.get('JUMPCP', False):
        jumpcp_dir = os.path.join(BENCHMARKS_DIR, 'jumpcp1')
        feature_conversion_cmd = (
            f"python feature_extraction.py f --root_dir {config['JUMPCP_IMAGES_PATH']} --model_path {config['MODEL_PATH']} --feat_dir {config['JUMPCP_FEATURES_PATH']} --model {config['MODEL_TYPE']}"
        )
        feature_aggregation_normalization_cmd = (
            f"python well_level_aggregation.py --features_path {config['JUMPCP_FEATURES_PATH']}/{config['MODEL_TYPE']} --model {config['MODEL_TYPE']}"
        )
        benchmark_cmd = (
            f"python run_evaluation.py --model {config['MODEL_TYPE']}"
        )
        run_command(classifier_cmd, cwd=jumpcp_dir)
        run_command(feature_aggregation_normalization_cmd, cwd=jumpcp_dir)
        run_command(benchmark_cmd, cwd=jumpcp_dir)

if __name__ == "__main__":
    main()
