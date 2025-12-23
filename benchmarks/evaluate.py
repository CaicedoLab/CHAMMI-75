"""
Master file to run all evaluations at once
Run the commands using a config file (benchmark_config.yaml)
"""

import yaml
import subprocess
import os

CONFIG_PATH = os.path.join(os.path.dirname(__file__), "benchmark_config.yaml")
BENCHMARKS_DIR = os.path.dirname(__file__)


# Helper to run a command in a specific folder
def run_command(cmd, cwd=None):
    print(f"Running: {cmd} in {cwd if cwd else os.getcwd()}")
    subprocess.run(cmd, shell=True, check=True, cwd=cwd)


def load_config(path):
    with open(path, "r") as f:
        return yaml.safe_load(f)


def main():
    config = load_config(CONFIG_PATH)

    # CHAMMI
    if config.get("CHAMMI", False):
        morphem_dir = os.path.join(BENCHMARKS_DIR, "CHAMMI")
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
            f'python -c "from CHAMMI.benchmark import run_benchmark; '
            f"run_benchmark('{config['CHAMMI_IMAGES_PATH']}', '{config['CHAMMI_SCORE_PATH']}', "
            f"'{config['CHAMMI_FEATURES_PATH']}', f'pretrained_{config['MODEL_TYPE']}_features.npy')\""
        )
        run_command(benchmark_cmd, cwd=BENCHMARKS_DIR)

    # HPA
    if config.get("HPA", False):
        hpa_dir = os.path.join(BENCHMARKS_DIR, "HPAv23")
        hpa_cmd = (
            f"accelerate launch --multi_gpu --num_processes={config['NUM_GPUS']} accelerate_hpa_features.py "
            f"--model {config['MODEL_TYPE']} "
            f"--model_path {config['MODEL_PATH']} "
            f"--image_folder {config['HPA_IMAGES_PATH']} "
            f"--batch_size {config['HPA_BATCH_SIZE']} "
            f"--num_workers {config['HPA_NUM_WORKERS']} "
            f"--output_folder {config['HPA_FEATURES_PATH']}"
        )
        run_command(hpa_cmd, cwd=hpa_dir)

        train_cmd = f"python train_classification.py -f {config['HPA_FEATURES_PATH']} -cc locations -uc challenge_cats"
        run_command(train_cmd, cwd=hpa_dir)

        train_cmd = f"python train_classification.py -f {config['HPA_FEATURES_PATH']} -cc locations -uc unique_cats"
        run_command(train_cmd, cwd=hpa_dir)

    # Neuron Features
    if config.get("CellPHIE", False):
        neuron_dir = os.path.join(BENCHMARKS_DIR, "CellPHIE")
        neuron_cmd = (
            f"accelerate launch --multi_gpu --num_processes={config['NUM_GPUS']} extraction.py "
            f"--model {config['MODEL_TYPE']} "
            f"--image_folder {config['CellPHIE_IMAGES_PATH']} "
            f"--model_path {config['MODEL_PATH']} "
            f"--output_folder {config['CellPHIE_FEATURES_PATH']} "
            f"--num_workers {config['CellPHIE_NUM_WORKERS']}"
        )
        run_command(neuron_cmd, cwd=neuron_dir)

        classifier_cmd = (
            f"python classifier.py --embedding_path {config['CellPHIE_FEATURES_PATH']} --embed_dim {config['CellPHIE_EMBED_DIM']}"
        )
        run_command(classifier_cmd, cwd=neuron_dir)

    # IDR-17 Benchmark
    if config.get("IDR-17", False):
        idr17_dir = os.path.join(BENCHMARKS_DIR, "IDR0017")

        idr_cmd = f"python feature_extraction.py --model_path {config['MODEL_PATH']} --model_type {config['MODEL_TYPE']} --batch_size 2048 --images_folder {config['IDR_DATA_FOLDER']} --output_folder {config['IDR_FEATURES_PATH']} --num_workers {config['IDR_NUM_WORKERS']}"
        run_command(idr_cmd, cwd=idr17_dir)

        benchmark_cmd = f"python idr0017_benchmark.py --features_dir {config['IDR_FEATURES_PATH']} --metadata_path {config['IDR_METADATA_PATH']} --save_dir {config['IDR_BENCHMARK_SAVE_DIR']}"
        run_command(benchmark_cmd, cwd=idr17_dir)

    # JUMPCP Features
    if config.get("JUMPCP", False):
        jumpcp_dir = os.path.join(BENCHMARKS_DIR, "JUMPCP")
        feature_conversion_cmd = f"python feature_extraction.py --root_dir {config['JUMPCP_IMAGES_PATH']} --model_path {config['MODEL_PATH']} --feat_dir {config['JUMPCP_FEATURES_PATH']} --model {config['MODEL_TYPE']} --batch_size {config['JUMPCP_BATCH_SIZE']}"
        feature_aggregation_normalization_cmd = f"python well_level_aggregation.py --profiles {config['JUMPCP_FEATURES_PATH']} --model {config['MODEL_TYPE']} --output_folder {config['JUMPCP_SCORE_PATH']}"
        benchmark_cmd = f"python run_evaluation.py --feat_dir {config['JUMPCP_SCORE_PATH']} --model {config['MODEL_TYPE']} --output_folder {config['JUMPCP_SCORE_PATH']}"
        run_command(feature_conversion_cmd, cwd=jumpcp_dir)
        run_command(feature_aggregation_normalization_cmd, cwd=jumpcp_dir)
        run_command(benchmark_cmd, cwd=jumpcp_dir)
    
    if config.get("RBC_MC", False):
        rbc_mc_dir = os.path.join(BENCHMARKS_DIR, "RBC-MC")
        rbc_mc_cmd = (
            f"accelerate launch --num_processes=1 extraction.py "
            f"--model {config['MODEL_TYPE']} "
            f"--model_path {config['MODEL_PATH']} "
            f"--output_folder {config['RBC_MC_FEATURES_PATH']} "
            f"--image_folder {config['RBC_MC_IMAGES_PATH']} "
        )
        run_command(rbc_mc_cmd, cwd=rbc_mc_dir)

        # Run scoring from benchmarks folder
        benchmark_cmd = (
            f'python regression.py'
            f' --output_folder {config["RBC_MC_SCORE_PATH"]}'
            f' --pkl_path {os.path.join(config["RBC_MC_FEATURES_PATH"], "embeddings.pkl")}'
        )
        run_command(benchmark_cmd, cwd=rbc_mc_dir)


if __name__ == "__main__":
    main()
