ROOT = pelican://chtc.wisc.edu/morgridge/datavault/projects/Morgridge_Caicedo/projects
container_image = $(ROOT)/benchmarking/iclr-benchmarking.sif
log = logs/train$(Cluster).log
universe = container
executable = execute_eval.sh
arguments = $(Process)
output = logs/train$(Cluster)_$(Process).out
error = logs/train$(Cluster)_$(Process).err
environment = "MODEL_PATH=$(model_path) CHECKPOINT=$(checkpoint) FEATURE_DIR=$(feature_out) MODEL_TYPE=$(model_type) MODEL_SIZE=$(model_size)"

# Specify that HTCondor should transfer files to and from the
#  computer where each job runs. The last of these lines *would* be
#  used if there were any other files needed for the executable to use.
should_transfer_files = YES
when_to_transfer_output = ON_EXIT_OR_EVICT
transfer_input_files = execute_eval.sh, /home/jgpeters3/CHAMMI-75, $(ROOT)/foundation_models_and_benchmarking/chammi_dataset.zip
# Tell HTCondor what amount of compute resources 
#  each job will need on the computer where it runs.
# ( Machine == "jcaicedogpu0000.chtc.wisc.edu" || Machine == "jcaicedogpu0001.chtc.wisc.edu" || Machine == "jcaicedogpu0002.chtc.wisc.edu" || Machine == "coba2000.chtc.wisc.edu" )
# requirements = ( Machine == "jcaicedogpu0000.chtc.wisc.edu" || Machine == "jcaicedogpu0001.chtc.wisc.edu" || Machine == "jcaicedogpu0002.chtc.wisc.edu" || Machine == "coba2000.chtc.wisc.edu" )
requirements = ( Machine == "gpu4005.chtc.wisc.edu" || Machine == "gpu4006.chtc.wisc.edu" )
+WantGPULab = true
request_cpus = 9
request_memory = 42GB
request_disk =  100GB
request_gpus = 1
# +is_resumable = true
queue 1
