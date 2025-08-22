container_image = file:///staging/groups/caicedo_group/images/channel_vit_dino.sif
log = logs/train$(Cluster).log
universe = container
executable = execute_job.sh
arguments = $(Process)
output = logs/train$(Cluster)_$(Process).out
error = logs/train$(Cluster)_$(Process).err
environment = "WANDB_API_KEY=$(wandb_key) CONFIG_NAME=$(config_name)"

# Specify that HTCondor should transfer files to and from the
#  computer where each job runs. The last of these lines *would* be
#  used if there were any other files needed for the executable to use.
should_transfer_files = YES
when_to_transfer_output = ON_EXIT_OR_EVICT
transfer_input_files = execute_job.sh, /home/jgpeters3/CHAMMI-75, $(config_path)
# , /hdd/jcaicedo/morphem/dataset/sampling/chammi_train.zip, /hdd/jcaicedo/morphem/dataset/sampling/multi_channel_chammi_metadata.csv 
# we don't transfer the dataset for 10ds and 75sd experiments, they are on the servers. 
# Tell HTCondor what amount of compute resources
#  each job will need on the computer where it runs.

requirements = ( Machine == "jcaicedogpu0000.chtc.wisc.edu" || Machine == "jcaicedogpu0001.chtc.wisc.edu" || Machine == "jcaicedogpu0002.chtc.wisc.edu" )
request_cpus = 96
request_gpus = 8
request_memory = 500GB
request_disk =  100GB
queue 1

# run this file with condor_submit wandb_key=$WANDB_API_KEY chtc_job.sh