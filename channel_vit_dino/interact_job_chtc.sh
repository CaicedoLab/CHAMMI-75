container_image = file:///staging/groups/caicedo_group/images/channel_vit_dino.sif
log = logs/train$(Cluster).log
universe = container
executable = execute_job.sh
arguments = $(Process)
output = logs/train$(Cluster)_$(Process).out
error = logs/train$(Cluster)_$(Process).err
environment = "WANDB_API_KEY=$(<$HOME/wandb_api_key.txt)"


should_transfer_files = YES
when_to_transfer_output = ON_EXIT_OR_EVICT
transfer_input_files = execute_job.sh 

requirements = ( Machine == "jcaicedogpu0000.chtc.wisc.edu" || Machine == "jcaicedogpu0001.chtc.wisc.edu" || Machine == "jcaicedogpu0002.chtc.wisc.edu" )
request_cpus = 4
request_gpus = 1
request_memory = 32GB
request_disk =  32GB
queue 1

# run this file with condor_submit wandb_key=$WANDB_API_KEY chtc_job.sh