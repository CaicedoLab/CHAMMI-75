docker_image = vidit2003/trainfoundationmodel:v1
log = Get_Embeds$(Cluster).log

# Specify your executable (single binary or a script that runs several
#  commands), arguments, and a files for HTCondor to store standard
#  output (or "screen output").
#  $(Process) will be a integer number for each job, starting with "0"
#  and increasing for the relevant number of jobs.
executable = execute.sh
arguments = $(Process)
output = Get_Embeds$(Cluster)_$(Process).out
error = Get_Embeds$(Cluster)_$(Process).err

# Specify that HTCondor should transfer files to and from the
#  computer where each job runs. The last of these lines *would* be
#  used if there were any other files needed for the executable to use.
should_transfer_files = YES
when_to_transfer_output = ON_EXIT_OR_EVICT
transfer_input_files = execute.sh, /home/vagrawal22/caicedo/FoundationModels/FoundationModels/dataset, /home/vagrawal22/caicedo/FoundationModels/dinov1
# Tell HTCondor what amount of compute resources
#  each job will need on the computer where it runs.

requirements = (Machine == "jcaicedogpu0002.chtc.wisc.edu")
request_cpus = 96
request_memory = 500GB
request_disk =  200GB
request_gpus = 8
queue 1
