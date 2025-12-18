Running benchmarks, you can hack models.py script to add your new models to run all the five benchmarks. 
Use the benchmark_config.yaml to point to the correct data sources.


There are two ways one can run the benchmarks code:

1. You can configure benchmark_config.yaml file which contains all the needed parameters to run all five benchmarks. Each benchmark can be turned on or off using the boolean value present in the .yaml file. After this, we need to run the script evaluate.py to run the corresponding benchmark.
2. You can directly go to the folder of that benchmark and run the command.
