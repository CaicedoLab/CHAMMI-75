#!/usr/bin/python3

import argparse
import os
import pathlib
import sys
import re
import classad
import htcondor
import htcondor.dags


def read_archives():
    with open("archives_7.txt", "r") as f:
        archives = [re.sub("\n", "", i) for i in f.readlines()]

    return archives


def generateDAG(prefix, working_dir, max_running):
    archives = read_archives()
    print(archives)
    compress_submit_description = htcondor.Submit(
        {
            "executable": "sc_fe.sh",
            "arguments": "$(archive)",
            "universe": "docker",
            "docker_image": "docker://arkkienkeli/recursion_mim:10",
            "request_disk": "100GB",
            "request_cpus": 8,
            "request_gpus": 1,
            "request_memory": "64GB",
            "log": f"./work/{prefix}-$(CLUSTER).log",
            "should_transfer_files": "YES",
            "when_to_transfer_output": "ON_EXIT",
            # Hack to target only the tech refresh hosts (which have significantly more network capacity)
            "requirements": "(TARGET.HasMorgridgeHdd ?: false) == true",
            "transfer_input_files": "sc_inference.py, sc_dataset_openphenom.py, sc_fe.sh, vit_encoder.py, huggingface_mae.py",
            "transfer_output_files": "/dev/null",
            "output": "./work/op-fe-$(JOB)_$(RETRY).out",
            "error": "./work/op-fe-$(JOB)_$(RETRY).err",
        }
    )
    working_dir_path = pathlib.Path(working_dir)

    dag = htcondor.dags.DAG()
    dag.layer(
        name=prefix,
        submit_description=compress_submit_description,
        vars=[
            {"node_name": f"OPFE-{idx}-", "archive": archives[idx]}
            for idx in range(len(archives))
        ],
        retries=int(1),
    )

    dag_dir = pathlib.Path(working_dir).absolute()
    dag_file = htcondor.dags.write_dag(
        dag, dag_dir, node_name_formatter=htcondor.dags.SimpleFormatter("_")
    )
    dag_submit = htcondor.Submit.from_dag(
        str(dag_file), {"batch-name": prefix, "maxjobs": max_running}
    )

    os.chdir(dag_dir)
    schedd = htcondor.Schedd()
    submit_result = schedd.submit(dag_submit)
    print(
        "Compress jobs were submitted as DAG with JobID %d.0" % submit_result.cluster()
    )


def countDags(prefix):
    schedd = htcondor.Schedd()
    return len(
        list(
            schedd.query(
                constraint="JobBatchName =?= %s" % classad.quote(prefix), projection=[]
            )
        )
    )


def topMain():
    parser = argparse.ArgumentParser(description="OpenPhenom feature extraction")
    parser.add_argument(
        "--instance", help="Instance name for the run", default="op-jump"
    )
    parser.add_argument(
        "-w",
        "--working-dir",
        help="Working directory for the DAG associated with the download instance",
        default=".",
    )
    parser.add_argument(
        "-r", "--max-running", help="Maximum number of running jobs", default=136
    )
    args = parser.parse_args()
    if countDags(args.instance):
        print(
            f"Cannot submit new compression named {args.instance}; one already exists in queue"
        )
        return 2

    generateDAG(args.instance, args.working_dir, args.max_running)
    return 0


def main():
    # The same script serves as both the driver and the EP-side wrapper. Look
    # at argv[1] to see what we should do in order to avoid dumping confusing help
    # options to the user
    if len(sys.argv) > 1 and sys.argv[1] in ["exec"]:
        return helperMain()

    return topMain()


if __name__ == "__main__":
    sys.exit(main())
