import os
import random
import subprocess
import sys

import ray

train_script = sys.argv[1]

master_addr = os.environ.get("MASTER_ADDR", "127.0.0.1")
# master_port = os.environ.get("MASTER_PORT", str(random.randint(20001, 29999)))
master_port = "29905" # especially for aliyun
nnodes = int(os.environ.get("NODE_COUNT", "1"))
node_rank = int(os.environ.get("NODE_RANK", "0"))

print(f"current rank: {node_rank}, nnodes: {nnodes}, master_addr: {master_addr}, master_port: {master_port}")

if nnodes <= 1:
    process = subprocess.run(
        f'{train_script}',
        shell=True,
    )
    exit(process.returncode)

if node_rank == 0:
    # 如果镜像拉起的慢，可以把 sleep 时间调大一些，直到所有节点都能连上 master
    process = subprocess.run(
        f'ray start --head --node-ip-address={master_addr} --port={master_port} && sleep 60s && {train_script}',
        shell=True,
    )
    exit(process.returncode)
else:
    process = subprocess.run(
        f'ray start --address {master_addr}:{master_port} --block',
        shell=True,
    )
    exit(process.returncode)
