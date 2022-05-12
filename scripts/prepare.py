import os

N_ENT = 1000
PREFIX = "100k"

COMMAND_REWRITE = "set -ex && python src/thesis/experiments/prepare.py --queries"

COMMAND_DATASET = "set -ex && python src/thesis/experiments/prepare.py " \
                  "--dataset {name} " \
                  f"-e {N_ENT} -r {{ratio}} -t 6 -c 3 -l 1 " \
                  "--relations --graph --gold-standard"

# 1) Rewrite queries
os.system(COMMAND_REWRITE)

# 2) Generate datasets, graphs and gold standards
for ratio_percent in range(0, 110, 10):
    name = f"{PREFIX}{ratio_percent}"
    ratio = ratio_percent / 100
    command = COMMAND_DATASET.format(name=name, ratio=ratio)
    os.system(command)
