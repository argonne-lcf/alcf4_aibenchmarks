# Building dependencies for Megatron-DeepSpeed
# Please run the following commands from a compute nodes

module load frameworks
export PBS_O_WORKDIR=$(pwd)
source <(curl -s https://raw.githubusercontent.com/saforem2/ezpz/refs/heads/main/src/ezpz/bin/utils.sh)
$(ezpz_setup_env | tee build.log)
python3 -m pip install -e "git+https://github.com/saforem2/ezpz#egg=ezpz" --require-virtualenv
python3 -m pip install deepspeed
echo "#/bin/bash
module load frameworks
source $VIRTUAL_ENV/bin/activate
" > conda.sh
chmod 755 conda.sh