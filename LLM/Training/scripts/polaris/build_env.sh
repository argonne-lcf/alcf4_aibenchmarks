# Building dependencies for Megatron-DeepSpeed
# Please run the following commands from a compute nodes
module load conda
export PBS_O_WORKDIR=$(pwd)
source <(curl -s https://raw.githubusercontent.com/saforem2/ezpz/refs/heads/main/src/ezpz/bin/utils.sh)
$(ezpz_setup_env | tee build.log)
python3 -m pip install -e "git+https://github.com/saforem2/ezpz#egg=ezpz" --require-virtualenv
python3 -m pip install deepspeed
echo "#/bin/bash
module load conda
source $VIRTUAL_ENV/bin/activate
" > conda.sh
[ -e Megatron-DeepSpeed ] || git clone https://github.com/argonne-lcf/Megatron-DeepSpeed
chmod 755 conda.sh