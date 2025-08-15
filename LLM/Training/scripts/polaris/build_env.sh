#!/bin/bash
# Building dependencies for Megatron-DeepSpeed
# Please run the following commands from a compute nodes
# -- conda envs
module load conda
conda activate

export PBS_O_WORKDIR=$(pwd)
export PYTHONNOUSERSITE=1
source <(curl -s https://raw.githubusercontent.com/saforem2/ezpz/refs/heads/main/src/ezpz/bin/utils.sh)
NO_COLOR=1 ezpz_setup_env | tee build.log
export VIRTUAL_ENV=$(dirname $(dirname $(grep "Using python from:" build.log | awk -F': ' '{print $2}' | sed 's/\x1b\[[0-9;]*m//g')))
source $VIRTUAL_ENV/bin/activate
python3 -m pip install -e "git+https://github.com/saforem2/ezpz#egg=ezpz" --require-virtualenv
python3 -m pip install flash-attn --no-build-isolation
python3 -m pip install deepspeed
export BASE_PREFIX=$(python3 -c "import sys; print(sys.base_prefix)")
[ -e $VIRTUAL_ENV/bin/python3-config ] || cp $BASE_PREFIX/bin/python3-config $VIRTUAL_ENV/bin/python3-config
echo "#!/bin/bash
module load conda
export VIRTUAL_ENV=$VIRTUAL_ENV
source \$VIRTUAL_ENV/bin/activate
" > conda.sh
[[ -e Megatron-DeepSpeed ]] || git clone https://github.com/argonne-lcf/Megatron-DeepSpeed
cd Megatron-DeepSpeed/megatron/data
make 
cd -
chmod 755 conda.sh