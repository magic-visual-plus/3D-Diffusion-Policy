# use the same command as training except the script
# for example:
# bash scripts/infer.sh dp3 adroit_hammer 0322 0 0



DEBUG=False

alg_name=${1}
task_name=${2}
config_name=${alg_name}
addition_info=${3}
seed=${4}
exp_name=${task_name}-${alg_name}-${addition_info}
run_dir="/mnt/d/3.0/3D-Diffusion-Policy/data/outputs/${exp_name}_seed${seed}"

gpu_id=${5}


cd 3D-Diffusion-Policy

export HYDRA_FULL_ERROR=1
export CUDA_VISIBLE_DEVICES=${gpu_id}
python infer.py --config-name=${config_name}.yaml \
                            task=${task_name} \
                            hydra.run.dir=${run_dir} \
                            exp_name=${exp_name}



                                