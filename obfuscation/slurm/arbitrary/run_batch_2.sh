#!/bin/bash

#SBATCH --job-name=arbitrary

#SBATCH --array=1-920

###SBATCH --partition=price
#SBATCH --partition=p_ps848

#SBATCH --requeue

#SBATCH --nodes=1

#SBATCH --ntasks=1

#SBATCH --cpus-per-task=10

#SBATCH --gres=gpu:1

#SBATCH --mem=20000

#SBATCH --time=5:00:00

#SBATCH --output=/projects/f_ps848_1/zhaobin/adversarial/obfuscation/slurm/logs/%x_%A_%a_%N.txt

#SBATCH --error=/projects/f_ps848_1/zhaobin/adversarial/obfuscation/slurm/logs/%x_%A_%a_%N_err.txt

ROWINDEX=$((SLURM_ARRAY_TASK_ID + 1000))
echo ROWINDEX

# DECLARE PATH VARIABLES
USER_DIR=/scratch/zl430                             # locate conda env
PROJECT_DIR=/projects/f_ps848_1/zhaobin/adversarial # locate working directory

TORCH_ENV=coco # pytorch conda env
DB_ENV=db      # mongodb conda env

activate_env() {
  source "${USER_DIR}"/miniconda3/etc/profile.d/conda.sh && conda activate "$1"
}

echo "Running mongod with env ${DB_ENV}"
activate_env "${DB_ENV}"

LOG_DIR="${USER_DIR}"/fiftyone/${SLURM_JOB_NAME}/${SLURM_ARRAY_JOB_ID}_${ROWINDEX}
mkdir -p "${LOG_DIR}"
echo LOG_DIR: "${LOG_DIR}"

RND_PT=$(shuf -i 1025-65000 -n 1) # 1-1024 requires sudo
echo RND_PT: "${RND_PT}"

# https://www.mongodb.com/docs/v6.0/tutorial/manage-mongodb-processes/
mongod --dbpath "${LOG_DIR}" --port "${RND_PT}" --fork --logpath "${LOG_DIR}".log || exit

echo
echo "Running pytorch with env ${TORCH_ENV}"
activate_env "${TORCH_ENV}"

# https://voxel51.com/docs/fiftyone/user_guide/config.html#configuring-a-mongodb-connection
export FIFTYONE_DATABASE_URI=mongodb://127.0.0.1:${RND_PT}/

# python working directory is `obfuscation`
cd "${PROJECT_DIR}"/obfuscation || exit
bash "./slurm/${SLURM_JOB_NAME}/${ROWINDEX}.sh"

echo
echo "Completed and shutting down mongod..."
activate_env "${DB_ENV}"
mongod --shutdown --dbpath "${LOG_DIR}"
