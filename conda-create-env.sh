#!/usr/bin/env bash
set -eu -o pipefail

DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" >/dev/null 2>&1 && pwd -P )"

# create a new conda environment
conda env create -f $DIR/environment.yml

# create files to set when activating/deactivating the environment
ENV_DIR=$HOME/miniconda/envs/qbm
mkdir -p $ENV_DIR/etc/conda/activate.d
mkdir -p $ENV_DIR/etc/conda/deactivate.d
touch $ENV_DIR/etc/conda/activate.d/env_vars.sh
touch $ENV_DIR/etc/conda/deactivate.d/env_vars.sh

# set activation of environment variables
cat <<EOT > $ENV_DIR/etc/conda/activate.d/env_vars.sh
export QBM_PROJECT_DIR=\$HOME/thesis
EOT

# set deactivation of environment variables
cat <<EOT > $ENV_DIR/etc/conda/deactivate.d/env_vars.sh
unset QBM_PROJECT_DIR
EOT
