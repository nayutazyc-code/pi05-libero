#!/usr/bin/env bash
# LIBERO client venv (py3.8): NJU uv-python mirror lacks 3.8 build -> use conda (tsinghua) python3.8.
set -eo pipefail
source /root/autodl-tmp/env.sh
export UV_DEFAULT_INDEX=http://mirrors.aliyun.com/pypi/simple UV_INSECURE_HOST="mirrors.aliyun.com download.pytorch.org" UV_HTTP_TIMEOUT=300
[ -x $W/conda/py38/bin/python ] || conda create -y -p $W/conda/py38 python=3.8
cd $W/openpi
rm -rf examples/libero/.venv
uv venv --python $W/conda/py38/bin/python examples/libero/.venv
source examples/libero/.venv/bin/activate
uv pip sync examples/libero/requirements.txt third_party/libero/requirements.txt --find-links https://mirrors.aliyun.com/pytorch-wheels/cu113/ --index-strategy=unsafe-best-match
uv pip install -e packages/openpi-client -e third_party/libero
python -c "import libero, openpi_client, robosuite; print('LIBERO_VENV_OK')"
