#!/usr/bin/env bash
# Phase0 install v2: openpi + LIBERO + aloha from codeload tarballs (git clone over github is flaky on AutoDL),
# pinned to openpi 215abfb2 and its submodule commits. AutoDL network_turbo for github; aliyun for PyPI.
set -eo pipefail
source /root/autodl-tmp/env.sh
git config --global --unset-all url.https://ghfast.top/https://github.com/.insteadof 2>/dev/null || true
source /etc/network_turbo >/dev/null 2>&1 || true
export no_proxy=mirror.nju.edu.cn,mirrors.aliyun.com,hf-mirror.com,localhost,127.0.0.1 NO_PROXY=mirror.nju.edu.cn,mirrors.aliyun.com,hf-mirror.com,localhost,127.0.0.1
export UV_DEFAULT_INDEX=http://mirrors.aliyun.com/pypi/simple UV_INSECURE_HOST=mirrors.aliyun.com UV_HTTP_TIMEOUT=300
OPENPI=215abfb217dbac7d5f1273282331b9b1866c0479
LIBERO=f78abd68ee283de9f9be3c8f7e2a9ad60246e95c
ALOHA=d1dc83afd89ded4379851257fe5d85632d31d5ec
fetch(){ for n in $(seq 1 20); do tar tzf "$2" >/dev/null 2>&1 && return 0; rm -f "$2"; curl -sSfL --retry 5 --retry-delay 5 -o "$2" "$1" && return 0; echo "retry $n $1"; sleep 5; done; return 1; }
pip install -q -U uv
mkdir -p $W/src; cd $W/src
fetch https://codeload.github.com/Physical-Intelligence/openpi/tar.gz/$OPENPI openpi-$OPENPI.tgz
fetch https://codeload.github.com/Lifelong-Robot-Learning/LIBERO/tar.gz/$LIBERO libero-$LIBERO.tgz
fetch https://codeload.github.com/Physical-Intelligence/aloha/tar.gz/$ALOHA aloha-$ALOHA.tgz
sha256sum *.tgz
cd $W; rm -rf openpi; mkdir openpi
tar xzf src/openpi-$OPENPI.tgz -C openpi --strip-components=1
tar xzf src/libero-$LIBERO.tgz -C openpi/third_party/libero --strip-components=1
tar xzf src/aloha-$ALOHA.tgz -C openpi/third_party/aloha --strip-components=1
echo "OPENPI_COMMIT=$OPENPI LIBERO_COMMIT=$LIBERO ALOHA_COMMIT=$ALOHA"
cd openpi
for n in 1 2 3 4 5; do GIT_LFS_SKIP_SMUDGE=1 uv sync && break; echo "uv sync retry $n"; sleep 15; done
GIT_LFS_SKIP_SMUDGE=1 uv pip install -e .
uv venv --python 3.8 examples/libero/.venv
source examples/libero/.venv/bin/activate
uv pip sync examples/libero/requirements.txt third_party/libero/requirements.txt --extra-index-url https://download.pytorch.org/whl/cu113 --index-strategy=unsafe-best-match
uv pip install -e packages/openpi-client -e third_party/libero
deactivate
echo INSTALL_DONE
