# source me on remote (no secrets)
export PATH=/root/miniconda3/bin:/root/.local/bin:$PATH
export W=/root/autodl-tmp
export UV_CACHE_DIR=$W/cache/uv PIP_CACHE_DIR=$W/cache/pip XDG_CACHE_HOME=$W/cache
export OPENPI_DATA_HOME=$W/cache/openpi HF_HOME=$W/cache/hf HF_ENDPOINT=https://hf-mirror.com
export UV_PYTHON_INSTALL_DIR=$W/cache/uvpy TMPDIR=$W/tmp
export MUJOCO_GL=egl PYOPENGL_PLATFORM=egl
mkdir -p $W/cache $W/tmp $W/logs
export UV_PYTHON_INSTALL_MIRROR=https://mirror.nju.edu.cn/github-release/astral-sh/python-build-standalone
