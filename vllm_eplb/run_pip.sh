#!/usr/bin/env bash


EXTERNAL_REPO_DIR="/mnt/shared-storage-user/chenguanjie/huawei_eplb/extra_repos"
export MAX_JOBS=96
export PIP_CONFIG_FILE=/dev/null
export VLLM_CUTLASS_SRC_DIR="${EXTERNAL_REPO_DIR}/cutlass"
export FLASH_MLA_SRC_DIR="${EXTERNAL_REPO_DIR}/FlashMLA"
export QUTLASS_SRC_DIR="${EXTERNAL_REPO_DIR}/qutlass"
export TRITON_KERNELS_SRC_DIR="${EXTERNAL_REPO_DIR}/triton/python/triton_kernels/triton_kernels"
export VLLM_FLASH_ATTN_SRC_DIR="${EXTERNAL_REPO_DIR}/flash-attention"
export FETCHCONTENT_SOURCE_DIR_ONEDNN="${EXTERNAL_REPO_DIR}/oneDNN"
export ACL_ROOT_DIR="${EXTERNAL_REPO_DIR}/ComputeLibrary"
export TORCH_CUDA_ARCH_LIST="9.0"


# 0. download submodules recursively
# https://github.com/vllm-project/FlashMLA GIT_TAG 692917b1cda61b93ac9ee2d846ec54e75afe87b1
# https://github.com/IST-DASLab/qutlass.git GIT_TAG 830d2c4537c7396e14a02a46fbddd18b5d107c65
# https://github.com/triton-lang/triton/tree/main/python/triton_kernels GIT_TAG v3.6.0
# https://github.com/vllm-project/flash-attention.git  GIT_TAG 86f8f157cf82aa2342743752b97788922dd7de43
# https://github.com/oneapi-src/oneDNN.git  GIT_TAG v3.10
# https://github.com/ARM-software/ComputeLibrary.git GIT_TAG v52.6.0
# cd $VLLM_FLASH_ATTN_SRC_DIR && git submodule update --init --recursive

# 0. clean up
# find . -name "*.pyc" -delete
# find . -name "__pycache__" -delete
# rm -rf build/ dist/ *.egg-info

# 1. setuptools
# pip install "setuptools>=77.0.3,<81.0.0" "setuptools-scm>=8.0" "packaging==24.2" \
#     -i http://mirrors.i.h.pjlab.org.cn/pypi/simple/ \
#     --trusted-host mirrors.i.h.pjlab.org.cn

# 2. clear torch, vllm and flash-attn
pip install torch==2.9.0 torchvision==0.24.0 torchaudio==2.9.0 -i http://mirrors.i.h.pjlab.org.cn/pypi/simple/ --trusted-host mirrors.i.h.pjlab.org.cn

# 3. re-compile vllm (skip fla-3)
# cd vllm_v13
# pip install -e . -v -i http://mirrors.i.h.pjlab.org.cn/pypi/simple/ --trusted-host mirrors.i.h.pjlab.org.cn

# 4. remove deep_ep, re-compile
# cd vllm_v13/tools/ep_kernels

# 5. install nvshmem
# pip install nvshmem4py-cu12 nvidia-nvshmem-cu12 -i http://mirrors.i.h.pjlab.org.cn/pypi/simple/ --trusted-host mirrors.i.h.pjlab.org.cn --no-build-isolation

# # 6. 
# pip install numpy==1.26 -i http://mirrors.i.h.pjlab.org.cn/pypi/simple/ --trusted-host mirrors.i.h.pjlab.org.cn --no-build-isolation
# pip install opencv-python==4.9.0.80 -i http://mirrors.i.h.pjlab.org.cn/pypi/simple/ --trusted-host mirrors.i.h.pjlab.org.cn --no-build-isolation
