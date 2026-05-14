# 1. 基础镜像
# 对应你的 JetPack 5.1.2 (L4T 35.4.1)，包含 TensorRT 8.5.2
# 这是一个 ARM64 镜像，你在 x86 电脑上无法 run，但可以 build 和 pull
FROM nvcr.io/nvidia/l4t-tensorrt:r8.5.2.2-devel

LABEL version="1.0-orin-r35.4.1" description="DSVT Deploy Environment for Orin"

# 2. 替换 APT 源为中科大源 (加速下载)
# 注意：ARM 架构必须使用 'ubuntu-ports'，绝不能用普通的 'ubuntu'
RUN echo 'deb https://mirrors.ustc.edu.cn/ubuntu-ports/ focal main restricted universe multiverse' > /etc/apt/sources.list && \
    echo 'deb https://mirrors.ustc.edu.cn/ubuntu-ports/ focal-updates main restricted universe multiverse' >> /etc/apt/sources.list && \
    echo 'deb https://mirrors.ustc.edu.cn/ubuntu-ports/ focal-backports main restricted universe multiverse' >> /etc/apt/sources.list && \
    echo 'deb https://mirrors.ustc.edu.cn/ubuntu-ports/ focal-security main restricted universe multiverse' >> /etc/apt/sources.list

# 3. 安装系统依赖
# python3-dev: 编译 pybind11 必须
# libeigen3-dev: 你的算法依赖
ENV DEBIAN_FRONTEND=noninteractive
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    python3-pip \
    python3-dev \
    git \
    wget \
    libeigen3-dev \
    libopenblas-dev \
    && rm -rf /var/lib/apt/lists/*

# 4. 安装 CMake 3.25 (ARM64)
# 系统自带的 cmake 太老，这里直接下载 ARM64 预编译版
# 这一步会自动在构建时下载，你不需要手动去官网下文件了
RUN wget https://github.com/Kitware/CMake/releases/download/v3.25.1/cmake-3.25.1-linux-aarch64.sh -O /tmp/cmake-install.sh \
    && chmod +x /tmp/cmake-install.sh \
    && /tmp/cmake-install.sh --skip-license --exclude-subdir --prefix=/usr/local \
    && rm /tmp/cmake-install.sh

# 5. 配置 Python 环境 (直接使用系统 Python 3.8)
# 升级 pip 并换源
RUN python3 -m pip install --upgrade pip && \
    python3 -m pip config set global.index-url https://mirrors.ustc.edu.cn/pypi/web/simple

# 6. 安装核心 Python 库
# 注意：numpy 和 tensorrt 在基础镜像里通常已经预装了，
# 我们这里只安装额外的编译工具。
RUN python3 -m pip install pybind11

# 7. (可选) 验证环境
# 这一步在 x86 上 build 可能会失败（因为无法加载 GPU 驱动），建议注释掉，或者忽略报错。
# 如果是在 Orin 本机 build，这行可以用来验证。
# RUN python3 -c "import tensorrt; print('TensorRT Ready:', tensorrt.__version__)"

WORKDIR /workspace

CMD ["/bin/bash"]