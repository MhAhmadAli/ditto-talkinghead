# Base image with CUDA 12.2 + cuDNN + TensorRT 8.6.1
FROM nvcr.io/nvidia/tensorrt:23.06-py3

USER root

RUN apt update && apt install -y git git-lfs wget curl ffmpeg

# Install PyTorch (compatible with CUDA 12)
RUN pip install torch==2.5.1 torchvision==0.20.1 torchaudio==2.5.1 --index-url https://download.pytorch.org/whl/cu121

# Install TensorRT Python bindings & polygraphy
RUN pip install cuda-python librosa tqdm filetype imageio opencv_python_headless scikit-image cython cuda-python imageio-ffmpeg colored polygraphy numpy==2.0.1

WORKDIR /workspace

# Copy your source
COPY . /workspace

RUN git clone https://huggingface.co/digital-avatar/ditto-talkinghead checkpoints

# Default command
CMD ["bash"]
