FROM runpod/pytorch:2.0.1-py3.10-cuda11.8.0-devel-ubuntu22.04

ENV CUDA_HOME=/usr/local/cuda-11.8

WORKDIR /workspace



RUN apt-get update && apt-get install -y \
    git \
    g++-9 \
    gcc-9 \
    cmake \
    libgmp-dev \
    libmpfr-dev \
    libboost-all-dev \
    libcgal-dev \
    && rm -rf /var/lib/apt/lists/*


# Update alternatives for GCC and G++
RUN update-alternatives --install /usr/bin/g++ g++ /usr/bin/g++-9 20
RUN update-alternatives --install /usr/bin/gcc gcc /usr/bin/gcc-9 9



ENV PATH /usr/local/cuda-11.8/bin:$PATH
ENV LD_LIBRARY_PATH /usr/local/cuda-11.8/lib64:$LD_LIBRARY_PATH

RUN nvcc --version
RUN which nvcc


RUN git clone https://github.com/SpectacularAI/point-cloud-tools.git
COPY . /workspace/gaussian-opacity-field


WORKDIR /workspace/gaussian-opacity-field

RUN pip install blinker --ignore-installed


# Install other Python requirements
COPY requirements.txt /workspace/gaussian-opacity-field/
RUN pip install -r requirements.txt

RUN pip install blinker --ignore-installed

RUN pip install -r requirements.txt


# Install submodules
RUN pip install submodules/diff-gaussian-rasterization/diff_gaussian_rasterization-0.0.0-cp310-cp310-linux_x86_64.whl

RUN pip install submodules/simple-knn/simple_knn-0.0.0-cp310-cp310-linux_x86_64.whl


RUN pip install scikit-image kornia tqdm diffusers accelerate einops transformers 

RUN pip install xformers==0.0.20



# Build and install Tetra-nerf for triangulation
#WORKDIR /workspace/gaussian-opacity-field/submodules/tetra-triangulation
#RUN cmake . && make && pip install -e .
#RUN pip install submodules/tetra-triangulation/tetra_nerf-0.1.1-py3-none-any.whl

WORKDIR /workspace/gaussian-opacity-field/submodules/tetra-triangulation
RUN pip install -e .


RUN pip install runpod 

# Return to the main directory
WORKDIR /workspace/gaussian-opacity-field


CMD ["python", "handler.py"]
