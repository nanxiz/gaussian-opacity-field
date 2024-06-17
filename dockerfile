FROM runpod/pytorch:2.0.1-py3.10-cuda11.8.0-devel-ubuntu22.04

ENV DEBIAN_FRONTEND=noninteractive

# Define environment variables for use in the build
ENV MINICONDA_INSTALLER_SCRIPT Miniconda3-py38_23.1.0-1-Linux-x86_64.sh
ENV MINICONDA_PREFIX /usr/local

WORKDIR /workspace

# Install necessary packages
RUN apt-get update && apt-get install -y \
    wget \
    git \
    g++-9 \
    gcc-9 \
    make \
    cmake \
    && apt-get clean \
    && rm -rf /var/lib/apt/lists/*

RUN git clone https://github.com/SpectacularAI/point-cloud-tools.git

# Update alternatives for GCC and G++
RUN update-alternatives --install /usr/bin/g++ g++ /usr/bin/g++-9 20 \
    && update-alternatives --install /usr/bin/gcc gcc /usr/bin/gcc-9 9

# Download and install Miniconda
RUN wget https://repo.anaconda.com/miniconda/$MINICONDA_INSTALLER_SCRIPT \
    && chmod +x $MINICONDA_INSTALLER_SCRIPT \
    && ./$MINICONDA_INSTALLER_SCRIPT -b -f -p $MINICONDA_PREFIX \
    && rm $MINICONDA_INSTALLER_SCRIPT


COPY . /workspace/gaussian-opacity-field

WORKDIR /workspace/gaussian-opacity-field

RUN conda create -y -n gof python=3.9 \
    && echo "source activate gof" > ~/.bashrc

RUN conda init bash


# Initialize Conda for bash shell
SHELL ["conda", "run", "-n", "gof", "/bin/bash", "-c"]

RUN pip install torch==1.12.1+cu113 torchvision==0.13.1+cu113 -f https://download.pytorch.org/whl/torch_stable.html


# Install CUDA toolkit
RUN conda install cudatoolkit-dev=11.3 -c conda-forge -y

# Install other Python requirements
COPY requirements.txt /workspace/gaussian-opacity-field/
RUN pip install -r requirements.txt


# Install submodules
RUN pip install submodules/diff-gaussian-rasterization \
    && pip install submodules/simple-knn/

# Install additional Conda packages
RUN conda install cmake -y && \
    conda install conda-forge::gmp -y && \
    conda install conda-forge::cgal -y


# Build and install Tetra-nerf for triangulation
WORKDIR /workspace/gaussian-opacity-fields/submodules/tetra-triangulation
RUN cmake . && make && pip install -e .



#RUN pip install pyvista

#RUN pip install fast-simplification

#RUN pip install pandas pandas pyarrow

RUN pip install runpod 

# Return to the main directory
WORKDIR /workspace/gaussian-opacity-field


CMD ["python", "handler.py"]
