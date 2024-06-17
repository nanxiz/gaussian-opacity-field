FROM runpod/pytorch:2.0.1-py3.10-cuda11.8.0-devel-ubuntu22.04

ENV DEBIAN_FRONTEND=noninteractive

# Define environment variables for use in the build
ENV MINICONDA_INSTALLER_SCRIPT Miniconda3-py38_23.1.0-1-Linux-x86_64.sh
ENV MINICONDA_PREFIX /usr/local

# Install necessary packages
RUN apt-get update && apt-get install -y wget g++-9 gcc-9 git cmake make

# Set alternatives for gcc and g++
RUN update-alternatives --install /usr/bin/g++ g++ /usr/bin/g++-9 20 && \
    update-alternatives --install /usr/bin/gcc gcc /usr/bin/gcc-9 9

# Download and install Miniconda
RUN wget https://repo.continuum.io/miniconda/$MINICONDA_INSTALLER_SCRIPT && \
    chmod +x $MINICONDA_INSTALLER_SCRIPT && \
    ./$MINICONDA_INSTALLER_SCRIPT -b -f -p $MINICONDA_PREFIX && \
    rm $MINICONDA_INSTALLER_SCRIPT

# Initialize Conda
RUN ln -s $MINICONDA_PREFIX/etc/profile.d/conda.sh /etc/profile.d/conda.sh && \
    echo ". $MINICONDA_PREFIX/etc/profile.d/conda.sh" >> ~/.bashrc && \
    echo "conda activate base" >> ~/.bashrc

# Clone the repository
RUN git clone https://github.com/autonomousvision/gaussian-opacity-fields.git
WORKDIR /gaussian-opacity-fields

# Create a new Conda environment
RUN conda create -y -n gof python=3.8 && \
    echo "conda activate gof" >> ~/.bashrc

# Install PyTorch
RUN . $MINICONDA_PREFIX/etc/profile.d/conda.sh && \
    conda activate gof && \
    pip install torch==1.12.1+cu113 torchvision==0.13.1+cu113 -f https://download.pytorch.org/whl/torch_stable.html

# Install additional Conda packages
RUN conda install cudatoolkit-dev=11.3 -c conda-forge -y && \
    conda install cmake -y && \
    conda install conda-forge::gmp -y && \
    conda install conda-forge::cgal -y

# Install Python requirements
COPY requirements.txt /gaussian-opacity-fields/
RUN pip install -r requirements.txt

# Install custom submodules
RUN pip install submodules/diff-gaussian-rasterization && \
    pip install submodules/simple-knn/

# Build and install Tetra-nerf for triangulation
WORKDIR /gaussian-opacity-fields/submodules/tetra-triangulation
RUN cmake . && make && pip install -e .

# Return to the main directory
WORKDIR /gaussian-opacity-fields

