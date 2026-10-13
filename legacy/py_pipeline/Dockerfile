# Research-grade Dockerfile for PINN Thermal Dataset Generator
# Leverages official FEniCSx image which has PETSc, SLEPc, MPI, and DOLFINx precompiled.
FROM dolfinx/dolfinx:v0.8.0

# Set non-interactive mode for installation
ENV DEBIAN_FRONTEND=noninteractive

# Update apt-get and install baseline utilities
RUN apt-get update && apt-get install -y --no-install-recommends \
    python3-pip \
    python3-setuptools \
    libgl1-mesa-glx \
    libglib2.0-0 \
    && rm -rf /var/lib/apt/lists/*

# Establish pipeline execution context
WORKDIR /workspace

# Copy requirements list first to cache layer installation
COPY requirements.txt /workspace/requirements.txt

# Install standard numeric, data-indexing, and visualization python tools
RUN pip3 install --no-cache-dir -r requirements.txt

# Copy dataset generator scripts to image
COPY floorplan_generator.py /workspace/
COPY pde_solver.py /workspace/
COPY sensor_generator.py /workspace/
COPY dataset_generator.py /workspace/
COPY visualization.py /workspace/
COPY validate_dataset.py /workspace/

# Establish default execution entry point
# Runs prototype collection generation (500 train / 100 val / 100 test using DOLFINx)
ENTRYPOINT ["python3", "dataset_generator.py"]
CMD ["--scale", "prototype"]
