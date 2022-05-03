##############
# Base Image #
##############

# Get base image
FROM continuumio/miniconda3:4.10.3p0

################
# Dependencies #
################

# Install basic dependencies and tools
RUN apt-get update && apt-get install -y \
 curl \
 apt-utils \
 pkg-config \
 apt-transport-https \
 debconf-utils \
 gcc \
 build-essential \
 g++-6 \
 && rm -rf /var/libapt/lists/*

# Adding custom ms repository
RUN curl https://packages.microsoft.com/keys/microsoft.asc | apt-key add -
RUN curl https://packages.microsoft.com/config/debian/9/prod.list > /etc/apt/sources.list.d/mssql-release.list

# # Install SQL Server drivers
# RUN apt-get update &&  ACCEPT_EULA=Y apt-get install -y \
#  mssql-tools

# RUN echo 'export PATH="$PATH:/opt/mssql-tools/bin"' >> ~/.bashrc
# RUN /bin/bash -c "source ~/.bashrc"

# # optional: for unixODBC development headers
# RUN apt-get install -y unixodbc-dev 

####################################
# Conda Environment specific steps #
####################################

COPY . /home
RUN conda create --name myenv python=3.9.5
RUN conda env list
SHELL ["conda", "run", "-n", "myenv", "/bin/bash", "-c"]
RUN pip install -r /home/requirements.txt

##################
# Create folders #
##################

RUN mkdir /home/artifacts

##################
# Set working directory and environment #
##################
WORKDIR /home 
RUN pip install -e .

#CMD ["python", "/home/src/main.py"]

