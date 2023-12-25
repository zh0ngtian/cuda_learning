set -e

export CUDA_VISIBLE_DEVICES="2"

nvcc -o $1 -arch=compute_75 -code=sm_75 $1.cu -lcublas

./$1
