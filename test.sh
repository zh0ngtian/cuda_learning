set -e

export CUDA_VISIBLE_DEVICES="2"

nvcc -o $1 -arch=compute_86 -code=sm_86 $1.cu -lcublas -lcudnn

./$1
