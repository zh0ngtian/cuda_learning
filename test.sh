set -e

export CUDA_VISIBLE_DEVICES="2"

nvcc -o matmul -arch=compute_75 -code=sm_75 matmul.cu -lcublas

./matmul
