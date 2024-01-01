set -e

export CUDA_VISIBLE_DEVICES="2"

nvcc -o build/$1 -arch=compute_86 -code=sm_86 kernels/$1.cu -lcublas -lcudnn

./build/$1
