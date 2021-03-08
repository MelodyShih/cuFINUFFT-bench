#!/bin/zsh
has_gpu=0
nvidia-smi > /dev/null && has_gpu=1

if [ $has_gpu = "1" ]; then
	echo "running 2d type 1"
	python nuffttype1.py 2 > ../results/2d1.dat
	echo "running 3d type 1"
	python nuffttype1.py 3 > ../results/3d1.dat
	echo "running 2d type 2"
	python nuffttype2.py 2 > ../results/2d2.dat
	echo "running 3d type 2"
	python nuffttype2.py 3 > ../results/3d2.dat
else
NODENAME='skylake'
#NODENAME='broadwell'
	echo "running 2d type 1"
	python nuffttype1.py 2 > ../results/$NODENAME/2d1_cpu.dat
	echo "running 3d type 1"
	python nuffttype1.py 3 > ../results/$NODENAME/3d1_cpu.dat
	echo "running 2d type 2"
	python nuffttype2.py 2 > ../results/$NODENAME/2d2_cpu.dat
	echo "running 3d type 2"
	python nuffttype2.py 3 > ../results/$NODENAME/3d2_cpu.dat
fi 
