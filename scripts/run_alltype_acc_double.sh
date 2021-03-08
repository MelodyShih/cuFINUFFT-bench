#!/bin/zsh
has_gpu=0
nvidia-smi > /dev/null && has_gpu=1

if [ $has_gpu = "1" ]; then
	echo "running 2d type 1"
	python nuffttype1_acc_double.py 2 > ../results/2d1_acc_double.dat
	echo "running 3d type 1"
	python nuffttype1_acc_double.py 3 > ../results/3d1_acc_double.dat
	echo "running 2d type 2"
	python nuffttype2_acc_double.py 2 > ../results/2d2_acc_double.dat
	echo "running 3d type 2"
	python nuffttype2_acc_double.py 3 > ../results/3d2_acc_double.dat
else
#NODENAME='broadwell'
NODENAME='skylake'
	echo "running 2d type 1"
	python nuffttype1_acc_double.py 2 > ../results/$NODENAME/2d1_acc_cpu_double.dat
	echo "running 3d type 1"
	python nuffttype1_acc_double.py 3 > ../results/$NODENAME/3d1_acc_cpu_double.dat
	echo "running 2d type 2"
	python nuffttype2_acc_double.py 2 > ../results/$NODENAME/2d2_acc_cpu_double.dat
	echo "running 3d type 2"
	python nuffttype2_acc_double.py 3 > ../results/$NODENAME/3d2_acc_cpu_double.dat
fi 
