while true; 
do nvidia-smi -l 5 --query-gpu=utilization.gpu,utilization.memory,memory.total,memory.free,memory.used --format=csv >> gpu_utillization_2.log; sleep 1; 
done 
