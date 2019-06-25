export PYTHONPATH=$(pwd)

for trial in 0
do
	for param in "10,relu,20,10" "20,relu,40,20" "40,relu,80,40" "60,relu,120,60" "10,elu,20,10" "20,elu,40,20" "40,elu,80,40" "60,elu,120,60" "10,sigmoid,20,10" "20,sigmoid,40,20" "40,sigmoid,80,40" "60,sigmoid,120,60" "10,relu,10,20" "20,relu,20,40" "40,relu,40,80" "60,relu,60,120" "10,elu,10,20" "20,elu,20,40" "40,elu,40,80" "60,elu,60,120" "10,sigmoid,10,20" "20,sigmoid,20,40" "40,sigmoid,40,80" "60,sigmoid,60,120" "10,relu,10" "20,relu,20" "40,relu,40" "60,relu,60" "10,elu,10" "20,elu,20" "40,elu,40" "60,elu,60" "10,sigmoid,10" "20,sigmoid,20" "40,sigmoid,40" "60,sigmoid,60" "10,relu,20,10,10" "20,relu,40,20,20" "40,relu,80,40,40" "60,relu,120,60,60" "10,elu,20,10,10" "20,elu,40,20,20" "40,elu,80,40,40" "60,elu,120,60,60" "10,sigmoid,20,10,10" "20,sigmoid,40,20,20" "40,sigmoid,80,40,40" "60,sigmoid,120,60,60"
	do
	    python3 Navigation/MA3C-LocalThreads.py --num_slaves=3 --comm_size=10 --param_search="$param"
	    sleep 5
	    mkdir Navigation-10Comm-"$param"-Trial"$trial"
	    mv model train_* Navigation-10Comm-"$param"-Trial"$trial"/
	done
done
