export PYTHONPATH=$(pwd)

for trial in 0
do
	for comm in 0 1 5 10 20
	do
	    python3 Navigation/A3C-LocalThreads.py --num_slaves=3 --comm_size="$comm"
	    sleep 5
	    mkdir Navigation-"$comm"Comm-Trial"$trial"
	    mv model train_* Navigation-"$comm"Comm-Trial"$trial"/
	done
done
