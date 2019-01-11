export PYTHONPATH=$(pwd)

for trial in 0
do
	for comm in 0 1 2 5
	do
	    python3 Traffic/A3C-LocalThreads.py --num_slaves=3 --comm_size="$comm"
	    sleep 5
	    mkdir Traffic-"$comm"Comm-Trial"$trial"
	    mv *.log model train_* Traffic-"$comm"Comm-Trial"$trial"/
	done
done
