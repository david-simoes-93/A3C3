export PYTHONPATH=$(pwd)

for trial in 0
do
	for comm in 0
	do
	    python3 Traffic/MA3C-LocalThreads.py --num_slaves=3 --comm_size="$comm"
	    sleep 5
	    mkdir Traffic-"$comm"Comm-Trial"$trial"
	    mv *.log model train_* Traffic-"$comm"Comm-Trial"$trial"/
	done
done
