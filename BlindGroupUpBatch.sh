export PYTHONPATH=$(pwd)

for trial in 0
do
	for comm in 10 20
	do
	    python3 BlindGroupUp/A3C-LocalThreads.py --num_slaves=3 --comm_size="$comm"
	    sleep 5
	    mkdir BlindGroupUp-"$comm"Comm-Trial"$trial"
	    mv model train_* BlindGroupUp-"$comm"Comm-Trial"$trial"/
	done
done
