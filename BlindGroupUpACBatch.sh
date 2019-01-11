export PYTHONPATH=$(pwd)

for trial in 0 1 2
do
	for comm in 0
	do
		for ac in 1
		do
		    python3 BlindGroupUp/MA3C-LocalThreadsAC.py --num_slaves=3 --comm_size="$comm" --critic="$ac"
		    sleep 5
		    mkdir BlindGroupUp-"$comm"Comm-Trial"$trial"-AC"$ac"
		    mv model train_* BlindGroupUp-"$comm"Comm-Trial"$trial"-AC"$ac"/
		done
	done
	for comm in 5 10 20
	do
		for ac in 1 2 3
		do
		    python3 BlindGroupUp/MA3C-LocalThreadsAC.py --num_slaves=3 --comm_size="$comm" --critic="$ac"
		    sleep 5
		    mkdir BlindGroupUp-"$comm"Comm-Trial"$trial"-AC"$ac"
		    mv model train_* BlindGroupUp-"$comm"Comm-Trial"$trial"-AC"$ac"/
		done
	done
done
