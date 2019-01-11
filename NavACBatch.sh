export PYTHONPATH=$(pwd)

for trial in 0 1 2
do
    for comm in 0
	do
	    for ac in 1
		do
            python3 Navigation/MA3C-LocalThreads.py --num_slaves=3 --comm_size="$comm" --critic="$ac"
            sleep 5
            mkdir Navigation-"$comm"Comm-Trial"$trial"
            mv model train_* Navigation-"$comm"Comm-Trial"$trial"/
	    done
	done

	for comm in 5 10 20
	do
	    for ac in 1 2 3
		do
            python3 Navigation/MA3C-LocalThreads.py --num_slaves=3 --comm_size="$comm" --critic="$ac"
            sleep 5
            mkdir Navigation-"$comm"Comm-Trial"$trial"
            mv model train_* Navigation-"$comm"Comm-Trial"$trial"/
	    done
	done
done
