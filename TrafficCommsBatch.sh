export PYTHONPATH=$(pwd)

for trial in 0
do
	for comm in 5
	do
		for ac in 0 1 2 3
		do
		    python3 Traffic/MA3C-LocalThreads.py --num_slaves=3 --comm_size="$comm" --critic="$ac" --comm_delivery_failure_chance=0.5
		    mkdir Traffic-"$comm"Comm-AC"$ac"-Trial"$trial"-Loss
		    mv model train_* Traffic-"$comm"Comm-AC"$ac"-Trial"$trial"-Loss/
		done
	done
done

for trial in 1 
do
	for comm in 5
	do
		for ac in 0 1 2 3
		do
		    python3 Traffic/MA3C-LocalThreads.py --num_slaves=3 --comm_size="$comm" --critic="$ac" --comm_gaussian_noise=0.5
		    sleep 5
		    mkdir Traffic-"$comm"Comm-AC"$ac"-Trial"$trial"-Noise
		    mv model train_* Traffic-"$comm"Comm-AC"$ac"-Trial"$trial"-Noise/
		done
	done
done

for trial in 2
do
	for comm in 5
	do
		for ac in 0 1 2 3
		do
		    python3 Traffic/MA3C-LocalThreads.py --num_slaves=3 --comm_size="$comm" --critic="$ac" --comm_jumble_chance=0.5
		    sleep 5
		    mkdir Traffic-"$comm"Comm-AC"$ac"-Trial"$trial"-Jumble
		    mv model train_* Traffic-"$comm"Comm-AC"$ac"-Trial"$trial"-Jumble/
		done
	done
done

for trial in 3
do
	for comm in 5
	do
		for ac in 0 1 2 3
		do
		    python3 Traffic/MA3C-LocalThreads.py --num_slaves=3 --comm_size="$comm" --critic="$ac" --comm_delivery_failure_chance=0.5 --comm_gaussian_noise=0.5 --comm_jumble_chance=0.5
		    sleep 5
		    mkdir Traffic-"$comm"Comm-AC"$ac"-Trial"$trial"-All
		    mv model train_* Traffic-"$comm"Comm-AC"$ac"-Trial"$trial"-All/
		done
	done
done

for trial in 4
do
	for comm in 5
	do
		for ac in 0 1 2 3
		do
		    python3 Traffic/MA3C-LocalThreads.py --num_slaves=3 --comm_size="$comm" --critic="$ac"
		    sleep 5
		    mkdir Traffic-"$comm"Comm-AC"$ac"-Trial"$trial"-None
		    mv model train_* Traffic-"$comm"Comm-AC"$ac"-Trial"$trial"-None/
		done
	done
done