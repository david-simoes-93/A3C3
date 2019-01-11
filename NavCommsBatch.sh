export PYTHONPATH=$(pwd)

for trial in 0
do
	for comm in 20
	do
		for ac in 3
		do
		    python3 Navigation/MA3C-LocalThreads.py --num_slaves=3 --comm_size="$comm" --comm_delivery_failure_chance=0.5
		    sleep 5
		    mkdir Navigation-"$comm"Comm-AC"$ac"-Trial"$trial"-Loss
		    mv model train_* Navigation-"$comm"Comm-AC"$ac"-Trial"$trial"-Loss/
		done
	done
done

for trial in 1
do
	for comm in 20
	do
		for ac in 3
		do
		    python3 Navigation/MA3C-LocalThreads.py --num_slaves=3 --comm_size="$comm" --comm_gaussian_noise=0.5
		    sleep 5
		    mkdir Navigation-"$comm"Comm-AC"$ac"-Trial"$trial"-Noise
		    mv model train_* Navigation-"$comm"Comm-AC"$ac"-Trial"$trial"-Noise/
		done
	done
done

for trial in 2
do
	for comm in 20
	do
		for ac in 3
		do
		    python3 Navigation/MA3C-LocalThreads.py --num_slaves=3 --comm_size="$comm" --comm_jumble_chance=0.5
		    sleep 5
		    mkdir Navigation-"$comm"Comm-AC"$ac"-Trial"$trial"-Jumble
		    mv model train_* Navigation-"$comm"Comm-AC"$ac"-Trial"$trial"/-Jumble
		done
	done
done

for trial in 3
do
	for comm in 20
	do
		for ac in 3
		do
		    python3 Navigation/MA3C-LocalThreads.py --num_slaves=3 --comm_size="$comm" --comm_delivery_failure_chance=0.5 --comm_gaussian_noise=0.5 --comm_jumble_chance=0.5
		    sleep 5
		    mkdir Navigation-"$comm"Comm-AC"$ac"-Trial"$trial"-All
		    mv model train_* Navigation-"$comm"Comm-AC"$ac"-Trial"$trial"-All/
		done
	done
done

for trial in 4
do
	for comm in 20
	do
		for ac in 3
		do
		    python3 Navigation/MA3C-LocalThreads.py --num_slaves=3 --comm_size="$comm"
		    sleep 5
		    mkdir Navigation-"$comm"Comm-AC"$ac"-Trial"$trial"-None
		    mv model train_* Navigation-"$comm"Comm-AC"$ac"-Trial"$trial"-None/
		done
	done
done