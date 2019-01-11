export PYTHONPATH=$(pwd)

#for trial in 0
#do
#	for comm in 20
#	do
#		for ac in 1
#		do
#		    python3 BlindGroupUp/A3C-LocalThreads.py --num_slaves=3 --comm_size="$comm" --critic="$ac" --comm_delivery_failure_chance=0.5
#		    sleep 5
#		    mkdir BlindGroupUp-"$comm"Comm-AC"$ac"-Trial"$trial"
#		    mv model train_* BlindGroupUp-"$comm"Comm-AC"$ac"-Trial"$trial"/
#		done
#	done
#done

#for trial in 1
#do
#	for comm in 20
#	do
#		for ac in 1
#		do
#		    python3 BlindGroupUp/A3C-LocalThreads.py --num_slaves=3 --comm_size="$comm" --critic="$ac" --comm_gaussian_noise=0.5
#		    sleep 5
#		    mkdir BlindGroupUp-"$comm"Comm-AC"$ac"-Trial"$trial"
#		    mv model train_* BlindGroupUp-"$comm"Comm-AC"$ac"-Trial"$trial"/
#		done
#	done
#done

#for trial in 2
#do
#	for comm in 20
#	do
#		for ac in 1
#		do
#		    python3 BlindGroupUp/A3C-LocalThreads.py --num_slaves=3 --comm_size="$comm" --critic="$ac" --comm_jumble_chance=0.5
#		    sleep 5
#		    mkdir BlindGroupUp-"$comm"Comm-AC"$ac"-Trial"$trial"
#		    mv model train_* BlindGroupUp-"$comm"Comm-AC"$ac"-Trial"$trial"/
#		done
#	done
#done

for trial in 3
do
	for comm in 20
	do
		for ac in 1
		do
		    python3 BlindGroupUp/A3C-LocalThreads.py --num_slaves=3 --comm_size="$comm" --critic="$ac" --comm_delivery_failure_chance=0.5 --comm_gaussian_noise=0.5 --comm_jumble_chance=0.5
		    sleep 5
		    mkdir BlindGroupUp-"$comm"Comm-AC"$ac"-Trial"$trial"
		    mv model train_* BlindGroupUp-"$comm"Comm-AC"$ac"-Trial"$trial"/
		done
	done
done

exit

for trial in 4
do
	for comm in 20
	do
		for ac in 1
		do
		    python3 BlindGroupUp/A3C-LocalThreads.py --num_slaves=3 --comm_size="$comm" --critic="$ac"
		    sleep 5
		    mkdir BlindGroupUp-"$comm"Comm-AC"$ac"-Trial"$trial"
		    mv model train_* BlindGroupUp-"$comm"Comm-AC"$ac"-Trial"$trial"/
		done
	done
done
