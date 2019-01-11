export PYTHONPATH=$(pwd)

thread_number=12
count=`expr $thread_number - 1`

for trial in 0
do

	for comm in 10
	do
		for ac in 0 3
		do
			for i in `seq 0 $count`
			do
			    python3 Pursuit/A3C-Distributed.py --comm_size="$comm" --slaves_per_url="$thread_number" --urls=localhost --task_index="$i" --critic="$ac" --comm_delivery_failure_chance=0.5 &
			    echo python3 A3C/A3C-Distributed.py --task_max="$thread_number" --task_index="$i" --critic="$ac" 
			    sleep 5
			done
			wait
		    sleep 5
		    mkdir Pursuit-"$comm"Comm-Trial"$trial"-AC"$ac"
		    mv dist_model train_* Pursuit-"$comm"Comm-Trial"$trial"-AC"$ac"/
		done
	done
done

for trial in 1
do

	for comm in 10
	do
		for ac in 0 3
		do
			for i in `seq 0 $count`
			do
			    python3 Pursuit/A3C-Distributed.py --comm_size="$comm" --slaves_per_url="$thread_number" --urls=localhost --task_index="$i" --critic="$ac" --comm_gaussian_noise=0.5 &
			    echo python3 A3C/A3C-Distributed.py --task_max="$thread_number" --task_index="$i" --critic="$ac" 
			    sleep 5
			done
			wait
		    sleep 5
		    mkdir Pursuit-"$comm"Comm-Trial"$trial"-AC"$ac"
		    mv dist_model train_* Pursuit-"$comm"Comm-Trial"$trial"-AC"$ac"/
		done
	done
done


for trial in 2
do

	for comm in 10
	do
		for ac in 0 3
		do
			for i in `seq 0 $count`
			do
			    python3 Pursuit/A3C-Distributed.py --comm_size="$comm" --slaves_per_url="$thread_number" --urls=localhost --task_index="$i" --critic="$ac" --comm_jumble_chance=0.5 &
			    echo python3 A3C/A3C-Distributed.py --task_max="$thread_number" --task_index="$i" --critic="$ac" 
			    sleep 5
			done
			wait
		    sleep 5
		    mkdir Pursuit-"$comm"Comm-Trial"$trial"-AC"$ac"
		    mv dist_model train_* Pursuit-"$comm"Comm-Trial"$trial"-AC"$ac"/
		done
	done
done

for trial in 3
do

	for comm in 10
	do
		for ac in 0 3
		do
			for i in `seq 0 $count`
			do
			    python3 Pursuit/A3C-Distributed.py --comm_size="$comm" --slaves_per_url="$thread_number" --urls=localhost --task_index="$i" --critic="$ac" --comm_delivery_failure_chance=0.5 --comm_gaussian_noise=0.5 --comm_jumble_chance=0.5 &
			    echo python3 A3C/A3C-Distributed.py --task_max="$thread_number" --task_index="$i" --critic="$ac" 
			    sleep 5
			done
			wait
		    sleep 5
		    mkdir Pursuit-"$comm"Comm-Trial"$trial"-AC"$ac"
		    mv dist_model train_* Pursuit-"$comm"Comm-Trial"$trial"-AC"$ac"/
		done
	done
done

for trial in 4
do

	for comm in 10
	do
		for ac in 0 3
		do
			for i in `seq 0 $count`
			do
			    python3 Pursuit/A3C-Distributed.py --comm_size="$comm" --slaves_per_url="$thread_number" --urls=localhost --task_index="$i" --critic="$ac" &
			    echo python3 A3C/A3C-Distributed.py --task_max="$thread_number" --task_index="$i" --critic="$ac" 
			    sleep 5
			done
			wait
		    sleep 5
		    mkdir Pursuit-"$comm"Comm-Trial"$trial"-AC"$ac"
		    mv dist_model train_* Pursuit-"$comm"Comm-Trial"$trial"-AC"$ac"/
		done
	done
done
