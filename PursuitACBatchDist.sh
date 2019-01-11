export PYTHONPATH=$(pwd)

thread_number=12
count=`expr $thread_number - 1`

for trial in 0 1 2
do

	for comm in 10 20 5
	do
		for ac in 1 2 3 0
		do
			for i in `seq 0 $count`
			do
			    python3 Pursuit/MA3C-Distributed.py --comm_size="$comm" --slaves_per_url="$thread_number" --urls=localhost --task_index="$i" --critic="$ac" &
			    echo python3 A3C/MA3C-Distributed.py --task_max="$thread_number" --task_index="$i" --critic="$ac" 
			    sleep 5
			done
			wait
		    sleep 5
		    mkdir Pursuit-"$comm"Comm-Trial"$trial"-AC"$ac"
		    mv dist_model train_* Pursuit-"$comm"Comm-Trial"$trial"-AC"$ac"/
		done
	done

	for comm in 0
	do
		for ac in 1 0
		do
			for i in `seq 0 $count`
			do
			    python3 Pursuit/MA3C-Distributed.py --comm_size="$comm" --slaves_per_url="$thread_number" --urls=localhost --task_index="$i" --critic="$ac" &
			    echo python3 A3C/MA3C-Distributed.py --task_max="$thread_number" --task_index="$i" --critic="$ac"
			    sleep 5
			done
			wait
		    sleep 5
		    mkdir Pursuit-"$comm"Comm-Trial"$trial"-AC"$ac"
		    mv dist_model train_* Pursuit-"$comm"Comm-Trial"$trial"-AC"$ac"/
		done
	done
done
