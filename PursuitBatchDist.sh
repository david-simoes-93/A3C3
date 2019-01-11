export PYTHONPATH=$(pwd)

thread_number=12
count=`expr $thread_number - 1`

for trial in 0 1 2
do
	for comm in 0 1 5 10 20
	do
		for i in `seq 0 $count`
		do
		    python3 Pursuit/A3C-Distributed.py --comm_size="$comm" --slaves_per_url="$thread_number" --urls=localhost --task_index="$i" &
		    echo python3 A3C/A3C-Distributed.py --task_max="$thread_number" --task_index="$i"
		    sleep 5
		done
		wait
	    sleep 5
	    mkdir Pursuit-"$comm"Comm-Trial"$trial"
	    mv dist_model train_* Pursuit-"$comm"Comm-Trial"$trial"/
	done
done
