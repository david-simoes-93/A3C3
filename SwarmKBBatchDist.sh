export PYTHONPATH=$(pwd)

thread_number=12
count=`expr $thread_number - 1`

for trial in 0
do
	for comm in 2
	do
		for i in `seq 0 $count`
		do
		    python3 KiloBotsSwarm/MA3C-Distributed.py --comm_size="$comm" --slaves_per_url="$thread_number" --urls=localhost --task_index="$i" &
		    echo python3 KiloBotsSwarm/MA3C-Distributed.py --task_max="$thread_number" --task_index="$i"
		    sleep 5
		done
		wait
	    sleep 5
	    mkdir KiloBotsSwarm-"$comm"Comm-Trial"$trial"
	    mv dist_model train_* KiloBotsSwarm-"$comm"Comm-Trial"$trial"/
	done
done

for trial in 1
do
	for comm in 2
	do
		for i in `seq 0 $count`
		do
		    python3 KiloBotsSwarm/MA3C-Distributed.py --swarm_type=ordered --comm_size="$comm" --slaves_per_url="$thread_number" --urls=localhost --task_index="$i" &
		    echo python3 KiloBotsSwarm/MA3C-Distributed.py --task_max="$thread_number" --task_index="$i"
		    sleep 5
		done
		wait
	    sleep 5
	    mkdir KiloBotsSwarm-"$comm"Comm-Trial"$trial"ordered
	    mv dist_model train_* KiloBotsSwarm-"$comm"Comm-Trial"$trial"ordered/
	done
done

for trial in 2
do
	for comm in 2
	do
		for i in `seq 0 $count`
		do
		    python3 KiloBotsSwarm/MA3C-Distributed.py --swarm_type=max --comm_size="$comm" --slaves_per_url="$thread_number" --urls=localhost --task_index="$i" &
		    echo python3 KiloBotsSwarm/MA3C-Distributed.py --task_max="$thread_number" --task_index="$i"
		    sleep 5
		done
		wait
	    sleep 5
	    mkdir KiloBotsSwarm-"$comm"Comm-Trial"$trial"max
	    mv dist_model train_* KiloBotsSwarm-"$comm"Comm-Trial"$trial"max/
	done
done

for trial in 3
do
	for comm in 2
	do
		for i in `seq 0 $count`
		do
		    python3 KiloBotsSwarm/MA3C-Distributed.py --swarm_type=mean --comm_size="$comm" --slaves_per_url="$thread_number" --urls=localhost --task_index="$i" &
		    echo python3 KiloBotsSwarm/MA3C-Distributed.py --task_max="$thread_number" --task_index="$i"
		    sleep 5
		done
		wait
	    sleep 5
	    mkdir KiloBotsSwarm-"$comm"Comm-Trial"$trial"mean
	    mv dist_model train_* KiloBotsSwarm-"$comm"Comm-Trial"$trial"mean/
	done
done

for trial in 4
do
	for comm in 2
	do
		for i in `seq 0 $count`
		do
		    python3 KiloBotsSwarm/MA3C-Distributed.py --swarm_type=softmax --comm_size="$comm" --slaves_per_url="$thread_number" --urls=localhost --task_index="$i" &
		    echo python3 KiloBotsSwarm/MA3C-Distributed.py --task_max="$thread_number" --task_index="$i"
		    sleep 5
		done
		wait
	    sleep 5
	    mkdir KiloBotsSwarm-"$comm"Comm-Trial"$trial"softmax
	    mv dist_model train_* KiloBotsSwarm-"$comm"Comm-Trial"$trial"softmax/
	done
done