#!/bin/bash
# FC Portugal 3D 2008 binary

#echo "Launch simulator"
#simspark &
#sleep 3
host="localhost"
if [ $# -gt 0 ]; then
	host="${1}"  
    echo host=$host
fi


echo "Launch 1"
build/fcpagent -u 1 -r 0 -h ${host} $2 >/dev/null 2>&1 &
sleep 1 
echo "Launch 2"
build/fcpagent -u 2 -r 0 -h ${host} $2 >/dev/null 2>&1 &
sleep 1
echo "Launch 3"
build/fcpagent -u 3 -r 2 -h ${host} $2 >/dev/null 2>&1 &
sleep 1
echo "Launch 4"
build/fcpagent -u 4 -r 4 -h ${host} >/dev/null 2>&1 &
sleep 1
echo "Launch 5"
build/fcpagent -u 5 -r 4 -h ${host} $2 >/dev/null 2>&1 &
sleep 1
echo "Launch 6"
build/fcpagent -u 6 -r 2 -h ${host} $2 >/dev/null 2>&1 &
sleep 1
echo "Launch 7"
build/fcpagent -u 7 -r 4 -h ${host} $2 >/dev/null 2>&1 &
sleep 1
echo "Launch 8"
build/fcpagent -u 8 -r 4 -h ${host} >/dev/null 2>&1 &
sleep 1
echo "Launch 9"
build/fcpagent -u 9 -r 4 -h ${host} $2 >/dev/null 2>&1 &
sleep 1
echo "Launch 10"
build/fcpagent -u 10 -r 4 -h ${host} $2 >/dev/null 2>&1 &
sleep 1
echo "Launch 11"
build/fcpagent -u 11 -r 4  -h ${host} $2 >/dev/null 2>&1 &
