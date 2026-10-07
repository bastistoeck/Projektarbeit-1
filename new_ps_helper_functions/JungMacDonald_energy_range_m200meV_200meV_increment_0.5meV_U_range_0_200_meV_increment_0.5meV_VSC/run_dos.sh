#!/bin/bash

rm timer.dat

date >> timer.dat

rm out_idos.dat


for i in {0,99,199,299,399}   

do
	rm -r it_$i

	mkdir it_$i

	cp dos_numpy.py it_$i

	cd it_$i

	echo "iteration $i" >> it_counter.txt

	rm out_dos.dat

	date >> out_dos.dat
	
	mpiexec -np 4 python3 dos_numpy.py >> out_dos.dat

	date >> out_dos.dat

	rm dos_numpy.py

	cd ..
	
	rm -r it_$i

done

mpiexec -np 4 python3 get_idos.py >> out_idos.dat

date >> timer.dat
