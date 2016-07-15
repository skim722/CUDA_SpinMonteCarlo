GC = g++
GOPT = -O3 
heisenberg_model: heisenberg.o crystal.o magnet.o findnRhomb.o vector.o
	$(GC) $(GOPT) -o heisenberg_model heisenberg.o crystal.o magnet.o findnRhomb.o vector.o

heisenberg.o: heisenberg.cpp heisenberg.h magnet.h crystal.h
	$(GC) $(GOPT) -c -o heisenberg.o heisenberg.cpp

crystal.o: crystal.cpp crystal.h
	$(GC) $(GOPT) -c crystal.cpp

magnet.o: magnet.cpp magnet.h crystal.h
	$(GC) $(GOPT) -c magnet.cpp

vector.o: vector.cpp vector.h
	$(GC) $(GOPT) -c vector.cpp

findnRhomb.o: findnRhomb.cpp vector.h
	$(GC) $(GOPT) -c findnRhomb.cpp

clean:
	rm *.o
