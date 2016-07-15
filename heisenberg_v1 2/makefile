GC = g++
GOPT = -O3 

heisenberg:  heisenberg.o crystal.o magnet.o findnRhomb.o vector.o KKR.o
	$(GC) $(GOPT) -o heisenberg heisenberg.o crystal.o magnet.o findnRhomb.o vector.o KKR.o


test: heisenberg_old.o crystal.o magnet.o findnRhomb.o vector.o KKR.o
	$(GC) $(GOPT) -o test heisenberg_old.o crystal.o magnet.o findnRhomb.o vector.o KKR.o


heisenberg_model_nd: heisenberg_model_nd.o crystal.o magnet.o findnRhomb.o vector.o KKR.o
	$(GC) $(GOPT) -o test heisenberg_model_nd.o crystal.o magnet.o findnRhomb.o vector.o KKR.o

heisenberg_old.o: heisenberg_old.cpp heisenberg_old.h magnet.h crystal.h KKR.h
	$(GC) $(GOPT) -c -o heisenberg_old.o heisenberg_old.cpp

heisenberg.o: heisenberg.cpp heisenberg.h magnet.h crystal.h KKR.h
	$(GC) $(GOPT) -c -o heisenberg.o heisenberg.cpp

heisenberg_model_nd.o: heisenberg_model_nd.cpp heisenberg.h magnet.h crystal.h KKR.h
	$(GC) $(GOPT) -c -o heisenberg_model_nd.o heisenberg_model_nd.cpp

crystal.o: crystal.cpp crystal.h
	$(GC) $(GOPT) -c crystal.cpp

magnet.o: magnet.cpp magnet.h KKR.h
	$(GC) $(GOPT) -c magnet.cpp

KKR.o: KKR.cpp KKR.h
	$(GC) $(GOPT) -c KKR.cpp

vector.o: vector.cpp vector.h
	$(GC) $(GOPT) -c vector.cpp

findnRhomb.o: findnRhomb.cpp vector.h
	$(GC) $(GOPT) -c findnRhomb.cpp

clean:
	rm *.o
