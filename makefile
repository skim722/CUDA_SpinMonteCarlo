GC = g++
GOPT = -O3 
heisenberg_model: heisenberg.o 
	$(GC) $(GOPT) -o heisenberg_model heisenberg.o 

heisenberg.o: heisenberg.cpp heisenberg.h 
	$(GC) $(GOPT) -c -o heisenberg.o heisenberg.cpp

clean:
	rm *.o
