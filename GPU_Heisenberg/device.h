
__device__ inline Spin2D makeDelta(dtype delta_max,dtype randomValue1,dtype randomValue2)
{
	Spin2D delta;
	//Choose a random vector, uniformly distributed inside unit sphere
	delta.x = 2.*randomValue1-1.0;
	delta.y = 2.*randomValue2-1.0;
	//Set to default vector if 0-vector was made.
	delta.x = ((delta.x==0) and (delta.y==0)) ? 1 : delta.x;
	//Scale delta
	dtype norm = sqrt(delta.x*delta.x + delta.y*delta.y);
	delta.x *= delta_max/norm;
	delta.y *= delta_max/norm;
	return delta;
}

__device__ inline void try_move2D(Spin2D& try_spin,Spin2D& delta)
{
	//
	try_spin.x += delta.x;
	try_spin.y += delta.y;
	dtype norm  = sqrt(try_spin.x*try_spin.x + try_spin.y*try_spin.y);
	try_spin.x /= norm;
	try_spin.y /= norm;
}

__device__ inline dtype dEnergydSpin2D(Spin2D spin, 
	Spin2D neighbour0, Spin2D neighbour1, Spin2D neighbour2, Spin2D neighbour3, 
	dtype J, dtype Hx, dtype Hy)
{
	dtype energyI = 
		spin.x*neighbour0.x + spin.y*neighbour0.y + 
		spin.x*neighbour1.x + spin.y*neighbour1.y + 
		spin.x*neighbour2.x + spin.y*neighbour2.y + 
		spin.x*neighbour3.x + spin.y*neighbour3.y;
	energyI *= J;
	energyI -= 2 * ( spin.x*Hx + spin.y*Hy );
	return -energyI;
}

__device__ inline int determineOffset(int i,int set0,int globalx,int maxx)
{//i is row number in line (0 to LINE_LENGTH-1 inclusive)
	// int oddrow = i & 1;
	// int offset = oddrow ? (set0 ? 1 : -1) : (set0 ? -1 : 1);
	// bool set1 = not set0;
	// bool evenrow = not oddrow;
	// bool needsLeft = (set0 and evenrow) or (set1 and oddrow);
	// bool needsRight = not needsLeft;
	// offset = (globalx==0 and needsLeft) ? maxx : offset;
	// offset = (globalx==maxx and needsRight) ? -maxx : offset;
	// return offset;
	int offset = (i & 1) ? (set0 ? 1 : -1) : (set0 ? -1 : 1);
	bool needsLeft = (set0 and (not (i & 1))) or ((not set0) and (i & 1));
	offset = (globalx==0 and needsLeft) ? maxx : offset;
	offset = (globalx==maxx and (not needsLeft)) ? -maxx : offset;
	return offset;
}