#include <limits>
#include <stdio.h>

int main()
{
	unsigned int add = 10;
	unsigned int offset = 5;
	unsigned int maxint = std::numeric_limits<unsigned int>::max();
	unsigned int test = maxint-offset;
	//printf("Test: %d, %d\n",test+add,RAND_MAX);
	printf("unsigned int max: %u\n",maxint);
}