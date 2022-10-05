#include <stdlib.h>
#include <stdio.h>
#include <math.h>
#include "dnnising.h"

/*********************************************************************************/

void main()
{
	char dirname[256] = "opt-125m";
	int hidden_size = 768;
	int num_transformer_layers = 12;
	int num_nodes = hidden_size;
	int num_layers = num_transformer_layers * 4;
	int max_steps = 1000000;
	int num_reps = 10;

	double*** J;
	int** S, *nodes;

	double e;

	printf("Allocate.\n");
	alloc(&J, &S, &nodes, num_layers, num_nodes);
	printf("Spins: %d, Bonds: %d\n", spins(nodes, num_layers), bonds(nodes, num_layers));

	printf("Read.\n");
	read(J, nodes, num_layers, dirname);

	//printf("Shuffle.\n");
	//shuffle(J, nodes, num_layers, num_reps);

	printf("MCMC\n");
	e = mcmc(J, S, nodes, num_layers, max_steps);
	printf("Final energy: %f\n", e);

	save(S, nodes, num_layers);
}