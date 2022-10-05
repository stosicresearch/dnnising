#include <stdlib.h>
#include <stdio.h>
#include <math.h>

/****************** Marsagila's MWC random number generator *******************************/
#define	RMAX	4294967295UL
#define s1new	(s1=(18000*(s1&65535)+(s1>>16)))
#define s2new	(s2=(30903*(s2&65535)+(s2>>16)))
#define DUNI	(((s1new<<16)+(s2new&0xffff))*2.3283064370807973754314699618685e-10)
unsigned long s1 = 1, s2 = 2;
//unsigned long s1 = 3, s2 = 4;
/*****************************************************************************************/

void read(double*** J, int* nodes, int num_layers, char* dirname) {
	FILE* h;
	char fname[256], dname[256];
	char fmt[][64] = {
	"layers.%d.qkv.txt",
	"layers.%d.proj.txt",
	"layers.%d.fc1.txt",
	"layers.%d.fc2.txt" };
	double tmp = 0;

	for (int l = 0; l < num_layers; l++) {
		sprintf(dname, "%s/%s", dirname, fmt[l % 4]);
		sprintf(fname, dname, l / 4);

		h = fopen(fname, "r+");
		for (int i = 0; i < nodes[l]; i++) {
			for (int j = 0; j < nodes[l + 1]; j++) {
				// rows are input nodes, cols are output nodes
				fscanf(h, "%lf", &tmp);	// read in inverted row/column order
				// rows are input nodes, cols are output nodes
				J[l][i][j] = tmp;
			}
		}
		fclose(h);
		printf("%s\n", fname);
	}
}

/*********************************************************************************/

void save(int** S, int* nodes, int num_layers)
{
	FILE* h;
	char fname[256];

	for (int l = 0; l <= num_layers; l++) {
		sprintf(fname, "spins/l%d.dat", l);
		h = fopen(fname, "w+");
		for (int i = 0; i < nodes[l]; i++) {
			fprintf(h, "%d\n", S[l][i]);
		}
		fclose(h);
	}
}

/*********************************************************************************/

int* alloc_nodes(int num_layers, int num_nodes) {
	int* nodes = (int*)calloc(num_layers + 1, sizeof(int));
	for (int l = 0; l < num_layers; l += 4) {
		nodes[l + 0] = num_nodes;
		nodes[l + 1] = num_nodes;
		nodes[l + 2] = num_nodes;
		nodes[l + 3] = num_nodes * 4;
	}
	nodes[num_layers] = num_nodes;
	return nodes;
}

int** alloc_spins(int* nodes, int num_layers) {
	int** s;
	s = (int**)calloc(num_layers + 1, sizeof(int*));
	for (int l = 0; l <= num_layers; l++) {
		s[l] = (int*)calloc(nodes[l], sizeof(int));

		for (int i = 0; i < nodes[l]; i++) {
			s[l][i] = 1;
		}
	}
	return s;
}

double*** alloc_couplings(int* nodes, int num_layers)
{
	double** dmem, *** dlayer, *** data;

	dmem = (double**)calloc(num_layers + 1, sizeof(double*));
	dlayer = (double***)calloc(num_layers + 1, sizeof(double**));
	data = (double***)calloc(num_layers, sizeof(double**));

	for (int l = 0; l < num_layers; l++) {
		// allocate memory for the whole layer (matrix)
		dmem[l] = (double*)calloc(nodes[l] * nodes[l + 1], sizeof(double));

		// allocate memory to store row addresses
		dlayer[l] = (double**)calloc(nodes[l], sizeof(double*));
		// store position of each row
		for (int i = 0; i < nodes[l]; i++)
			dlayer[l][i] = dmem[l] + nodes[l + 1] * i;

		data[l] = dlayer[l];
	}
	return data;
}

void alloc(double**** J, int*** S, int** nodes, int num_layers, int num_nodes) {
	*nodes = alloc_nodes(num_layers, num_nodes);
	*S = alloc_spins(*nodes, num_layers);
	*J = alloc_couplings(*nodes, num_layers);
}

/*********************************************************************************/

void shuffle(double*** J, int* nodes, int num_layers, int num_reps) {
	for (int r = 0; r < num_reps; r++) {
		for (int l = 0; l < num_layers; l++) {
			for (int i = 0; i < nodes[l]; i++) {
				for (int j = 0; j < nodes[l + 1]; j++) {
					int l1 = DUNI * num_layers;
					int i1 = DUNI * nodes[l1];
					int j1 = DUNI * nodes[l1 + 1];
					double tmp = J[l][i][j];
					J[l][i][j] = J[l1][i1][j1];
					J[l1][i1][j1] = tmp;
				}
			}
		}
	}
}

/*********************************************************************************/

int spins(int* nodes, int num_layers) {
	int total = 0;
	for (int l = 0; l <= num_layers; l++) {
		total += nodes[l];
	}
	return total;
}

int bonds(int* nodes, int num_layers) {
	int total = 0;
	for (int l = 0; l < num_layers; l++) {
		total += (nodes[l] * nodes[l + 1]);
	}
	return total;
}

/*********************************************************************************/

double energy(double*** J, int** S, int* nodes, int num_layers) {
	double e = 0;
	int s1, s2;
	for (int l = 0; l < num_layers; l++) {
		for (int i = 0; i < nodes[l]; i++) {
			s1 = S[l][i];
			for (int j = 0; j < nodes[l + 1]; j++) {
				s2 = S[l + 1][j];
				e += s1 == s2 ? -J[l][i][j] : +J[l][i][j];
			}
		}
	}
	return e;
}

/*********************************************************************************/

double mcmc_step(double e, double*** J, int** S, int* nodes, int num_layers) {
	int lmax, imax, l, i, steps = 0;
	int ss;
	double ee, de;
	// pick a random spin
	lmax = num_layers + 1;
	do {
		l = DUNI * lmax;
	} while (l >= lmax);
	imax = nodes[l];
	do {
		i = DUNI * imax;
	} while (i >= imax);
	ss = S[l][i];

	// calculate energy contribution for s[l][i]
	de = 0;
	// energy contributions from previous layer
	if (l > 0) {
		for (int j = 0; j < nodes[l - 1]; j++)
			de += ss == S[l - 1][j] ? -J[l - 1][j][i] : J[l - 1][j][i];
	}
	// energy contributions from next layer
	if (l < num_layers) {
		for (int j = 0; j < nodes[l + 1]; j++)
			de += ss == S[l + 1][j] ? -J[l][i][j] : J[l][i][j];
	}

	// energy if ss was flipped
	ee = e - 2 * de;

	// flip spin if energy is lower (quenching)
	if (de > 0) {
		S[l][i] = -S[l][i];
		e = ee;
	}
	return e;
}

double mcmc(double*** J, int** S, int* nodes, int num_layers, int max_steps) {
	double e;
	e = energy(J, S, nodes, num_layers);
	for (int n = 0; n < max_steps; n++) {
		e = mcmc_step(e, J, S, nodes, num_layers);
		if (n % 10000 == 0) {
			printf("steps: %d\tenergy: %f\n", n, e);
		}
	}
	return e;
}

/*********************************************************************************/