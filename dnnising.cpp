#include <pybind11/pybind11.h>
#include <pybind11/numpy.h>
#include "dnnising.h"

namespace py = pybind11;

double*** J;
int** S, * nodes;
int num_layers;

void py_alloc(int layers, int num_nodes) {
    num_layers = layers;
    alloc(&J, &S, &nodes, num_layers, num_nodes);
}

void py_read(char* dirname) {
    read(J, nodes, num_layers, dirname);
}

void py_shuffle(int num_reps) {
    shuffle(J, nodes, num_layers, num_reps);
}

int py_spins() {
    return spins(nodes, num_layers);
}

int py_bonds() {
    return bonds(nodes, num_layers);
}

double py_energy() {
    return energy(J, S, nodes, num_layers);
}

double py_mcmc_step(double e) {
    e = mcmc_step(e, J, S, nodes, num_layers);
    return e;
}

double py_mcmc(int max_steps) {
    return mcmc(J, S, nodes, num_layers, max_steps);
}

void py_save() {
    save(S, nodes, num_layers);
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m)
{
    m.def("py_alloc", &py_alloc, "py_alloc");
    m.def("py_read", &py_read, "py_read");
    m.def("py_shuffle", &py_shuffle, "py_shuffle");
    m.def("py_spins", &py_spins, "py_spins");
    m.def("py_bonds", &py_bonds, "py_bonds");
    m.def("py_energy", &py_energy, "py_energy");
    m.def("py_mcmc_step", &py_mcmc_step, "py_mcmc_step");
    m.def("py_mcmc", &py_mcmc, "py_mcmc");
    m.def("py_save", &py_save, "py_save");
}