#include <pybind11/pybind11.h>
#include <pybind11/eigen.h>

#include "svm.h"

namespace py = pybind11;
using namespace pybind11::literals;

PYBIND11_MODULE(asksvm_utils, m) {
    m.doc() = "Example";

    m.def("computeSupport", &computeSupport, "Computes support vector", "K"_a, "Y"_a, "gamma"_a = 1);
}