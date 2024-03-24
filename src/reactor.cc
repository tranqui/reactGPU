#include <pybind11/pybind11.h>
#include <pybind11/eigen.h>
#include "reactor.cuh"

namespace py = pybind11;

PYBIND11_MODULE(reactor, m) {
    py::class_<Reactor>(m, "Reactor")
        .def(py::init<const Eigen::Ref<const State>&,
             const Eigen::Ref<const State>&,
             Scalar, Scalar, Scalar,
             Scalar, Scalar, Scalar, size_t>(),
             py::arg("u"), py::arg("v"),
             py::arg("dt"), py::arg("dx"), py::arg("dy"),
             py::arg("Du"), py::arg("Dv"), py::arg("k"),
             py::arg("current_step")=0)
        .def("run", &Reactor::run, py::arg("nsteps"))
        .def_property_readonly("u", &Reactor::get_u, py::return_value_policy::move)
        .def_property_readonly("v", &Reactor::get_v, py::return_value_policy::move)
        .def_property_readonly("step", &Reactor::step)
        .def_property_readonly("time", &Reactor::time);
}
