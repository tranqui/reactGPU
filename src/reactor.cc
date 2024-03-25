#include <pybind11/pybind11.h>
#include <pybind11/eigen.h>
#include "reactor.cuh"

namespace py = pybind11;

PYBIND11_MODULE(reactor, m)
{
    py::register_exception<kernel::CudaError>(m, "CudaException");

    using Cell = Reactor<CellPolarisation>;
    py::class_<Cell>(m, "CellPolarisation")
        .def(py::init<const Cell::InitialState&,
                      Scalar, Scalar, Scalar,
                      Scalar, Scalar,
                      Scalar,
                      size_t>(),
             py::arg("state"),
             py::arg("dt"), py::arg("dx"), py::arg("dy"),
             py::arg("Du"), py::arg("Dv"),
             py::arg("k"),
             py::arg("current_step")=0)
        .def("run", &Cell::run, py::arg("nsteps"))
        .def_property_readonly("state", &Cell::get_fields, py::return_value_policy::move)
        .def_property_readonly("u", [](const Cell& self) {
                return self.get_field<0>();
            }, py::return_value_policy::move)
        .def_property_readonly("v", [](const Cell& self) {
                return self.get_field<1>();
            }, py::return_value_policy::move)
        .def_property_readonly("step", &Cell::step)
        .def_property_readonly("time", &Cell::time);

    using Toy = Reactor<ToyModel>;
    py::class_<Toy>(m, "ToyModel")
        .def(py::init<const Toy::InitialState&,
                      Scalar, Scalar, Scalar,
                      Scalar, Scalar, Scalar,
                      Scalar,
                      size_t>(),
             py::arg("state"),
             py::arg("dt"), py::arg("dx"), py::arg("dy"),
             py::arg("Du"), py::arg("Dv"), py::arg("Dw"),
             py::arg("k"),
             py::arg("current_step")=0)
        .def("run", &Toy::run, py::arg("nsteps"))
        .def_property_readonly("state", &Toy::get_fields, py::return_value_policy::move)
        .def_property_readonly("u", [](const Toy& self) {
                return self.get_field<0>();
            }, py::return_value_policy::move)
        .def_property_readonly("v", [](const Toy& self) {
                return self.get_field<1>();
            }, py::return_value_policy::move)
        .def_property_readonly("step", &Toy::step)
        .def_property_readonly("time", &Toy::time);

}
