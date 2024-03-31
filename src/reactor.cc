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
                      int>(),
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
        .def(pybind11::init(
            [](const Toy::InitialState& state,
                Scalar dt, Scalar dx, Scalar dy,
                Scalar Du, Scalar Dv, Scalar Dw,
                Scalar kappa, Scalar lambda,
                Scalar X, Scalar Y,
                int current_step)
            {
                auto model = Toy(state, dt, dx, dy,
                                 Du, Dv, Dw, Du, Dv, Dw,
                                 kappa, lambda, X, Y, current_step);
                return model;
            }),
             py::arg("state"),
             py::arg("dt"), py::arg("dx"), py::arg("dy"),
             py::arg("Du"), py::arg("Dv"), py::arg("Dw"),
             py::arg("kappa"), py::arg("lambda"), py::arg("X"), py::arg("Y"),
             py::arg("current_step")=0)
        .def("run", &Toy::run, py::arg("nsteps"))
        .def_static("linear_flux",
            [](Scalar phi, Scalar Du, Scalar Dv, Scalar Dw,
               Scalar kappa, Scalar lambda, Scalar X, Scalar Y)
            {
                auto device_params = ToyModel::device_parameters(Du, Dv, Dw, kappa, lambda, X, Y);
                std::array<Scalar, device_params.size() - 2> flux_params;
                flux_params[0] = phi;
                for (int i = 3; i < device_params.size(); ++i)
                    flux_params[i-2] = device_params[i];
                auto L = unpack_array(&ToyModel::linear_flux, flux_params);
                Eigen::Matrix2<Scalar> L_ret;
                L_ret << L[0][0], L[0][1], L[1][0], L[1][1];
                return L_ret;
            }, py::return_value_policy::move)
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
