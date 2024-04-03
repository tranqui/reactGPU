#include <pybind11/pybind11.h>
#include <pybind11/eigen.h>
#include "reactor.cuh"

namespace py = pybind11;

template <typename Class>
void define_common_interface(py::class_<Class>& object)
{
    object.def("run", &Class::run, py::arg("nsteps"))
          .def_property_readonly("state", &Class::get_fields, py::return_value_policy::move)
          .def_property_readonly("step", &Class::step)
          .def_property_readonly("time", &Class::time)
          .def_property_readonly("host_parameters",
            [](const Class& self)
            {
                auto params = self.host_parameters();
                Eigen::Array<Scalar,Eigen::Dynamic,1> params_ret(params.size());
                for (int i = 0; i < params.size(); ++i)
                    params_ret(i) = params[i];
                return params_ret;
            }, py::return_value_policy::move)
          .def_property_readonly("device_parameters",
            [](const Class& self)
            {
                auto params = self.device_parameters();
                Eigen::Array<Scalar,Eigen::Dynamic,1> params_ret(params.size());
                for (int i = 0; i < params.size(); ++i)
                    params_ret(i) = params[i];
                return params_ret;
            }, py::return_value_policy::move);
}

PYBIND11_MODULE(reactor, m)
{
    py::register_exception<kernel::CudaError>(m, "CudaException");

    using Cell = Reactor<CellPolarisation>;
    using AmB = Reactor<ActiveModelB>;

    {
        py::class_<Cell> py_class(m, "CellPolarisation");
        py_class.def(py::init<const Cell::InitialState&,
                        Scalar, Scalar, Scalar,
                        Scalar, Scalar,
                        Scalar,
                        int>(),
                py::arg("state"),
                py::arg("dt"), py::arg("dx"), py::arg("dy"),
                py::arg("Du"), py::arg("Dv"),
                py::arg("k"),
                py::arg("current_step")=0)
                .def_property_readonly("u",
                    [](const Cell& self)
                    {
                        return self.get_field<0>();
                    }, py::return_value_policy::move)
                .def_property_readonly("v",
                    [](const Cell& self)
                    {
                        return self.get_field<1>();
                    }, py::return_value_policy::move);
        define_common_interface(py_class);
    }

    {
        py::class_<AmB> py_class(m, "ActiveModelB");
        py_class.def(pybind11::init(
                    [](const AmB::InitialState& state,
                        Scalar dt, Scalar dx, Scalar dy,
                        Scalar Du, Scalar Dv, Scalar Dw,
                        Scalar kappa, Scalar lambda,
                        Scalar X, Scalar Y,
                        int current_step)
                    {
                        auto model = AmB(state, dt, dx, dy,
                                        Du, Dv, Dw, Du, Dv, Dw,
                                        kappa, lambda, X, Y, current_step);
                        return model;
                    }),
                    py::arg("state"),
                    py::arg("dt"), py::arg("dx"), py::arg("dy"),
                    py::arg("Du"), py::arg("Dv"), py::arg("Dw"),
                    py::arg("kappa"), py::arg("lambda"), py::arg("X"), py::arg("Y"),
                    py::arg("current_step")=0)
                .def_static("linear_flux",
                    [](Scalar phi, Scalar Du, Scalar Dv, Scalar Dw,
                    Scalar kappa, Scalar lambda, Scalar X, Scalar Y)
                    {
                        auto device_params = ActiveModelB::device_parameters(Du, Dv, Dw, kappa, lambda, X, Y);
                        std::array<Scalar, device_params.size() - 2> flux_params;
                        flux_params[0] = phi;
                        for (int i = 3; i < device_params.size(); ++i)
                            flux_params[i-2] = device_params[i];
                        auto L = unpack_array(&ActiveModelB::linear_flux, flux_params);
                        Eigen::Matrix2<Scalar> L_ret;
                        L_ret << L[0][0], L[0][1], L[1][0], L[1][1];
                        return L_ret;
                    }, py::return_value_policy::move)
                .def_property_readonly("u",
                    [](const AmB& self)
                    {
                        return self.get_field<0>();
                    }, py::return_value_policy::move)
                .def_property_readonly("v",
                    [](const AmB& self)
                    {
                        return self.get_field<1>();
                    }, py::return_value_policy::move)
                .def_property_readonly("w",
                    [](const AmB& self)
                    {
                        return self.get_field<2>();
                    }, py::return_value_policy::move);
        define_common_interface(py_class);
    }
}
