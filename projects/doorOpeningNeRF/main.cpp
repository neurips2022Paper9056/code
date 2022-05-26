#include "doorEnvironment.h"
#include <pybind11/pybind11.h>
#include <torch/csrc/utils/pybind.h>

namespace py = pybind11;

void init_CfgFileParameters() {
  char* argv[2] = {(char*)"rai", (char*)"-python"};
  int argc = 0;
  rai::initCmdLine(argc, argv);
}

PYBIND11_MODULE(doorEnvironment, m) {
  init_CfgFileParameters();
  py::class_<DoorEnvironment>(m, "DoorEnvironment")
    .def(py::init<>())
    .def("initEnvironment", &DoorEnvironment::initEnvironment, "", py::arg("seed") = 0, py::arg("wallOffset") = false)
    .def("checkWithinBounds", &DoorEnvironment::checkWithinBounds)
    .def("watch", &DoorEnvironment::watch)
    .def("getObservation", &DoorEnvironment::getObservation)
    .def("getState", &DoorEnvironment::getState)
    .def("getHighDimState", &DoorEnvironment::getHighDimState)
    .def("getKeypoints", &DoorEnvironment::getKeypoints)
    .def("getStateBounds", &DoorEnvironment::getStateBounds)
    .def("setState", py::overload_cast<const arr&>(&DoorEnvironment::setState))
    .def("setState", py::overload_cast<const torch::Tensor&>(&DoorEnvironment::setState))
    .def("simulateStep", &DoorEnvironment::simulateStep)
    .def("addAlternativeCameraObs", &DoorEnvironment::addAlternativeCameraObs);
}

