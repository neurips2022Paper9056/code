#undef LOG
#undef CHECK
#undef CHECK_EQ
#undef CHECK_GE
#undef CHECK_LE
#include <torch/script.h>
#undef LOG
#undef CHECK
#undef CHECK_EQ
#undef CHECK_GE
#undef CHECK_LE


#include <Core/util.h>
#include <Kin/kin.h>
#include <Kin/frame.h>
#include <Kin/cameraview.h>
#include <Kin/F_collisions.h>
#include <Kin/simulation.h>
#include <Kin/ImSModule.h>
#include "pushingEnvironment.h"
#include "cameraObs.h"

#include <pybind11/pybind11.h>
#include <torch/csrc/utils/pybind.h>

namespace py = pybind11;

torch::Device ImS::defaultTorchDevice("cuda");

void init_CfgFileParameters() {
  char* argv[2] = {(char*)"rai", (char*)"-python"};
  int argc = 2;
  rai::initCmdLine(argc, argv);
}

PYBIND11_MODULE(pushingEnvironment, m) {
  init_CfgFileParameters();
  py::class_<PushingEnvironment>(m, "PushingEnvironment")
    .def(py::init<>())
    .def("initEnvironment", &PushingEnvironment::initEnvironment, "", py::arg("seed") = 0, py::arg("pos_seed") = -1)
    .def("checkIfAllWithinBounds", &PushingEnvironment::checkIfAllWithinBounds)
    .def("checkIfObjectsWithinBounds", &PushingEnvironment::checkIfObjectsWithinBounds)
    .def("checkIfPusherWithinBounds", &PushingEnvironment::checkIfPusherWithinBounds)
    .def("watch", &PushingEnvironment::watch)
    .def("getObservation", &PushingEnvironment::getObservation)
    .def("getState", &PushingEnvironment::getState)
    .def("getHighDimState", &PushingEnvironment::getHighDimState)
    .def("getKeypoints", &PushingEnvironment::getKeypoints)
    .def("addObject", &PushingEnvironment::addObject)
    .def("setObjectPose", py::overload_cast<const char*, double, double>(&PushingEnvironment::setObjectPose))
    .def("setObjectPose", py::overload_cast<const char*, double, double, double>(&PushingEnvironment::setObjectPose))
    .def("initSimulation", &PushingEnvironment::initSimulation)
    .def("simulateStep", &PushingEnvironment::simulateStep);
}
