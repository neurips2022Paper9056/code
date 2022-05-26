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
#include <KOMO/komo.h>
#include <Kin/simulation.h>
#include <Kin/F_pose.h>
#include <Kin/ImSModule.h>
#include <Kin/F_ImS.h>

#include "mugEnvironment.h"

#include <pybind11/pybind11.h>
#include <torch/csrc/utils/pybind.h>

namespace py = pybind11;

torch::Device ImS::defaultTorchDevice("cuda");

void init_CfgFileParameters() {
  char* argv[2] = {(char*)"rai-pybind", (char*)"-python"};
  int argc = 2;
  rai::initCmdLine(argc, argv);
}

PYBIND11_MODULE(mugEnvironment, m) {
  init_CfgFileParameters();
  py::class_<MugEnvironment>(m, "MugEnvironment")
    .def(py::init<>())
    .def("initEnvironment", &MugEnvironment::initEnvironment)
    .def("initCameraForVideo", &MugEnvironment::initCameraForVideo, "", py::arg("showKeypoints") = false)
    .def("initCameraForRoundView", &MugEnvironment::initCameraForRoundView)
    .def("getSDFs", &MugEnvironment::getSDFs)
    .def("getObservation", &MugEnvironment::getObservation)
    .def("getObservationVideo", &MugEnvironment::getObservationVideo)
    .def("checkStateInSim", &MugEnvironment::checkStateInSim)
    .def("watch", &MugEnvironment::watch)
    .def("getState", &MugEnvironment::getState)
    .def("getHighDimState", &MugEnvironment::getHighDimState)
    .def("getKeypoints", &MugEnvironment::getKeypoints)
    .def("setState", py::overload_cast<const arr&>(&MugEnvironment::setState))
    .def("setState", py::overload_cast<const torch::Tensor&>(&MugEnvironment::setState))
    .def("getBoundingBoxLimits", &MugEnvironment::getBoundingBoxLimits)
    .def("addAlternativeCameraObs", &MugEnvironment::addAlternativeCameraObs); 
}



