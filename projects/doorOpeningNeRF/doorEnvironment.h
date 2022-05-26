#ifndef DOORENVIRONMENT_H
#define DOORENVIRONMENT_H

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

#include "cameraObs.h"

#include <Kin/kin.h>
#include <Kin/viewer.h>
#include <tuple>


struct DoorEnvironment {
  rai::Configuration* K = nullptr;

  CameraObs* camObs = nullptr;

  rai::Frame* doorFrame;
  rai::Frame* endeffFrame;

  std::vector<std::vector<double>> stateBounds;

  DoorEnvironment();
  ~DoorEnvironment();

  void initEnvironment(uint seed = 0, bool wallOffset = false);
  void watch();

  void simulateStep(std::vector<double> endeffDelta);

  torch::Tensor getState();
  torch::Tensor getHighDimState();
  torch::Tensor getKeypoints();
  void setState(const torch::Tensor& q);
  void setState(const arr& q);
  std::tuple<torch::Tensor, torch::Tensor, torch::Tensor, torch::Tensor, torch::Tensor> getObservation();
  bool checkWithinBounds();
  std::vector<std::vector<double>> getStateBounds();

  void initDefaultCameraObs();
  void addAlternativeCameraObs();
  void addDefaultCamera(double theta, double phi, double near, double far, double r = 0.5);

  void generateScene(double doorHandleHeight, double doorHandleX, double doorHandleZ, bool wallOffset);
};

#endif // DOORENVIRONMENT_H
