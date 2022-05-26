#ifndef MUGENVIRONMENT_H
#define MUGENVIRONMENT_H

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




struct MugEnvironment {
  rai::Configuration* K = nullptr;

  CameraObs* camObs = nullptr;
  
  rai::Configuration* KVideo = nullptr;
  CameraObs* camObsVideo = nullptr;

  rai::Frame* boundingBoxFrame;
  rai::Frame* mugFrame;
  rai::Frame* hookFrame;

  bool phisAreUpdated = false;
  torch::Tensor phiMug;
  torch::Tensor phiHook;

  MugEnvironment();
  ~MugEnvironment();

  void initEnvironment(uint seed = 0);
  void watch();
  
  void initCameraForVideo(bool showKeypoints=false);
  void initCameraForRoundView();

  torch::Tensor getState();
  torch::Tensor getHighDimState();
  torch::Tensor getKeypoints();
  void setState(const torch::Tensor& q);
  void setState(const arr& q);
  std::tuple<bool, bool> checkStateInSim();
  torch::Tensor getSDFs();
  std::tuple<float, float, float, float, float, float> getBoundingBoxLimits();
  std::tuple<torch::Tensor, torch::Tensor, torch::Tensor, torch::Tensor, torch::Tensor> getObservation();
  std::tuple<torch::Tensor, torch::Tensor, torch::Tensor, torch::Tensor, torch::Tensor> getObservationVideo();

  void ensurePhisAreUpdated();
  void initDefaultCameraObs();
  void addAlternativeCameraObs();
  void addDefaultCamera(double theta, double phi, CameraObs* camObsOverwrite=nullptr);
};

#endif // MUGENVIRONMENT_H
