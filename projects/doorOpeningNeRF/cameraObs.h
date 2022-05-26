#ifndef CAMERAOBS_H
#define CAMERAOBS_H

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

#include <Kin/kin.h>
#include <Kin/frame.h>
#include <Kin/cameraview.h>

struct CameraObs {
  rai::Configuration& K;
  uint h;
  uint w;
  rai::CameraView cameraView;
  StringA objectNames;

  CameraObs(rai::Configuration& K, uint h, uint w);
  ~CameraObs();

  void addCamera(const rai::Transformation& T, double focalLength, const arr& zRange);
  void addObjectToMaskList(const char* frameName);
  void getCameraObservations(torch::Tensor& I, torch::Tensor& M, torch::Tensor& KMat, torch::Tensor& KBounds);

  uint getNumberOfViews();
  uint getNumberOfMasks();
};

#endif // CAMERAOBS_H
