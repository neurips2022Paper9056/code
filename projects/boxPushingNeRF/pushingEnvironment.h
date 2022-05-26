#ifndef PUSHINGENVIRONMENT_H
#define PUSHINGENVIRONMENT_H

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
#include "cameraObs.h"



struct PushingEnvironment {
  rai::Configuration* K = nullptr;
  rai::Frame* tableFrame = nullptr;
  rai::Frame* pusherFrame = nullptr;
  FrameL allVisObjectFrames;
  FrameL objectFrames;

  CameraObs* camObs = nullptr;
  rai::Simulation* S = nullptr;

  double xLimLo, xLimHi, yLimLo, yLimHi;
  double xSamplingLimLo, xSamplingLimHi, ySamplingLimLo, ySamplingLimHi;

  PushingEnvironment();
  ~PushingEnvironment();

  uint initEnvironment(uint seed, int pos_seed);

  std::tuple<torch::Tensor, torch::Tensor, torch::Tensor, torch::Tensor, torch::Tensor> getObservation();
  torch::Tensor getState();
  torch::Tensor getHighDimState();
  torch::Tensor getKeypoints();

  void watch();


  void addPusher();
  void addObject(const char* name, uint colorType, double dx, double dy);
  void setObjectPose(const char* frameName, double x, double y);
  void setObjectPose(const char* frameName, double x, double y, double phi);

  void initSimulation();
  void simulateStep(std::vector<double> pusherDelta);

  bool sampleObjectPose(rai::Frame* frame, bool twoD);
  bool checkIfAllWithinBounds();
  bool checkIfObjectsWithinBounds();
  bool checkIfPusherWithinBounds();
  bool checkIfWithinBounds(rai::Frame* frame);
  void computeBoundsFromTableFrame();
  void initDefaultCameraObs();

  static void setObjectPose(rai::Frame* object, double x, double y);
  static void setObjectPose(rai::Frame* object, double x, double y, double phi);
  static rai::Frame* addPusherHelper(rai::Configuration& K);
  static rai::Frame* addObjectHelper(rai::Configuration& K, std::string name, uint colorType, double dx, double dy);
  inline static const double ZHEIGHTCOORDINATE = 0.57501;
};

#endif // PUSHINGENVIRONMENT_H
