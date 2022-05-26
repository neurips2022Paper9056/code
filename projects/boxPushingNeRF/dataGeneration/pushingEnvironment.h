#ifndef PUSHINGENVIRONMENT_H
#define PUSHINGENVIRONMENT_H

#include <Core/util.h>
#include <Kin/kin.h>
#include <Kin/frame.h>
#include <Kin/cameraview.h>
#include <Kin/F_collisions.h>
#include <Kin/simulation.h>
#include "dataGen.h"




struct PushingEnvironment {
  rai::Configuration* K;
  rai::Frame* tableFrame;
  rai::Frame* pusherFrame;
  FrameL allVisObjectFrames;
  FrameL objectFrames;

  double xLimLo, xLimHi, yLimLo, yLimHi;
  double xSamplingLimLo, xSamplingLimHi, ySamplingLimLo, ySamplingLimHi;

  PushingEnvironment(rai::Configuration* K);

  bool sampleObjectPose(rai::Frame* frame, bool twoD);
  bool checkIfAllWithinBounds();
  bool checkIfObjectsWithinBounds();
  bool checkIfWithinBounds(rai::Frame* frame);
  void computeBoundsFromTableFrame();

  static void setObjectPose(rai::Frame* object, double x, double y);
  static void setObjectPose(rai::Frame* object, double x, double y, double phi);
  static rai::Frame* addPusherHelper(rai::Configuration& K);
  static rai::Frame* addObjectHelper(rai::Configuration& K, std::string name);
};

#endif // PUSHINGENVIRONMENT_H
