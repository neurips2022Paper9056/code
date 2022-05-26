#ifndef F_IMS_H
#define F_IMS_H

#undef RAI_LOG
#undef RAI_CHECK
#undef RAI_CHECK_EQ
#undef RAI_CHECK_GE
#undef RAI_CHECK_LE
#include <torch/script.h>

#include "ImSModule.h"
#include "feature.h"

namespace ImS {

struct F_ImFunctional : Feature {
  torch::jit::script::Module FModel;
  torch::Tensor phiFixedGrid;
  bool hasFixed = false;

  F_ImFunctional(std::string networkModuleFileName, rai::Frame* frame = nullptr);
  F_ImFunctional(torch::jit::Module& network);

  virtual void phi2(arr& y, arr& J, const FrameL& F);
  virtual uint dim_phi2(const FrameL&) {
    return 1;
  }
};

struct F_ImFunctional_1Order : Feature {
  torch::jit::script::Module FModel;

  F_ImFunctional_1Order(std::string networkModuleFileName);

  virtual void phi2(arr& y, arr& J, const FrameL& F);
  virtual uint dim_phi2(const FrameL&) {
    return 1;
  }
};

struct F_ImPairCollision : Feature {
  float k;
  torch::Tensor phiFixedGrid;
  bool hasFixed = false;

  F_ImPairCollision(float k = 1000.0, rai::Frame* frame = nullptr);

  virtual void phi2(arr& y, arr& J, const FrameL& F);
  virtual uint dim_phi2(const FrameL&) {
    return 1;
  }
};

struct F_ImSDFGoal : Feature {
  torch::Tensor phiGoalGrid;

  F_ImSDFGoal(rai::Frame* frame, rai::Frame* boundingBoxFrame);
  F_ImSDFGoal(rai::Frame* frame, arr sGoal, rai::Frame* boundingBoxFrame);

  virtual void phi2(arr& y, arr& J, const FrameL& F);
  virtual uint dim_phi2(const FrameL&) {
    return 1;
  }
};

struct F_ImGoalOverlap : Feature {
  float k;
  F_ImGoalOverlap(float k = 100.0);
  virtual void phi2(arr& y, arr& J, const FrameL& F);
  virtual uint dim_phi2(const FrameL&) {
    return 1;
  }
};


}

#endif // F_IMS_H
