#ifndef IMS_MODULE_H
#define IMS_MODULE_H

#undef RAI_LOG
#undef RAI_CHECK
#undef RAI_CHECK_EQ
#undef RAI_CHECK_GE
#undef RAI_CHECK_LE
#include <torch/script.h>

#include "frame.h"

namespace ImS {

extern torch::Device defaultTorchDevice;

torch::Tensor convertQuatToRotMatrix(torch::Tensor quat);
torch::Tensor getTransformedXGrid(const torch::Tensor& q, const torch::Tensor& gridFlat);

torch::Tensor getSDF(rai::Frame* frame, rai::Frame* boundingBox, bool reshape = true);


struct BoundingBox;

struct Module {
  virtual torch::Tensor forward(const std::vector<torch::Tensor>& inputs) = 0;
  virtual void to(torch::Device device) = 0;
};

struct BoundingBox {
  uint dim;
  float xPos, yPos, zPos;
  float xLo, xHi, yLo, yHi, zLo, zHi;
  uint w,h,d;
  torch::Tensor gridFlat;
  torch::Tensor centerPos;
  void set(float xPos, float yPos, float zPos,
           float xLo, float xHi,
           float yLo, float yHi,
           float zLo, float zHi,
           uint w, uint h, uint d);
  void set(float xPos, float yPos,
           float xLo, float xHi,
           float yLo, float yHi,
           uint w, uint h);
  void extractFromFrame(rai::Frame* frame);

  inline torch::Tensor& getBoundingBox() { return gridFlat; }
  void to(torch::Device device);
};


struct ImAnalyticSDF : ImS::Module {
  std::vector<torch::Tensor> tList;
  std::vector<torch::Tensor> RList;
  std::vector<torch::Tensor> sizeList;
  std::vector<rai::ShapeType> SDFPrimitiveList;

  void extractFromFrame(rai::Frame* frame, bool relative = true);

  static std::shared_ptr<ImAnalyticSDF> applyOnFrame(rai::Frame* frame, bool relative = true);

  torch::Tensor forward(const std::vector<torch::Tensor>& inputs);
  void to(torch::Device device);
};



} // namespace ImS

#endif // IMS_MODULE_H
