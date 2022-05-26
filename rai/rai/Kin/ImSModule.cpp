#include "ImSModule.h"
#include "F_pose.h"

namespace ImS {

torch::Device defaultTorchDevice("cuda");

///////////////////////////////////////////////////////////////////////////////

torch::Tensor convertQuatToRotMatrix(torch::Tensor quat) {
  quat = quat/torch::norm(quat);
  torch::Tensor w = quat.index({0});
  torch::Tensor x = quat.index({1});
  torch::Tensor y = quat.index({2});
  torch::Tensor z = quat.index({3});
  torch::Tensor P1 = 2.0f * x;
  torch::Tensor P2 = 2.0f * y;
  torch::Tensor P3 = 2.0f * z;
  torch::Tensor q11 = x * P1;
  torch::Tensor q22 = y * P2;
  torch::Tensor q33 = z * P3;
  torch::Tensor q12 = x * P2;
  torch::Tensor q13 = x * P3;
  torch::Tensor q23 = y * P3;
  torch::Tensor q01 = w * P1;
  torch::Tensor q02 = w * P2;
  torch::Tensor q03 = w * P3;
  torch::Tensor R = torch::stack({
                                   1.0f-q22-q33,  q12-q03,       q13+q02,
                                   q12+q03,       1.0f-q11-q33,  q23-q01,
                                   q13-q02,       q23+q01,       1.0f-q11-q22
                                 });
  return R.reshape({3,3});
}

torch::Tensor getTransformedXGrid(const torch::Tensor& q, const torch::Tensor& gridFlat) {
  using namespace torch::indexing;
  torch::Tensor t = q.index({Slice(0,3)});
  torch::Tensor quat = q.index({Slice(3,None)});
  torch::Tensor R = convertQuatToRotMatrix(quat);
  return torch::mm(gridFlat - t, R); // computes R^T(x-t) = R^{-1}(x-t) as (x^T-t^T)R
}

/*
torch::Tensor getTransformedXGrid(const torch::Tensor& q, const torch::Tensor& gridFlat, bool is2D) {
  using namespace torch::indexing;
  if(!is2D) {
    torch::Tensor t = q.index({Slice(0,3)});
    torch::Tensor quat = q.index({Slice(3,None)});
    torch::Tensor R = convertQuatToRotMatrix(quat);
    return torch::mm(gridFlat - t, R); // computes R^T(x-t) = R^{-1}(x-t) as (x^T-t^T)R
  } else {
    //torch::Tensor t = q.index({Slice(0,2)});
    torch::Tensor phi = q.index({Slice(2)});
    torch::Tensor c = torch::cos(phi);
    torch::Tensor s = torch::sin(phi);
    torch::Tensor t = torch::stack({q.index({0}), q.index({1}), torch::tensor(0.626).to(phi.device())});
    torch::Tensor R = torch::stack({
                                    c, -s, torch::zeros_like(c),
                                    s,  c, torch::zeros_like(c),
                                    torch::zeros_like(c), torch::zeros_like(c), torch::ones_like(c)
                                   });
    R = R.reshape({3,3});
    return torch::mm(gridFlat - t, R);
  }
}*/


torch::Tensor getSDF(rai::Frame* frame, rai::Frame* boundingBox, bool reshape) {
  arr q;
  F_Pose().__phi2(q, NoArr, {frame});
  torch::Tensor t_q = torch::from_blob(q.p, {q.d0}, torch::kDouble).to(torch::kFloat).to(ImS::defaultTorchDevice);
  torch::Tensor X = ImS::getTransformedXGrid(t_q, boundingBox->ImBoundingBox->getBoundingBox());
  torch::Tensor phi = frame->ImSModule->forward({X});
  if(reshape) {
    uint d = boundingBox->ImBoundingBox->d;
    uint h = boundingBox->ImBoundingBox->h;
    uint w = boundingBox->ImBoundingBox->w;
    return phi.reshape({1,d,h,w});
  }
  return phi;
}


///////////////////////////////////////////////////////////////////////////////


void BoundingBox::set(float xPos, float yPos, float zPos,
                      float xLo, float xHi,
                      float yLo, float yHi,
                      float zLo, float zHi,
                      uint w, uint h, uint d)
{
  this->dim = 3;
  this->xPos = xPos;
  this->yPos = yPos;
  this->zPos = zPos;
  this->xLo = xLo;
  this->xHi = xHi;
  this->yLo = yLo;
  this->yHi = yHi;
  this->zLo = zLo;
  this->zHi = zHi;
  this->w = w;
  this->h = h;
  this->d = d;
  centerPos = torch::tensor({xPos, yPos, zPos}, torch::dtype(torch::kFloat).device(ImS::defaultTorchDevice));
  torch::Tensor xRange = torch::linspace(this->xLo, this->xHi, this->w, ImS::defaultTorchDevice);
  torch::Tensor yRange = torch::linspace(this->yLo, this->yHi, this->h, ImS::defaultTorchDevice);
  torch::Tensor zRange = torch::linspace(this->zLo, this->zHi, this->d, ImS::defaultTorchDevice);
  std::vector<torch::Tensor> grid = torch::meshgrid({zRange, yRange, xRange});
  gridFlat = torch::stack({grid[2].reshape(-1), grid[1].reshape(-1), grid[0].reshape(-1)}, 1);
}

void BoundingBox::set(float xPos, float yPos,
                      float xLo, float xHi,
                      float yLo, float yHi,
                      uint w, uint h)
{
  this->dim = 2;
  this->xPos = xPos;
  this->yPos = yPos;
  this->xLo = xLo;
  this->xHi = xHi;
  this->yLo = yLo;
  this->yHi = yHi;
  this->w = w;
  this->h = h;
  centerPos = torch::tensor({xPos, yPos}, torch::dtype(torch::kFloat).device(ImS::defaultTorchDevice));
  torch::Tensor xRange = torch::linspace(this->xLo, this->xHi, this->w, ImS::defaultTorchDevice);
  torch::Tensor yRange = torch::linspace(this->yLo, this->yHi, this->h, ImS::defaultTorchDevice);
  std::vector<torch::Tensor> grid = torch::meshgrid({yRange, xRange});
  gridFlat = torch::stack({grid[1].reshape(-1), grid[0].reshape(-1)}, 1);
}

void BoundingBox::extractFromFrame(rai::Frame* frame) {
  arr size = frame->getSize();
  arr pos = frame->getPosition();
  arr boundingBoxExtend = frame->ats["boundingBoxExtend"]->get<arr>();
  /*if(frame->ats["boundingBoxDimensions"]) {
    arr boundingBoxDimensions = frame->ats["boundingBoxDimensions"]->get<arr>();
    if(boundingBoxExtend.N == 3) {
      set(pos(0), pos(1), pos(2),
          pos(0) - boundingBoxDimensions(0), pos(0) + boundingBoxDimensions(1),
          pos(1) - boundingBoxDimensions(2), pos(1) + boundingBoxDimensions(3),
          pos(2) - boundingBoxDimensions(4), pos(2) + boundingBoxDimensions(5),
          (uint)boundingBoxExtend(0), (uint)boundingBoxExtend(1), (uint)boundingBoxExtend(2));
    } else {
      NIY;
    }
  } else {*/
  if(boundingBoxExtend.N == 3) {
    set(pos(0), pos(1), pos(2),
        pos(0) - size(0)/2.0f, pos(0) + size(0)/2.0f,
        pos(1) - size(1)/2.0f, pos(1) + size(1)/2.0f,
        pos(2) - size(2)/2.0f, pos(2) + size(2)/2.0f,
        (uint)boundingBoxExtend(0), (uint)boundingBoxExtend(1), (uint)boundingBoxExtend(2));
  } else if(boundingBoxExtend.N == 2) {
    set(pos(0), pos(1),
        pos(0) - size(0)/2.0f, pos(0) + size(0)/2.0f,
        pos(1) - size(1)/2.0f, pos(1) + size(1)/2.0f,
        (uint)boundingBoxExtend(0), (uint)boundingBoxExtend(1));
  } else {
    HALT("wrong dim of bounding box extend");
  }
}

void BoundingBox::to(torch::Device device) {
  gridFlat.to(device);
  centerPos.to(device);
}



///////////////////////////////////////////////////////////////////////////////



void ImAnalyticSDF::extractFromFrame(rai::Frame *frame, bool relative) {
  FrameL allFrames = {frame};
  frame->getRigidSubFrames(allFrames);

  for(rai::Frame* f : allFrames) {
    auto& shapeType = f->getShape().type();
    if(shapeType == rai::ShapeType::ST_ssBox ||
       shapeType == rai::ShapeType::ST_box ||
       shapeType == rai::ShapeType::ST_cylinder ||
       shapeType == rai::ShapeType::ST_capsule ||
       shapeType == rai::ShapeType::ST_sphere)
    {
      SDFPrimitiveList.push_back(shapeType);
      rai::Transformation trans_f = f->ensure_X();
      if(relative) {
        trans_f = trans_f / frame->ensure_X();
      }
      arr pos = trans_f.pos.getArr();
      arr quat = trans_f.rot.getArr4d();
      arr size = f->getSize();
      tList.push_back(torch::from_blob(pos.p, {3}, torch::kDouble).to(torch::kFloat).to(ImS::defaultTorchDevice));
      RList.push_back(convertQuatToRotMatrix(torch::from_blob(quat.p, {4}, torch::kDouble).to(torch::kFloat).to(ImS::defaultTorchDevice)));
      sizeList.push_back(torch::from_blob(size.p, {size.d0}, torch::kDouble).clone().to(torch::kFloat).to(ImS::defaultTorchDevice));
    }
  }
}

std::shared_ptr<ImAnalyticSDF> ImAnalyticSDF::applyOnFrame(rai::Frame* frame, bool relative) {
  std::shared_ptr<ImAnalyticSDF> SDF = std::make_shared<ImAnalyticSDF>();
  SDF->extractFromFrame(frame, relative);
  frame->ImSModule = SDF;
  return SDF;
}

torch::Tensor ImAnalyticSDF::forward(const std::vector<torch::Tensor>& inputs) {
  using namespace torch::indexing;
  std::vector<torch::Tensor> SDFs;
  for(uint i = 0; i < SDFPrimitiveList.size(); i++) {
    torch::Tensor x = torch::mm(inputs[0] - tList[i], RList[i]);  // computes R^T(x-t) = R^{-1}(x-t) as (x^T-t^T)R
    rai::ShapeType& shapeType = SDFPrimitiveList[i];
    torch::Tensor size = sizeList[i].clone();
    torch::Tensor phi;
    if(shapeType == rai::ShapeType::ST_ssBox) {
      //size = size/2.0f;
      size.index_put_({Slice(0,3)}, size.index({Slice(0,3)})/2.0f - size.index({3}));
      torch::Tensor q = torch::abs(x) - size.index({Slice(0,3)});
      phi = torch::norm(torch::maximum(q, torch::zeros_like(q)), 2, 1)
          + torch::minimum(torch::maximum(q.index({Slice(), 0}), torch::maximum(q.index({Slice(), 1}), q.index({Slice(), 2}))), torch::zeros_like(q.index({Slice(), 0}))) - size.index({3});
    } else if(shapeType == rai::ShapeType::ST_box) {
      size = size/2.0f;
      torch::Tensor q = torch::abs(x) - size;
      phi = torch::norm(torch::maximum(q, torch::zeros_like(q)), 2, 1)
          + torch::minimum(torch::maximum(q.index({Slice(), 0}), torch::maximum(q.index({Slice(), 1}), q.index({Slice(), 2}))), torch::zeros_like(q.index({Slice(), 0})));
    } else if(shapeType == rai::ShapeType::ST_cylinder) {
      torch::Tensor d1 = torch::abs(torch::norm(x.index({Slice(), Slice(0,2)}), 2, 1)) - size.index({1});
      torch::Tensor d2 = torch::abs(x.index({Slice(), 2})) - size.index({0})/2.0;
      torch::Tensor d = torch::stack({d1, d2}, 1);
      phi = torch::minimum(torch::maximum(d1, d2), torch::zeros_like(d1))
          + torch::norm(torch::maximum(d, torch::zeros_like(d)), 2, 1);
    } else if(shapeType == rai::ShapeType::ST_capsule) {
      torch::Tensor p = x.clone();
      p.index_put_({Slice(), 2}, x.index({Slice(), 2}) - torch::clamp(x.index({Slice(), 2}), (-size.index({0})/2.0).item(), (size.index({0})/2.0).item()));
      phi = torch::norm(p, 2, 1) - size.index({1});
    } else if(shapeType == rai::ShapeType::ST_sphere) {
      phi = torch::norm(x, 2, 1) - size.index({0});
    } else {
      NIY;
    }
    SDFs.push_back(phi);
  }
  return std::get<0>(torch::min(torch::stack(SDFs), 0));
}

void ImAnalyticSDF::to(torch::Device device) {
  for(torch::Tensor& t : tList) t.to(device);
  for(torch::Tensor& t : RList) t.to(device);
  for(torch::Tensor& t : sizeList) t.to(device);
}


///////////////////////////////////////////////////////////////////////////////



} // namespace ImS



