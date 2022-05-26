#include "F_ImS.h"

#include "F_qFeatures.h"
#include "F_pose.h"
#include "forceExchange.h"
#include <torch/csrc/api/include/torch/autograd.h>
#include "F_collisions.h"

namespace {

arr convert_Tensor2arrRef(const torch::Tensor& tensor) {
  torch::Tensor tmp = tensor.cpu().to(torch::kDouble);
  return arr(tmp.data_ptr<double>(), sizeof(torch::kDouble)*tmp.numel(), false);
}

torch::Tensor convert_arr2Tensor(const arr& array) {
  return torch::from_blob(array.p, {array.d0}, torch::kDouble).to(torch::kFloat).to(ImS::defaultTorchDevice);
}

}

ImS::F_ImFunctional::F_ImFunctional(std::string networkModuleFileName, rai::Frame* frame) {
  FModel = torch::jit::load(networkModuleFileName, ImS::defaultTorchDevice);
  /*if(frame) {
    hasFixed = true;

  }*/
}

ImS::F_ImFunctional::F_ImFunctional(torch::jit::Module &network) {
  FModel = network;
}

void ImS::F_ImFunctional::phi2(arr& y, arr& J, const FrameL& F) {
  rai::Frame* boundingBoxFrame = F.elem(-1);
  std::vector<arr> JqList;
  std::vector<torch::Tensor> inputList;
  std::vector<torch::IValue> phiGridList;
  for(uint i = 0; i < F.N-1; i++) {
    rai::Frame* frame = F.elem(i);
    arr q, Jq;
    //F_qItself(uintA{frame->ID}).__phi(q, (!!J) ? Jq : NoArr, frame->C);
    F_Pose().__phi2(q, (!!J) ? Jq : NoArr, {frame});
    torch::Tensor t_q = convert_arr2Tensor(q);
    if(!!J && frame->joint->type != rai::JT_rigid) {
      t_q.requires_grad_(true);
      JqList.push_back(Jq);
    }
    torch::Tensor X = ImS::getTransformedXGrid(t_q, boundingBoxFrame->ImBoundingBox->getBoundingBox());
    uint d = boundingBoxFrame->ImBoundingBox->d;
    uint h = boundingBoxFrame->ImBoundingBox->h;
    uint w = boundingBoxFrame->ImBoundingBox->w;
    phiGridList.push_back(frame->ImSModule->forward({X}).reshape({1,d,h,w}));
    if(frame->joint->type != rai::JT_rigid) inputList.push_back(t_q);
  }
  auto output = FModel.forward(phiGridList).toTensor();
  y.resize(1);
  y(0) = output.item().to<double>();

  if(!!J) {
    auto gradient = torch::autograd::grad({output}, inputList);
    F(0,0)->C.jacobian_zero(J, 1);
    for(uint i = 0; i < gradient.size(); i++) {
      arr JO = convert_Tensor2arrRef(gradient.at(i));
      J += JO.reshape(1, JO.d0)*JqList.at(i);
    }
  }
}



///////////////////////////////////////////////////////////////////////////////



ImS::F_ImFunctional_1Order::F_ImFunctional_1Order(std::string networkModuleFileName) {
  FModel = torch::jit::load(networkModuleFileName, ImS::defaultTorchDevice);
}

void ImS::F_ImFunctional_1Order::phi2(arr& y, arr& J, const FrameL& F) {
  rai::Frame* boundingBoxFrame = F.elem(-1);
  uint d = boundingBoxFrame->ImBoundingBox->d;
  uint h = boundingBoxFrame->ImBoundingBox->h;
  uint w = boundingBoxFrame->ImBoundingBox->w;
  std::vector<arr> JqList;
  std::vector<torch::Tensor> inputList;
  std::vector<torch::IValue> phiGridList;
  for(auto index : { std::pair<uint, uint>{0,0}, std::pair<uint, uint>{1,0}, std::pair<uint, uint>{0,1}, std::pair<uint, uint>{1,1} }) {
    rai::Frame* frame = F(index.first, index.second);
    arr q, Jq;
    F_Pose().__phi2(q, (!!J) ? Jq : NoArr, {frame});
    torch::Tensor t_q = convert_arr2Tensor(q);
    if(!!J) {
      t_q.requires_grad_(true);
      JqList.push_back(Jq);
    }
    torch::Tensor X = ImS::getTransformedXGrid(t_q, boundingBoxFrame->ImBoundingBox->getBoundingBox());
    phiGridList.push_back(frame->ImSModule->forward({X}).reshape({1,d,h,w}));
    inputList.push_back(t_q);
  }
  torch::Tensor phiPred = FModel.forward({phiGridList[0], phiGridList[2], phiGridList[3],
                                          boundingBoxFrame->ImBoundingBox->getBoundingBox().index({torch::indexing::Slice(), torch::indexing::Slice(0,2)})
                                         }).toTensor().view({1,d,h,w});
  torch::Tensor output = (phiPred - phiGridList[1].toTensor()).square().mean();
  y.resize(1);
  y(0) = output.item().to<double>();

  if(!!J) {
    auto gradient = torch::autograd::grad({output}, inputList);
    F(0,0)->C.jacobian_zero(J, 1);
    for(uint i = 0; i < gradient.size(); i++) {
      arr JO = convert_Tensor2arrRef(gradient.at(i));
      J += JO.reshape(1, JO.d0)*JqList.at(i);
    }
  }
}


///////////////////////////////////////////////////////////////////////////////



ImS::F_ImPairCollision::F_ImPairCollision(float k, rai::Frame* frame) {
  this->k = k;
}

void ImS::F_ImPairCollision::phi2(arr& y, arr& J, const FrameL& F) {
  rai::Frame* boundingBoxFrame = F.elem(-1);
  std::vector<arr> JqList;
  std::vector<torch::Tensor> inputList;
  std::vector<torch::Tensor> phiGridList;
  for(uint i = 0; i < 2; i++) {
    rai::Frame* frame = F.elem(i);
    arr q, Jq;
    F_Pose().__phi2(q, (!!J) ? Jq : NoArr, {frame});
    torch::Tensor t_q = convert_arr2Tensor(q);
    if(!!J /*&& frame->joint->type != rai::JT_rigid*/) {
      t_q.requires_grad_(true);
      JqList.push_back(Jq);
    }
    torch::Tensor X = ImS::getTransformedXGrid(t_q, boundingBoxFrame->ImBoundingBox->getBoundingBox());
    phiGridList.push_back(frame->ImSModule->forward({X}));
    if(true || frame->joint->type != rai::JT_rigid) inputList.push_back(t_q);
  }
  auto output = (torch::sigmoid(-k*phiGridList[0])*torch::sigmoid(-k*phiGridList[1])).mean();
  y.resize(1);
  y(0) = output.item().to<double>();

  if(!!J) {
    auto gradient = torch::autograd::grad({output}, inputList);
    F(0,0)->C.jacobian_zero(J, 1);
    for(uint i = 0; i < gradient.size(); i++) {
      arr JO = convert_Tensor2arrRef(gradient.at(i));
      J += JO.reshape(1, JO.d0)*JqList.at(i);
    }
  }
}



///////////////////////////////////////////////////////////////////////////////


ImS::F_ImSDFGoal::F_ImSDFGoal(rai::Frame *frame, rai::Frame *boundingBoxFrame) {
  phiGoalGrid = frame->ImSModule->forward({boundingBoxFrame->ImBoundingBox->getBoundingBox()});
}

ImS::F_ImSDFGoal::F_ImSDFGoal(rai::Frame* frame, arr sGoal, rai::Frame *boundingBoxFrame) {
  cout << sGoal << endl;
  torch::Tensor t_q = convert_arr2Tensor(sGoal);
  torch::Tensor X = ImS::getTransformedXGrid(t_q, boundingBoxFrame->ImBoundingBox->getBoundingBox());
  phiGoalGrid = frame->ImSModule->forward({X});
  //torch::Tensor tmp = phiGoalGrid.cpu().to(torch::kDouble);
  //FILE("goalSDF.dat") << arr(tmp.data_ptr<double>(), sizeof(torch::kDouble)*tmp.numel(), false);
}

void ImS::F_ImSDFGoal::phi2(arr &y, arr &J, const FrameL &F) {
  rai::Frame* boundingBoxFrame = F.elem(-1);

  rai::Frame* frame = F.elem(0);
  arr q, Jq;
  F_Pose().__phi2(q, (!!J) ? Jq : NoArr, {frame});
  //F_qItself(uintA{frame->ID}).__phi(q, (!!J) ? Jq : NoArr, frame->C);
  torch::Tensor t_q = convert_arr2Tensor(q);
  if(!!J) {
    t_q.requires_grad_(true);
  }
  torch::Tensor X = ImS::getTransformedXGrid(t_q, boundingBoxFrame->ImBoundingBox->getBoundingBox());
  torch::Tensor phiGrid = frame->ImSModule->forward({X});
  torch::Tensor overlappIntegral = (phiGrid-phiGoalGrid).square().mean();
  y.resize(1);
  y(0) = overlappIntegral.item().to<double>();
  if(!!J) {
    auto gradient = torch::autograd::grad({overlappIntegral}, {t_q});
    arr JO = convert_Tensor2arrRef(gradient.at(0));
    J = JO.reshape(1, JO.d0)*Jq;
  }
}



///////////////////////////////////////////////////////////////////////////////



ImS::F_ImGoalOverlap::F_ImGoalOverlap(float k) {
  this->k = k;
}

void ImS::F_ImGoalOverlap::phi2(arr& y, arr& J, const FrameL& F) {
  rai::Frame* boundingBoxFrame = F.elem(-1);
  std::vector<arr> JqList;
  std::vector<torch::Tensor> inputList;
  std::vector<torch::Tensor> phiGridList;
  for(uint i = 0; i < 2; i++) {
    rai::Frame* frame = F.elem(i);
    arr q, Jq;
    F_Pose().__phi2(q, (!!J) ? Jq : NoArr, {frame});
    torch::Tensor t_q = convert_arr2Tensor(q);
    if(!!J && frame->joint->type != rai::JT_rigid) {
      t_q.requires_grad_(true);
      JqList.push_back(Jq);
    }
    torch::Tensor X = ImS::getTransformedXGrid(t_q, boundingBoxFrame->ImBoundingBox->getBoundingBox());
    phiGridList.push_back(frame->ImSModule->forward({X}));
    if(frame->joint->type != rai::JT_rigid) inputList.push_back(t_q);
  }
  /*arr tmp = convert_Tensor2arrRef(phiGridList[0]);
  //FILE("phi1.dat") << tmp;
  tmp = convert_Tensor2arrRef(phiGridList[1]);
  //FILE("phi2.dat") << tmp;*/
  auto output = (torch::sigmoid(-k*phiGridList[0])*torch::sigmoid(k*phiGridList[1])).mean();
  y.resize(1);
  y(0) = output.item().to<double>();
  if(!!J) {
    auto gradient = torch::autograd::grad({output}, inputList);
    F(0,0)->C.jacobian_zero(J, 1);
    for(uint i = 0; i < gradient.size(); i++) {
      arr JO = convert_Tensor2arrRef(gradient.at(i));
      J += JO.reshape(1, JO.d0)*JqList.at(i);
    }
  }
}


///////////////////////////////////////////////////////////////////////////////

