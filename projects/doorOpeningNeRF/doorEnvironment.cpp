#include "doorEnvironment.h"

#include <KOMO/komo.h>
#include <Kin/F_pose.h>
#include <Kin/F_qFeatures.h>




DoorEnvironment::DoorEnvironment() {}

DoorEnvironment::~DoorEnvironment() {
  if(K) delete K;
  if(camObs) delete camObs;
}

void DoorEnvironment::initEnvironment(uint seed, bool wallOffset) {
  if(K) delete K;
  K = new rai::Configuration;
  K->addFile("world.g");

  rnd.seed(seed);
  double doorHandleHeight = rnd.uni()*0.05 + 0.03;
  double doorHandleX = rnd.uni()*0.07 + 0.1;
  double doorHandleZ = rnd.uni()*0.16 - 0.08;
  generateScene(doorHandleHeight, doorHandleX, doorHandleZ, wallOffset);

  stateBounds = {{-0.35, 0.0}, {-0.25, 0.25}, {-0.1, 0.04}, {-0.08 + 0.2,  -0.08 + 0.16 + 0.2}};

  doorFrame = K->getFrame("door");
  endeffFrame = K->getFrame("endeff");

  if(camObs) delete camObs;
  camObs = new CameraObs(*K, 150, 200);
  camObs->addObjectToMaskList(doorFrame->name);
  camObs->addObjectToMaskList(K->getFrame("doorHandle")->name);
  camObs->addObjectToMaskList(endeffFrame->name);
  initDefaultCameraObs();
}

void DoorEnvironment::watch() {
  K->watch();
}

namespace {

void solveProjection(rai::Configuration& K, rai::Frame* fixedFrame) {
  KOMO komo;
  komo.setModel(K, true);
  komo.setTiming(1, 1, 0.01, 1);
  komo.add_qControlObjective({}, 1);
  komo.add_collision(true, 0.0, 10.0);
  komo.addObjective({}, make_shared<F_qLimits2>(), {"ALL"}, OT_ineq, {100.0});
  komo.addObjective({}, FS_pose, {fixedFrame->name}, OT_eq, {10.0}, NoArr, 1);
  komo.verbose = 0;
  komo.optimize(0.0);
  arr q = komo.getConfiguration_q(0);
  K.setJointState(q);
}

}

void DoorEnvironment::simulateStep(std::vector<double> endeffDelta) {
  for(uint t = 0; t < 5; t++) {
    uint i = endeffFrame->joint->qIndex;
    uint n = endeffFrame->joint->qDim();
    arr q = K->getJointState();
    for(uint j = 0; j < n; j++) q(i+j) += endeffDelta[j]/5.0;
    K->setJointState(q);
    solveProjection(*K, endeffFrame);
    solveProjection(*K, doorFrame);
  }
}

torch::Tensor DoorEnvironment::getState() {
  arr q = K->getJointState();
  return torch::from_blob(q.p, {q.N}, torch::kDouble).to(torch::kFloat).clone();
}

torch::Tensor DoorEnvironment::getHighDimState() {
  uint L = 10;
  arr q = K->getJointState();
  arr state;
//  state.reserveMEM(L*q.N*2);
  for(uint i = 0; i < q.N; i++) {
    for(uint l = 0; l < L; l++) {
      double x = static_cast<double>(powl(2, l))*RAI_PI*q(i);
      state.append(sin(x));
      state.append(cos(x));
    }
  }
  return torch::from_blob(state.p, {state.N}, torch::kDouble).to(torch::kFloat).clone();
}

torch::Tensor DoorEnvironment::getKeypoints() {
  arr KP;
  for(auto f : {"door", "doorHandle", "doorHandle_m1", "doorHandle_m2", "endeff"}) {
    KP.append(K->getFrame(f)->getPosition());
  }
  return torch::from_blob(KP.p, {KP.N/3,3}, torch::kDouble).to(torch::kFloat).clone();
}

void DoorEnvironment::setState(const torch::Tensor& q) {
  arr qA(q.cpu().to(torch::kDouble).data_ptr<double>(), sizeof(torch::kDouble)*q.numel(), false);
  setState(qA);
}

void DoorEnvironment::setState(const arr& q) {
  K->setJointState(q);
}


std::tuple<torch::Tensor, torch::Tensor, torch::Tensor, torch::Tensor, torch::Tensor> DoorEnvironment::getObservation() {
  torch::Tensor I, M, K, KBounds;
  camObs->getCameraObservations(I, M, K, KBounds);
  using namespace torch::indexing;
  torch::Tensor KT = K.unsqueeze(0).transpose(2,3);
  torch::Tensor E = torch::eye(4).reshape({1,1,4,4}).repeat({KT.size(0),KT.size(1),1,1});
  E.index_put_({Slice(), Slice(), Slice(None,4), Slice(None,3)}, KT);
  torch::Tensor KInvT = torch::linalg_inv(E);
  KInvT = KInvT.index({Slice(), Slice(), Slice(None,4), Slice(None,3)});
  return {I.unsqueeze(0).to(torch::kFloat)/255.0, M.unsqueeze(0).to(torch::kFloat), KT, KInvT, KBounds.unsqueeze(0)};
}

bool DoorEnvironment::checkWithinBounds() {
  arr q = K->getJointState();
  for(uint i = 0; i < q.N; i++) {
    const double& qI = q(i);
    if(qI < stateBounds[i][0] || qI > stateBounds[i][1]) return false;
  }
  return true;
}

std::vector<std::vector<double> > DoorEnvironment::getStateBounds() {
  return stateBounds;
}

void DoorEnvironment::initDefaultCameraObs() {
  addDefaultCamera(90.0, -90.0, 0.3, 0.8);
  addDefaultCamera(70.0, -130.0, 0.2, 0.9);
  addDefaultCamera(50.0, -60.0, 0.2, 0.95);
  addDefaultCamera(10.0, -90.0, 0.35, 0.9, 0.6);
}

void DoorEnvironment::addAlternativeCameraObs() {
  addDefaultCamera(85.0, -80.0, 0.3, 0.8);
  addDefaultCamera(65.0, -135.0, 0.2, 0.9);
  addDefaultCamera(55.0, -55.0, 0.2, 0.95);
  addDefaultCamera(14.0, -95.0, 0.35, 0.9, 0.6);
}

void DoorEnvironment::addDefaultCamera(double theta, double phi, double near, double far, double r) {
  rai::Transformation T;
  T.setZero();
  T.rot.setRpy(0.0, theta*RAI_PI/180.0, phi*RAI_PI/180.0);
  T.addRelativeRotationDeg(90.0, 0.0, 0.0, 1.0);
  arr back = conv_vec2arr(T.rot*rai::Vector(0.0, 0.0, r));
  arr center = {-0.1, 0.0, 0.2};
  T.pos = center + back;
  camObs->addCamera(T, 0.8, {near, far});
}

void DoorEnvironment::generateScene(double doorHandleHeight, double doorHandleX, double doorHandleZ, bool wallOffset) {
  double doorHeight = 0.4;
  double doorWidth = 0.4;

  K->addFrame("doorCenter", "world")->set_Q()->addRelativeTranslation({0.0, 0.0, doorHeight/2.0});
  rai::Frame& door = K->addFrame("door", "doorCenter")
      ->setColor({0.9, 0.8, 0.0})
      .setContact(1)
      .setMass(1.0)
      .setJoint(rai::JointType::JT_transX)
      .setShape(rai::ShapeType::ST_ssBox, {doorWidth, 0.03, doorHeight, 0.005});
  door.joint->limits = {-0.35, 0.0};

  rai::Frame& wallRight = K->addFrame("wallRight", "doorCenter")
      ->setColor({0.9, 0.8, 0.0})
      .setContact(1)
      .setShape(rai::ShapeType::ST_ssBox, {doorWidth, 0.03, doorHeight, 0.005});
  wallRight.set_Q()->addRelativeTranslation({-doorWidth*0.95, 0.03, 0.0});

  rai::Frame& wallLeft = K->addFrame("wallLeft", "doorCenter")
      ->setColor({0.9, 0.8, 0.0})
      .setContact(1)
      .setShape(rai::ShapeType::ST_ssBox, {doorWidth, 0.03, doorHeight, 0.005});
  if(wallOffset) {
    wallLeft.set_Q()->addRelativeTranslation({doorWidth-0.02, -0.03, 0.0});
  } else {
    wallLeft.set_Q()->addRelativeTranslation({doorWidth, 0.0, 0.0});
  }

  rai::Frame& doorHandle = K->addFrame("doorHandle", "door")
      ->setColor({0.0, 0.3, 0.9})
      .setContact(1)
      .setMass(0.04)
      .setShape(rai::ShapeType::ST_ssBox, {0.02, 0.04, doorHandleHeight, 0.005});
  doorHandle.set_Q()->addRelativeTranslation({doorHandleX, -0.03, doorHandleZ});

  K->addFrame("endeff", "world")
      ->setJoint(rai::JointType::JT_trans3);
  rai::Frame& endeffShape = K->addFrame("endeffShape", "endeff")
      ->setColor({1.0, 0.0, 0.0})
      .setContact(1)
      .setShape(rai::ShapeType::ST_capsule, {0.04, 0.012});
  endeffShape.set_Q()->addRelativeRotationDeg(90.0, 1.0, 0.0, 0.0);

  K->addFrame("doorHandle_m1", "doorHandle")->setShape(rai::ShapeType::ST_marker, {0.1})
      .set_Q()->addRelativeTranslation(0.01, -0.02, doorHandleHeight/2.0);
  K->addFrame("doorHandle_m2", "doorHandle")->setShape(rai::ShapeType::ST_marker, {0.1})
      .set_Q()->addRelativeTranslation(-0.01, -0.02, -doorHandleHeight/2.0);
}
