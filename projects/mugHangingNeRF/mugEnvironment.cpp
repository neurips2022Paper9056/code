#include "mugEnvironment.h"

#include <Kin/simulation.h>
#include <Kin/F_pose.h>
#include <Kin/ImSModule.h>
#include <Kin/F_ImS.h>

namespace {

void addHanger(rai::Configuration& K, uint mode) {
  double hangerVertLength = rnd.uni()*0.1 + 0.45; //0.5
  double hangerRadius = 0.015; //rnd.uni()*0.01 + 0.01; //0.015;
  K.addFrame("hanger", "table")
      ->setColor({0.3, 0.6, 0.3})
      .setContact(1)
      .setShape(rai::ShapeType::ST_capsule, {hangerVertLength, hangerRadius})
      .setJoint(rai::JointType::JT_rigid)
      .set_Q()->pos = {-0.2, -0.2, hangerVertLength/2.0};

  double hangerHorzLength = rnd.uni()*0.1 + 0.1; //0.15;
  double distanceFromTop = rnd.uni()*0.1; //0.1;
  double hangerHorzRot = 0.0;

  double hangerHorzVertPos = hangerVertLength/2.0 - distanceFromTop;
  while(hangerHorzVertPos + hangerVertLength/2.0 > 0.55 || hangerHorzVertPos + hangerVertLength/2.0 < 0.45) {
    distanceFromTop = rnd.uni()*0.1;
    hangerHorzVertPos = hangerVertLength/2.0 - distanceFromTop;
  }

  if(mode == 0) {
    hangerHorzRot = 0.0;
  } else if(mode == 1) {
    hangerHorzRot = rnd.uni()*45.0;
  } else if(mode == 2) {
    hangerHorzRot = 0.0;
  }

  K.addFrame("hanger_h1", "hanger")
      ->setColor({0.3, 0.6, 0.3})
      .setContact(1)
      .setShape(rai::ShapeType::ST_capsule, {hangerHorzLength/cos(hangerHorzRot/180*M_PI), hangerRadius})
      .set_Q()->addRelativeTranslation({hangerHorzLength/2.0, 0.0, hangerHorzVertPos})
              .addRelativeRotationDeg(90.0 - hangerHorzRot, 0.0, 1.0, 0.0);

  if(mode == 2) {
    double hanger_h2VertLength = rnd.uni()*0.07 + 0.03; //0.05;
    K.addFrame("hanger_h2", "hanger")
        ->setColor({0.3, 0.6, 0.3})
        .setContact(1)
        .setShape(rai::ShapeType::ST_capsule, {hanger_h2VertLength, hangerRadius})
        .set_Q()->addRelativeTranslation({hangerHorzLength, 0.0, hanger_h2VertLength/2.0 + hangerHorzVertPos});
  }
  K.addFrame("hanger_m1", "hanger_h1")->setShape(rai::ShapeType::ST_marker, {0.1}).set_Q()->addRelativeTranslation(0.0, 0.0, hangerHorzLength/2.0);
}

void addMug(rai::Configuration& K, uint mode) {
  double mugRadius = rnd.uni()*0.02 + 0.03; //0.05;
  double mugLength = rnd.uni()*0.1 + 0.15; //0.2;
  K.addFrame("mug", "world")
      ->setColor({0.9, 0.8, 0.0})
      .setContact(1)
      .setMass(1.0)
      .setJoint(rai::JointType::JT_trans3)
      .setShape(rai::ShapeType::ST_cylinder, {mugLength, mugRadius})
      .set_Q()->addRelativeTranslation(-0.11, -0.13, 0.5);
//      .set_Q()->addRelativeTranslation(-0.11, -0.13, 0.48);
  double mugHangerExtend = 0.02;
  double mugHangerWidth = rnd.uni()*0.025 + 0.035;
  double mugHangerHeight = rnd.uni()*(mugLength - mugHangerExtend*2.0 - 0.035) + 0.035;
  double tmp = mugLength/2.0 - mugHangerExtend - mugHangerHeight/2.0;
  double mugHangerPos = rnd.uni()*tmp;
  K.addFrame("mug_h1", "mug")
      ->setColor({0.9, 0.8, 0.0})
      .setContact(-1)
      .setMass(0.05)
      .setShape(rai::ShapeType::ST_ssBox, {mugHangerWidth + mugHangerExtend, mugHangerExtend, mugHangerExtend, 0.005})
      .set_Q()->addRelativeRotationDeg(-90.0, 0.0, 0.0, 1.0)
              .addRelativeTranslation(mugRadius + mugHangerWidth/2.0, 0.0, mugHangerHeight/2.0 + mugHangerExtend/2.0 + mugHangerPos);
  K.addFrame("mug_h2", "mug")
      ->setColor({0.9, 0.8, 0.0})
      .setContact(-1)
      .setMass(0.05)
      .setShape(rai::ShapeType::ST_ssBox, {mugHangerExtend, mugHangerExtend, mugHangerHeight+mugHangerExtend*2.0, 0.005})
      .set_Q()->addRelativeRotationDeg(-90.0, 0.0, 0.0, 1.0)
              .addRelativeTranslation(mugRadius + mugHangerWidth + mugHangerExtend/2.0, 0.0, mugHangerPos);
  if(mode > 0) {
    K.addFrame("mug_h3", "mug")
        ->setColor({0.9, 0.8, 0.0})
        .setContact(-1)
        .setMass(0.05)
        .setShape(rai::ShapeType::ST_ssBox, {mugHangerWidth + mugHangerExtend, mugHangerExtend, mugHangerExtend, 0.005})
        .set_Q()->addRelativeRotationDeg(-90.0, 0.0, 0.0, 1.0)
                .addRelativeTranslation(mugRadius + mugHangerWidth/2.0, 0.0, -mugHangerHeight/2.0 - mugHangerExtend/2.0 + mugHangerPos);
  }
  K.addFrame("mug_m1", "mug")->setShape(rai::ShapeType::ST_marker, {0.1}).set_Q()->addRelativeTranslation(0.0, -(mugRadius + mugHangerWidth/2.0), mugHangerPos);
  K.addFrame("mug_m2", "mug")->setShape(rai::ShapeType::ST_marker, {0.1}).set_Q()->addRelativeTranslation(mugRadius, 0.0, mugLength/2.0);
}

}


MugEnvironment::MugEnvironment() {}

MugEnvironment::~MugEnvironment() {
  if(K) delete K;
  if(camObs) delete camObs;
  if(camObsVideo) delete camObsVideo;
}

void MugEnvironment::initEnvironment(uint seed) {
  if(K) delete K;
  K = new rai::Configuration;
  K->addFile("main.g");
  rnd.seed(seed);
  addMug(*K, 1);
  addHanger(*K, 0);
  boundingBoxFrame = K->getFrame("boundingBox");
  mugFrame = K->getFrame("mug");
  hookFrame = K->getFrame("hanger");
  ImS::ImAnalyticSDF::applyOnFrame(mugFrame, true);
  ImS::ImAnalyticSDF::applyOnFrame(hookFrame, true);
  phisAreUpdated = false;

  if(camObs) delete camObs;
  camObs = new CameraObs(*K, 180, 180);
  camObs->addObjectToMaskList(mugFrame->name);
  camObs->addObjectToMaskList(hookFrame->name);
  initDefaultCameraObs();
}

void MugEnvironment::watch() {
  K->watch();
}

void MugEnvironment::initCameraForVideo(bool showKeypoints) {
  if(camObsVideo) delete camObsVideo;
  if(KVideo) delete KVideo;

  KVideo = new rai::Configuration(*K);

  if(showKeypoints) {
    for(auto fName : {"mug_m1", "mug_m2"}) {
      KVideo->getFrame(fName)->setShape(rai::ShapeType::ST_sphere, {0.01}).setColor({1.0, 0.0, 0.0});
    }
    KVideo->getFrame("hanger_m1")->setShape(rai::ShapeType::ST_sphere, {0.016}).setColor({1.0, 0.0, 0.0});
  }


  camObsVideo = new CameraObs(*KVideo, 800, 800);
  camObsVideo->addObjectToMaskList(mugFrame->name);
  camObsVideo->addObjectToMaskList(hookFrame->name);
  
  addDefaultCamera(60.0, -10.0, camObsVideo);
  addDefaultCamera(70.0, 78.0, camObsVideo);
  addDefaultCamera(65.0, 185.0, camObsVideo);
  addDefaultCamera(60.0, 270.0, camObsVideo);
}

void MugEnvironment::initCameraForRoundView() {
  if(camObsVideo) delete camObsVideo;
  if(KVideo) delete KVideo;
  KVideo = new rai::Configuration(*K);
  camObsVideo = new CameraObs(*KVideo, 180, 180);
  camObsVideo->addObjectToMaskList(mugFrame->name);
  camObsVideo->addObjectToMaskList(hookFrame->name);
  
  for(uint i = 0; i < 360; i += 2) {
    addDefaultCamera(60.0, (double)i, camObsVideo);   
  }
}


torch::Tensor MugEnvironment::getState() {
  arr q = K->getJointState();
  return torch::from_blob(q.p, {q.N}, torch::kDouble).to(torch::kFloat).clone();
}

torch::Tensor MugEnvironment::getHighDimState() {
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

torch::Tensor MugEnvironment::getKeypoints() {
  arr KP;
  for(auto f : {"mug", "mug_m1", "mug_m2", "hanger_m1"}) {
    KP.append(K->getFrame(f)->getPosition());
  }
  return torch::from_blob(KP.p, {4,3}, torch::kDouble).to(torch::kFloat).clone();
}

void MugEnvironment::setState(const torch::Tensor& q) {
  arr qA(q.cpu().to(torch::kDouble).data_ptr<double>(), sizeof(torch::kDouble)*q.numel(), false);
  setState(qA);
}

void MugEnvironment::setState(const arr& q) {
  K->setJointState(q);
  phisAreUpdated = false;
  if(KVideo) KVideo->setJointState(q);
}

std::tuple<bool, bool> MugEnvironment::checkStateInSim() {
  ensurePhisAreUpdated();

  float collisionThreshold = 0.5;
  float k = 1000;
  float overlapIntegral
      = (torch::sigmoid(-k*phiMug.reshape({-1}))*torch::sigmoid(-k*phiHook.reshape({-1}))).sum().item().toFloat();
  bool collision = overlapIntegral >= collisionThreshold;
  if(collision) {
    return std::make_tuple(false, true);
  }

  arr mugPosBeforeSim = mugFrame->getPosition();
  arr frameStateBeforeSim = K->getFrameState();

  rai::Simulation sim(*K, rai::Simulation::_bullet, 0);
  for(uint i = 0; i < 500; i++) {
    sim.step({}, 0.01, rai::Simulation::ControlMode::_none);
  }
  sim.addImp(rai::Simulation::ImpType::_objectImpulseOnce, {"mug"}, {0.1, 0.1, 1.0});
  for(uint i = 0; i < 200; i++) {
    sim.step({}, 0.01, rai::Simulation::ControlMode::_none);
  }
  arr mugPosAfterSim = mugFrame->getPosition();
  bool success = mugPosAfterSim(2) > 0.25;
  double zDiff = fabs(mugPosAfterSim(2) - mugPosBeforeSim(2));
  success = success && zDiff < 0.09;

  K->setFrameState(frameStateBeforeSim);

  return std::make_tuple(success, false);
}

torch::Tensor MugEnvironment::getSDFs() {
  ensurePhisAreUpdated();
  return torch::cat({phiMug, phiHook}, 0); // 2, d, h, w
}

std::tuple<float, float, float, float, float, float> MugEnvironment::getBoundingBoxLimits() {
  const auto& bb = boundingBoxFrame->ImBoundingBox;
  return {
    bb->xLo, bb->xHi,
    bb->yLo, bb->yHi,
    bb->zLo, bb->zHi
  };
}

std::tuple<torch::Tensor, torch::Tensor, torch::Tensor, torch::Tensor, torch::Tensor> MugEnvironment::getObservation() {
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

std::tuple<torch::Tensor, torch::Tensor, torch::Tensor, torch::Tensor, torch::Tensor> MugEnvironment::getObservationVideo() {
  torch::Tensor I, M, K, KBounds;
  camObsVideo->getCameraObservations(I, M, K, KBounds);
  using namespace torch::indexing;
  torch::Tensor KT = K.unsqueeze(0).transpose(2,3);
  torch::Tensor E = torch::eye(4).reshape({1,1,4,4}).repeat({KT.size(0),KT.size(1),1,1});
  E.index_put_({Slice(), Slice(), Slice(None,4), Slice(None,3)}, KT);
  torch::Tensor KInvT = torch::linalg_inv(E);
  KInvT = KInvT.index({Slice(), Slice(), Slice(None,4), Slice(None,3)});
  return {I.unsqueeze(0).to(torch::kFloat)/255.0, M.unsqueeze(0).to(torch::kFloat), KT, KInvT, KBounds.unsqueeze(0)};
}

void MugEnvironment::ensurePhisAreUpdated() {
  if(!phisAreUpdated) {
    phiMug = ImS::getSDF(mugFrame, boundingBoxFrame, true);
    phiHook = ImS::getSDF(hookFrame, boundingBoxFrame, true);
    phisAreUpdated = true;
  }
}

void MugEnvironment::initDefaultCameraObs() {
  addDefaultCamera(60.0, -10.0);
  addDefaultCamera(70.0, 78.0);
  addDefaultCamera(65.0, 185.0);
  addDefaultCamera(60.0, 270.0);
}

void MugEnvironment::addAlternativeCameraObs() {
  addDefaultCamera(70.0, -15.0);
  addDefaultCamera(60.0, 70.0);
  addDefaultCamera(58.0, 170.0);
  addDefaultCamera(65.0, 290.0);
}

void MugEnvironment::addDefaultCamera(double theta, double phi, CameraObs* camObsOverwrite) {
  rai::Transformation T;
  T.setZero();
  T.rot.setRpy(0.0, theta*RAI_PI/180.0, phi*RAI_PI/180.0);
  T.addRelativeRotationDeg(90.0, 0.0, 0.0, 1.0);
  arr back = conv_vec2arr(T.rot*rai::Vector(0.0, 0.0, 0.5));
  arr center = boundingBoxFrame->getPosition();
  T.pos = center + back;
  
  CameraObs* co;
  if(camObsOverwrite) {
    co = camObsOverwrite;   
  } else {
    co = this->camObs;
  }
  co->addCamera(T, 0.8, {0.01, 0.9});
}
