#include "pushingEnvironment.h"


PushingEnvironment::PushingEnvironment() {}

PushingEnvironment::~PushingEnvironment() {
  if(K) delete K;
  if(camObs) delete camObs;
  if(S) delete S;
}

uint PushingEnvironment::initEnvironment(uint seed, int pos_seed) {
  rnd.seed(seed);

  if(K) delete K;
  if(S) delete S;
  allVisObjectFrames.clear();
  objectFrames.clear();

  K = new rai::Configuration;
  K->addFile("world.g");
  tableFrame = K->getFrame("table");

  computeBoundsFromTableFrame();

  initDefaultCameraObs();

  // add pusher and object of random size and color
  addPusher();
  double dx = rnd.uni()*0.04 + 0.03;
  double dy = rnd.uni()*0.04 + 0.03;
  uint colorType = rnd.num(0,1);
  addObject("o1", colorType, dx, dy);

  // optional seed for pose sampling part
  if(pos_seed >= 0){
    rnd.seed(pos_seed);
  }

  // sample poses of pusher and object
  sampleObjectPose(pusherFrame, true);
  while(sampleObjectPose(K->getFrame("o1"), false));

  return colorType;
}

std::tuple<torch::Tensor, torch::Tensor, torch::Tensor, torch::Tensor, torch::Tensor> PushingEnvironment::getObservation() {
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

torch::Tensor PushingEnvironment::getState() {
  arr state;
  arr posPusher = pusherFrame->getPosition();
  state.append(~arr{posPusher(0), posPusher(1)});

  for(auto* obj : objectFrames) {
    arr pos = obj->getPosition();
    state.append(~arr{pos(0), pos(1)});
    arr quat = obj->getQuaternion();
    state.append(~quat);
  }
  return torch::from_blob(state.p, {state.N}, torch::kDouble).to(torch::kFloat).clone();
}

torch::Tensor PushingEnvironment::getHighDimState() {
  uint L = 10;
  
  arr q;
  arr posPusher = pusherFrame->getPosition();
  q.append(arr{posPusher(0), posPusher(1)});

  for(auto* obj : objectFrames) {
    arr pos = obj->getPosition();
    q.append(arr{pos(0), pos(1)});
    arr quat = obj->getQuaternion();
    q.append(quat);
  }
  
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

torch::Tensor PushingEnvironment::getKeypoints() {
  arr KP;
  for(auto* obj : objectFrames) {
    KP.append(obj->getPosition());
    KP.append(K->getFrame(STRING(obj->name << "_m1"))->getPosition());
    KP.append(K->getFrame(STRING(obj->name << "_m2"))->getPosition());
  }
  KP.append(K->getFrame("pusher")->getPosition());
  return torch::from_blob(KP.p, {KP.N/3,3}, torch::kDouble).to(torch::kFloat).clone();
}


void PushingEnvironment::watch() {
  K->watch(false);
}

void PushingEnvironment::addPusher() {
  rai::Frame* pusher = addPusherHelper(*K);
  allVisObjectFrames.append(pusher);
  pusherFrame = pusher;
  camObs->addObjectToMaskList(pusher->name);
}


void PushingEnvironment::addObject(const char *name, uint colorType, double dx, double dy) {
  rai::Frame* obj = addObjectHelper(*K, name, colorType, dx, dy);
  allVisObjectFrames.append(obj);
  objectFrames.append(obj);
  camObs->addObjectToMaskList(name);
  K->addFrame(STRING(name << "_m1"), name)->setShape(rai::ShapeType::ST_marker, {0.1})
      .set_Q()->addRelativeTranslation(dx/2.0, dy/2.0, 0.05/2.0);
  K->addFrame(STRING(name << "_m2"), name)->setShape(rai::ShapeType::ST_marker, {0.1})
      .set_Q()->addRelativeTranslation(-dx/2.0, -dy/2.0, -0.05/2.0);
}

void PushingEnvironment::setObjectPose(const char *frameName, double x, double y) {
  setObjectPose(K->getFrame(frameName), x, y);
}

void PushingEnvironment::setObjectPose(const char *frameName, double x, double y, double phi) {
  setObjectPose(K->getFrame(frameName), x, y, phi);
}

void PushingEnvironment::initSimulation() {
  if(S) delete S;
  S = new rai::Simulation(*K, rai::Simulation::_bullet, 0);
}

void PushingEnvironment::simulateStep(std::vector<double> pusherDelta) {
  arr posPusherDelta = {pusherDelta[0], pusherDelta[1], 0};
  arr posPusherAct = pusherFrame->getPosition();

  for(uint t = 0; t < 10; t++) {
    arr posPusherRefAct = ((double)t + 1.0)/10.0 * posPusherDelta + posPusherAct;
    pusherFrame->setPosition(posPusherRefAct);
    arr q = K->getJointState();
    S->step(q, 0.01, rai::Simulation::_position);

    for(auto* obj : objectFrames) {
      arr pos = obj->getPosition();
      setObjectPose(obj, pos(0), pos(1));
    }
  }
}


bool PushingEnvironment::sampleObjectPose(rai::Frame* frame, bool twoD) {
  double x = rnd.uni()*(xSamplingLimHi-xSamplingLimLo) + xSamplingLimLo;
  double y = rnd.uni()*(ySamplingLimHi-ySamplingLimLo) + ySamplingLimLo;
  if(!twoD) {
    double phi = rnd.uni()*90.0;
    setObjectPose(frame, x, y, phi);
  } else {
    setObjectPose(frame, x, y);
  }
  frame->C.stepSwift();
  arr coll, J;
  F_AccumulatedCollisions FC(0.0);
  uintA frameIDs;
  for(auto* f : allVisObjectFrames) frameIDs.append(f->ID);
  FC.setFrameIDs(frameIDs);
  FC.__phi(coll, J, frame->C);
  return coll(0) > 0.0;
}

bool PushingEnvironment::checkIfAllWithinBounds() {
  for(auto* f : allVisObjectFrames) {
    if(!checkIfWithinBounds(f)) return false;
  }
  return true;
}

bool PushingEnvironment::checkIfObjectsWithinBounds() {
  for(auto* f : objectFrames) {
    if(!checkIfWithinBounds(f)) return false;
  }
  return true;
}

bool PushingEnvironment::checkIfPusherWithinBounds() {
  return checkIfWithinBounds(pusherFrame);
}

bool PushingEnvironment::checkIfWithinBounds(rai::Frame* frame) {
  arr pos = frame->getPosition();
  return !(pos(0) < xLimLo || pos(0) > xLimHi || pos(1) < yLimLo || pos(1) > yLimHi);
}

void PushingEnvironment::computeBoundsFromTableFrame() {
  arr tmp = tableFrame->getSize();
  double xTable = tmp(0)*0.8;
  double yTable = tmp(1)*0.8;
  xLimLo = -xTable/2.0;
  xLimHi = -xLimLo;
  yLimLo = -yTable/2.0;
  yLimHi = -yLimLo;
  xSamplingLimLo = -xTable*0.9/2.0;
  xSamplingLimHi = -xSamplingLimLo;
  ySamplingLimLo = -yTable*0.9/2.0;
  ySamplingLimHi = -ySamplingLimLo;
}

void PushingEnvironment::initDefaultCameraObs() {
  uint h = 150;
  uint w = 200;
  if(camObs) delete camObs;
  camObs = new CameraObs(*K, h, w);
  camObs->addCamera(rai::Transformation({0.0, 0.05, 1.0}),
                   0.89, {0.3, 0.6});
  camObs->addCamera(rai::Transformation({0.0, -0.2, 1.0}).addRelativeRotationDeg(30.0, 1.0, 0.0, 0.0),
                   0.89, {0.3, 0.7});
  camObs->addCamera(rai::Transformation({0.5, 0.0, 0.7}).addRelativeRotationDeg(90.0, 0.0, 1.0, 0.0).addRelativeRotationDeg(90.0, 0.0, 0.0, 1.0),
                   0.89, {0.3, 0.8});
  camObs->addCamera(rai::Transformation({-0.3, 0.05, 0.8}).addRelativeRotationDeg(-40.0, 0.0, 1.0, 0.0).addRelativeRotationDeg(-90.0, 0.0, 0.0, 1.0),
                   0.89, {0.2, 0.65});
}


void PushingEnvironment::setObjectPose(rai::Frame* object, double x, double y) {
  auto X = object->set_X();
  X->pos = {x, y, ZHEIGHTCOORDINATE};
}

void PushingEnvironment::setObjectPose(rai::Frame* object, double x, double y, double phi) {
  auto X = object->set_X();
  X->pos = {x, y, ZHEIGHTCOORDINATE};
  X->rot.setRadZ(phi);
}


rai::Frame* PushingEnvironment::addPusherHelper(rai::Configuration& K) {
  double radius = 0.015;
  rai::Frame& obj = K.addFrame("pusher", "world")
                        ->setContact(1)
                        .setShape(rai::ShapeType::ST_capsule, {0.02, radius});
  obj.setJoint(rai::JointType::JT_free);
  obj.ats.newNode<double>("friction", {}, 0.1);
  obj.ats.newNode<double>("rollingFriction", {}, 0.0);
  obj.ats.newNode<double>("spinningFriction", {}, 0.0);
  obj.setColor({1.0, 0.0, 0.0});
  return &obj;
}


rai::Frame *PushingEnvironment::addObjectHelper(rai::Configuration &K, std::string name, uint colorType, double dx, double dy) {
  rai::Frame& obj = K.addFrame(name.c_str(), "world")
                        ->setContact(1)
                        .setMass(1.0)
                        .setShape(rai::ShapeType::ST_ssBox, {dx, dy, 0.05, 0.002});
  obj.setJoint(rai::JointType::JT_rigid);
  obj.ats.newNode<double>("friction", {}, 0.5);
  obj.ats.newNode<double>("rollingFriction", {}, 0.001);
  obj.ats.newNode<double>("spinningFriction", {}, 0.001);

  arr color;
  if(colorType == 0) {
    color = {0.0, 0.0, 1.0};
  } else if(colorType == 1) {
    color = {1.0, 1.0, 0.0};
  }
  obj.setColor(color);
  return &obj;
}
