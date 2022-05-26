#include "pushingEnvironment.h"

PushingEnvironment::PushingEnvironment(rai::Configuration* K) : K{K} {
  tableFrame = K->getFrame("table");
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


void PushingEnvironment::setObjectPose(rai::Frame* object, double x, double y) {
  auto X = object->set_X();
  X->pos = {x, y, 0.57501};
}

void PushingEnvironment::setObjectPose(rai::Frame* object, double x, double y, double phi) {
  auto X = object->set_X();
  X->pos = {x, y, 0.57501};
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

rai::Frame* PushingEnvironment::addObjectHelper(rai::Configuration& K, std::string name) {
  double x = rnd.uni()*0.04 + 0.03;
  double y = rnd.uni()*0.04 + 0.03;

  rai::Frame& obj = K.addFrame(name.c_str(), "world")
                        ->setContact(1)
                        .setMass(1.0)
                        .setShape(rai::ShapeType::ST_ssBox, {x, y, 0.05, 0.002});
  obj.setJoint(rai::JointType::JT_rigid);
  obj.ats.newNode<double>("friction", {}, 0.5);
  obj.ats.newNode<double>("rollingFriction", {}, 0.001);
  obj.ats.newNode<double>("spinningFriction", {}, 0.001);

  uint colorRnd = rnd.num(0,1);
  arr color;
  if(colorRnd == 0) {
    color = {0.0, 0.0, 1.0};
  } else if(colorRnd == 1) {
    color = {1.0, 1.0, 0.0};
  }
  obj.setColor(color);
  return &obj;
}
