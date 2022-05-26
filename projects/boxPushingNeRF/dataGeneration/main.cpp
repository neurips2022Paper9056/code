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
#include "dataGen.h"
#include "pushingEnvironment.h"
#include "json.hpp"


void generateData(std::string dataPath, uint numberOfObjects, long seed) {
  rnd.seed(seed);
  
  dataPath = "data/" + dataPath;
  system("mkdir -p data");
  HDF5File data(dataPath + ".hdf5");

  uint B = 100000;
  uint numberOfViews = 4;
  uint h = 150;
  uint w = 200;

  data.addDataset<uint8_t>("I", {B, numberOfViews, 3, h, w});
  data.addDataset<uint8_t>("M", {B, numberOfViews, numberOfObjects + 1, h, w});
  data.addDataset<float>("K", {B, numberOfViews, 3, 4});
  data.addDataset<float>("KBounds", {B, numberOfViews, 2});
  data.addDataset<float>("a", {B, 2});

  nlohmann::json envActionIndexData;

  uint datasetIndex = 0;

  for(uint envIndex = 0; envIndex < B; envIndex++) {
    rai::Configuration K;
    K.addFile("world.g");
    PushingEnvironment env(&K);
    env.computeBoundsFromTableFrame();

    rai::Frame* pusher = env.addPusherHelper(K);
    env.allVisObjectFrames.append(pusher);
    env.pusherFrame = pusher;
    env.sampleObjectPose(pusher, true);

    for(uint i = 0; i < numberOfObjects; i++) {
      rai::Frame* obj = env.addObjectHelper(K, "o_" + std::to_string(i));
      env.allVisObjectFrames.append(obj);
      env.objectFrames.append(obj);
      while(env.sampleObjectPose(obj, false));
    }

    CameraObs camObs(K, h, w);
    camObs.addCamera(rai::Transformation({0.0, 0.05, 1.0}),
                     0.89, {0.3, 0.6});
    camObs.addCamera(rai::Transformation({0.0, -0.2, 1.0}).addRelativeRotationDeg(30.0, 1.0, 0.0, 0.0),
                     0.89, {0.3, 0.7});
    camObs.addCamera(rai::Transformation({0.5, 0.0, 0.7}).addRelativeRotationDeg(90.0, 0.0, 1.0, 0.0).addRelativeRotationDeg(90.0, 0.0, 0.0, 1.0),
                     0.89, {0.3, 0.8});
    camObs.addCamera(rai::Transformation({-0.3, 0.05, 0.8}).addRelativeRotationDeg(-40.0, 0.0, 1.0, 0.0).addRelativeRotationDeg(-90.0, 0.0, 0.0, 1.0),
                     0.89, {0.2, 0.65});

    for(auto* f : env.allVisObjectFrames) {
      camObs.addObjectToMaskList(f->name);
    }

//    K.watch(true);

    rai::Simulation S(K, rai::Simulation::_bullet, 0);

    std::vector<int> envIndices;

    // pushing loop
    arr d = {0.0, 0.0, 0.0};
    for(uint i = 0; i < 10000; i++) {
      arr posPusher = pusher->getPosition();

      if(i == 0 || !env.checkIfWithinBounds(env.pusherFrame)) {
        uint objectIndex = rnd.num(0,numberOfObjects-1);
        d = K.getFrame(STRING("o_" << objectIndex))->getPosition() - posPusher;
      }

      arr r = ARR(rnd.gauss()*0.01, rnd.gauss()*0.01, 0.0);
      d += r;
      d(2) = 0.0;
      d /= length(d);

      arr posPusherNext = posPusher + 0.02*d;

      camObs.addToHDF5File(data, datasetIndex);
      arr deltaPusher = posPusherNext - posPusher;
      data.addToDataset<float>("a", floatA{(float)deltaPusher(0), (float)deltaPusher(1)}.reshape(1,2), datasetIndex);
      envIndices.push_back(datasetIndex);
      datasetIndex++;
      cout << datasetIndex << endl;
      if(datasetIndex >= B) break;

      arr posPusherAct = pusher->getPosition();

      for(uint t = 0; t < 10; t++) {
        arr posPusherRefAct = ((double)t + 1.0)/10.0 * (posPusherNext - posPusherAct) + posPusherAct;
        pusher->setPosition(posPusherRefAct);
        arr q = K.getJointState();
        S.step(q, 0.01, rai::Simulation::_position);

        for(auto* obj : env.objectFrames) {
          arr pos = obj->getPosition();
          env.setObjectPose(obj, pos(0), pos(1));
        }
      }

      //rai::wait(0.1);
      //K.watch(false);

      if(!env.checkIfObjectsWithinBounds()) break;
    } // pushing loop

    envActionIndexData.push_back(envIndices);

    if(datasetIndex >= B) break;

  } // env loop

  ofstream file(dataPath + ".json");
  file << envActionIndexData.dump();
}



///////////////////////////////////////////////////////////////////////////////




void generateDataWithAlternativeViews(std::string dataPath, uint numberOfObjects, long seed, uint B = 100000) {
  rnd.seed(seed);
  
  dataPath = "data/" + dataPath;
  system("mkdir -p data");
  HDF5File data(dataPath + ".hdf5");

  uint numberOfViews = 8;
  uint h = 150;
  uint w = 200;

  data.addDataset<uint8_t>("I", {B, numberOfViews, 3, h, w});
  data.addDataset<uint8_t>("M", {B, numberOfViews, numberOfObjects + 1, h, w});
  data.addDataset<float>("K", {B, numberOfViews, 3, 4});
  data.addDataset<float>("KBounds", {B, numberOfViews, 2});
  data.addDataset<float>("a", {B, 2});

  nlohmann::json envActionIndexData;

  uint datasetIndex = 0;

  for(uint envIndex = 0; envIndex < B; envIndex++) {
    rai::Configuration K;
    K.addFile("world.g");
    PushingEnvironment env(&K);
    env.computeBoundsFromTableFrame();

    rai::Frame* pusher = env.addPusherHelper(K);
    env.allVisObjectFrames.append(pusher);
    env.pusherFrame = pusher;
    env.sampleObjectPose(pusher, true);

    for(uint i = 0; i < numberOfObjects; i++) {
      rai::Frame* obj = env.addObjectHelper(K, "o_" + std::to_string(i));
      env.allVisObjectFrames.append(obj);
      env.objectFrames.append(obj);
      while(env.sampleObjectPose(obj, false));
    }

    CameraObs camObs(K, h, w);
    camObs.addCamera(rai::Transformation({0.0, 0.05, 1.0}),
                     0.89, {0.3, 0.6});
    camObs.addCamera(rai::Transformation({0.0, -0.2, 1.0}).addRelativeRotationDeg(30.0, 1.0, 0.0, 0.0),
                     0.89, {0.3, 0.7});
    camObs.addCamera(rai::Transformation({0.5, 0.0, 0.7}).addRelativeRotationDeg(90.0, 0.0, 1.0, 0.0).addRelativeRotationDeg(90.0, 0.0, 0.0, 1.0),
                     0.89, {0.3, 0.8});
    camObs.addCamera(rai::Transformation({-0.3, 0.05, 0.8}).addRelativeRotationDeg(-40.0, 0.0, 1.0, 0.0).addRelativeRotationDeg(-90.0, 0.0, 0.0, 1.0),
                     0.89, {0.2, 0.65});
                     
    // Alternative views
    camObs.addCamera(rai::Transformation({0.0, 0.0, 0.95}),
                     0.89, {0.3, 0.6});
    camObs.addCamera(rai::Transformation({0.05, -0.24, 1.0}).addRelativeRotationDeg(35.0, 1.0, 0.0, 0.0),
                     0.89, {0.3, 0.7});
    camObs.addCamera(rai::Transformation({0.4, 0.0, 0.7}).addRelativeRotationDeg(90.0, 0.0, 1.0, 0.0).addRelativeRotationDeg(90.0, 0.0, 0.0, 1.0),
                     0.89, {0.3, 0.8});
    camObs.addCamera(rai::Transformation({-0.4, -0.05, 0.8}).addRelativeRotationDeg(-40.0, 0.0, 1.0, 0.0).addRelativeRotationDeg(-90.0, 0.0, 0.0, 1.0),
                     0.89, {0.2, 0.65});

    for(auto* f : env.allVisObjectFrames) {
      camObs.addObjectToMaskList(f->name);
    }

//    K.watch(true);

    rai::Simulation S(K, rai::Simulation::_bullet, 0);

    std::vector<int> envIndices;

    // pushing loop
    arr d = {0.0, 0.0, 0.0};
    for(uint i = 0; i < 10000; i++) {
      arr posPusher = pusher->getPosition();

      if(i == 0 || !env.checkIfWithinBounds(env.pusherFrame)) {
        uint objectIndex = rnd.num(0,numberOfObjects-1);
        d = K.getFrame(STRING("o_" << objectIndex))->getPosition() - posPusher;
      }

      arr r = ARR(rnd.gauss()*0.01, rnd.gauss()*0.01, 0.0);
      d += r;
      d(2) = 0.0;
      d /= length(d);

      arr posPusherNext = posPusher + 0.02*d;

      camObs.addToHDF5File(data, datasetIndex);
      arr deltaPusher = posPusherNext - posPusher;
      data.addToDataset<float>("a", floatA{(float)deltaPusher(0), (float)deltaPusher(1)}.reshape(1,2), datasetIndex);
      envIndices.push_back(datasetIndex);
      datasetIndex++;
      cout << datasetIndex << endl;
      if(datasetIndex >= B) break;

      arr posPusherAct = pusher->getPosition();

      for(uint t = 0; t < 10; t++) {
        arr posPusherRefAct = ((double)t + 1.0)/10.0 * (posPusherNext - posPusherAct) + posPusherAct;
        pusher->setPosition(posPusherRefAct);
        arr q = K.getJointState();
        S.step(q, 0.01, rai::Simulation::_position);

        for(auto* obj : env.objectFrames) {
          arr pos = obj->getPosition();
          env.setObjectPose(obj, pos(0), pos(1));
        }
      }

      //rai::wait(0.1);
      //K.watch(false);

      if(!env.checkIfObjectsWithinBounds()) break;
    } // pushing loop

    envActionIndexData.push_back(envIndices);

    if(datasetIndex >= B) break;

  } // env loop

  ofstream file(dataPath + ".json");
  file << envActionIndexData.dump();
}



///////////////////////////////////////////////////////////////////////////////









int main(int argc,char **argv){
  rai::initCmdLine(argc, argv);

  generateData("nM_1", 1, 0);
  //generateDataWithAlternativeViews("nM_1_withAlternativeViews", 1, 0);
  //generateDataWithAlternativeViews("nM_1_withAlternativeViews_test", 1, 123456, 5000);
  
  return 0;
}
