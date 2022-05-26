#include "cameraObs.h"



CameraObs::CameraObs(rai::Configuration &K, uint h, uint w)
  : K(K), h(h), w(w), cameraView(K) {}

CameraObs::~CameraObs() {

}

void CameraObs::addCamera(const rai::Transformation& T, double focalLength, const arr& zRange) {
  uint cameraNumber = cameraView.sensors.N;
  std::string cameraName = "camera_" + std::to_string(cameraNumber);
  rai::Frame* cameraFrame = K.addFrame(cameraName.c_str());
  cameraFrame->setShape(rai::ShapeType::ST_marker, {0.1});

  cameraFrame->set_X() = T;
  cameraView.updateConfiguration(K);
  cameraView.addSensor(cameraName.c_str(), cameraName.c_str(), w, h, focalLength, -1., zRange);
}

void CameraObs::addObjectToMaskList(const char *frameName) {
  objectNames.append(frameName);
}

void CameraObs::getCameraObservations(torch::Tensor& I, torch::Tensor& M, torch::Tensor& KMat, torch::Tensor& KBounds) {
  cameraView.updateConfiguration(K);

  std::vector<torch::Tensor> imageList;
  std::vector<torch::Tensor> maskList;
  std::vector<torch::Tensor> KMatList;
  std::vector<torch::Tensor> KBoundsList;

  for(uint cameraNumber = 0; cameraNumber < getNumberOfViews(); cameraNumber++) {
    std::string cameraName = "camera_" + std::to_string(cameraNumber);
    cameraView.selectSensor(cameraName.c_str());

    arr KMatCurrent = cameraView.currentSensor->cam.getProjectionMatrix();
    // the following utilizes the fact that the data is stored in row major format and hence we can
    // get the 3x4 part of KMatCurrent this way
    torch::Tensor KMatCurrent_t = torch::from_blob(KMatCurrent.p, {3, 4}, torch::kDouble).clone().to(torch::kFloat).cpu();
    KMatList.push_back(KMatCurrent_t);

    torch::Tensor KBoundsCurrent_t = torch::tensor({cameraView.currentSensor->cam.zNear, cameraView.currentSensor->cam.zFar},
                                                   torch::kFloat);
    KBoundsList.push_back(KBoundsCurrent_t);

    byteA rgb;
    cameraView.renderMode = rai::CameraView::RenderMode::visuals;
    cameraView.computeImageAndDepth(rgb, NoFloatA);
    assert(rgb.d0 == h);
    assert(rgb.d1 == w);
    torch::Tensor rgb_t = torch::from_blob(rgb.p, {h, w, 3}, torch::kByte).clone().permute({2, 0, 1});
    imageList.push_back(rgb_t);

    std::vector<torch::Tensor> maskListInner;
    for(rai::String objectFrameName : objectNames) {
      byteA mask;
      cameraView.computeMask(mask, objectFrameName);
      torch::Tensor mask_t = torch::from_blob(mask.p, {h, w}, torch::kByte).clone();
      maskListInner.push_back(mask_t);
    }
    maskList.push_back(torch::stack(maskListInner));
  }

  I = torch::stack(imageList);         // V, C, H, W
  M = torch::stack(maskList);          // V, M, H, W
  KMat = torch::stack(KMatList);       // V, 3, 4
  using namespace torch::indexing;
  KMat.index({Slice(), 1}) *= -1.0f;
  KBounds = torch::stack(KBoundsList); // V, 2
}

uint CameraObs::getNumberOfViews() {
  return cameraView.sensors.N;
}

uint CameraObs::getNumberOfMasks() {
  return objectNames.N;
}
