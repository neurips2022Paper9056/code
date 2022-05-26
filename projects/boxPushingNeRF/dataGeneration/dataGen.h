#ifndef CAMERAOBS_H
#define CAMERAOBS_H

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

#include <Kin/kin.h>
#include <Kin/frame.h>
#include <Kin/cameraview.h>
#include <hdf5/serial/H5Cpp.h>

namespace {
template<typename T> struct DataType {};
template<> struct DataType<float> {
  inline static const H5::PredType H5Type = H5::PredType::NATIVE_FLOAT;
};
template<> struct DataType<int> {
  inline static const H5::PredType H5Type = H5::PredType::NATIVE_INT;
};
template<> struct DataType<uint8_t> {
  inline static const H5::PredType H5Type = H5::PredType::NATIVE_UINT8;
};
}

struct HDF5File {
  H5::H5File file;

  HDF5File(std::string fileName);
  ~HDF5File();

  template<typename T> void addDataset(const char* datasetName, std::vector<int64_t> dims) {
    uint nDim = dims.size();
    hsize_t dimsDataset[nDim];
    for(uint i = 0; i < nDim; i++) {
      dimsDataset[i] = dims[i];
    }
    file.createDataSet(datasetName, DataType<T>::H5Type, H5::DataSpace(nDim, dimsDataset));
  }

  template<typename T> void addToDataset(const char* datasetName, const torch::Tensor& tensor, uint index) {
    H5::DataSet dataset = file.openDataSet(datasetName);
    H5::DataSpace dataSpace = dataset.getSpace();
    uint nDims = dataSpace.getSimpleExtentNdims();
    assert(nDims == tensor.dim());
    hsize_t datasetSize[nDims];
    dataSpace.getSimpleExtentDims(datasetSize);
    std::vector<hsize_t> startDataset(nDims, 0);
    startDataset[0] = index;
    assert(index >= 0);
    assert(index < datasetSize[0]);
    std::vector<hsize_t> count(nDims, 0);
    for(uint i = 0; i < tensor.dim(); i++) {
      count[i] = tensor.size(i);
      if(i > 0) assert(count[i] == datasetSize[i]);
    }
    dataSpace.selectHyperslab(H5S_SELECT_SET, count.data(), startDataset.data());

    H5::DataSpace memSpace(nDims, count.data());
    std::vector<hsize_t> startMem(nDims, 0);
    memSpace.selectHyperslab(H5S_SELECT_SET, count.data(), startMem.data());
    dataset.write(tensor.cpu().data_ptr<T>(), DataType<T>::H5Type, memSpace, dataSpace);
  }

  template<typename T> void addToDataset(const char* datasetName, const rai::Array<T>& array, uint index) {
    H5::DataSet dataset = file.openDataSet(datasetName);
    H5::DataSpace dataSpace = dataset.getSpace();
    uint nDims = dataSpace.getSimpleExtentNdims();
    assert(nDims == array.nd);
    hsize_t datasetSize[nDims];
    dataSpace.getSimpleExtentDims(datasetSize);
    std::vector<hsize_t> startDataset(nDims, 0);
    startDataset[0] = index;
    assert(index >= 0);
    assert(index < datasetSize[0]);
    std::vector<hsize_t> count(nDims, 0);
    for(uint i = 0; i < array.nd; i++) {
      count[i] = array.d[i];
      if(i > 0) assert(count[i] == datasetSize[i]);
    }
    dataSpace.selectHyperslab(H5S_SELECT_SET, count.data(), startDataset.data());

    H5::DataSpace memSpace(nDims, count.data());
    std::vector<hsize_t> startMem(nDims, 0);
    memSpace.selectHyperslab(H5S_SELECT_SET, count.data(), startMem.data());
    dataset.write(array.p, DataType<T>::H5Type, memSpace, dataSpace);
  }

};

struct CameraObs {
  rai::Configuration& K;
  uint h;
  uint w;
  rai::CameraView cameraView;
  StringA objectNames;

  CameraObs(rai::Configuration& K, uint h, uint w);
  ~CameraObs();

  void addCamera(const rai::Transformation& T, double focalLength, const arr& zRange);
  void addObjectToMaskList(const char* frameName);
  void getCameraObservations(torch::Tensor& I, torch::Tensor& M, torch::Tensor& KMat, torch::Tensor& KBounds);

  void addToHDF5File(HDF5File& file, uint index);

  uint getNumberOfViews();
  uint getNumberOfMasks();
};

#endif // CAMERAOBS_H
