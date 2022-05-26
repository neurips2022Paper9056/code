/*  ------------------------------------------------------------------
    Copyright (c) 2011-2020 Marc Toussaint
    email: toussaint@tu-berlin.de

    This code is distributed under the MIT License.
    Please see <root-path>/LICENSE for details.
    --------------------------------------------------------------  */

#include "ann.h"
#include "algos.h"

#ifdef RAI_ANN

#include <ANN/ANN.h>

struct sANN {
  ANNkd_tree* tree=0;
  //PartialLeastSquares pls;
  rai::Array<double*> cpointers;
  uint treeSize;   //for how many entries in X have we build the tree?
  void clear() { if(tree) delete tree;   tree=nullptr;  cpointers.clear();  treeSize=0; }
};

ANN::ANN() {
  bufferSize = 1 <<10;
  self = make_unique<sANN>();
  self->tree = 0;
  self->treeSize = 0;
}

ANN::ANN(const ANN& ann) {
  bufferSize = 1 <<10;
  self = make_unique<sANN>();
  self->tree = 0;
  self->treeSize = 0;
  setX(ann.X);
}

ANN::~ANN() {
  self->clear();
  //annClose(); //mt09-07-29 this would close down the ANN lib completely
}

void ANN::clear() {
  self->clear();
  X.clear();
}

void ANN::setX(const arr& _XX) {
  self->clear();
  X=_XX;
}

void ANN::append(const arr& x) {
  double* p=X.p;
  X.append(x);
  if(X.N==x.d0) X.reshape(1, x.d0);
  if(X.p!=p) self->clear(); //when the memory location changed clear the tree! (implies recomputation)
}

void ANN::calculate() {
  if(self->treeSize == X.d0) return;
  self->clear();
  X.getCarray(self->cpointers);
  self->tree = new ANNkd_tree(self->cpointers.p, X.d0, X.d1);
  self->treeSize = X.d0;
}

void ANN::getkNN(arr& dists, intA& idx, const arr& x, uint k, double eps, bool verbose) {
  CHECK_GE(X.d0, k, "data has less (" <<X.d0 <<") than k=" <<k <<" points");
  CHECK_EQ(x.N, X.d1, "query point has wrong dimension. x.N=" << x.N << ", X.d1=" << X.d1);

  if(X.d0-self->treeSize>bufferSize) {
    if(verbose) std::cout <<"ANN recomputing: X.d0=" <<X.d0 <<" treeSize=" <<self->treeSize <<std::endl;
    calculate();
  }
  uint restStartsAt;
  if(self->treeSize>=k) {
    dists.resize(k);
    idx.resize(k);
    self->tree->annkSearch(x.p, k, idx.p, dists.p, eps);
    restStartsAt=self->treeSize;
  } else {
    dists.clear();
    idx.clear();
    restStartsAt=0;
  }

  //now check if in the rest of X there are even nearer points
  for(uint i=restStartsAt; i<X.d0; i++) {
    for(uint j=0; j<=idx.N && j<k; j++) {
      double d=sqrDistance(X[i], x);
      if(j==idx.N || d < dists(j)) {
        idx.insert(j, i);
        dists.insert(j, d);
        break;
      }
    }
  }
  if(idx.N>k) {
    idx.resizeCopy(k);
    dists.resizeCopy(k);
  }

  if(verbose) {
    std::cout
        <<"ANN query:"
        <<"\n data size = " <<X.d0 <<"  data dim = " <<X.d1 <<"  treeSize = " <<self->treeSize
        <<"\n query point " <<x
        <<"\n found neighbors:\n";
    for(uint i=0; i<idx.N; i++) {
      std::cout <<' '
                <<i <<' '
                <<idx(i) <<'\t'
                <<sqrt(dists(i)) <<'\t'
                <<X[idx(i)] <<std::endl;
    }
  }
}

uint ANN::getNN(const arr& x, double eps, bool verbose) {
  intA idx;
  arr dists;
  getkNN(dists, idx, x, 1, eps, verbose);
  return idx(0);
}

void ANN::getkNN(intA& idx, const arr& x, uint k, double eps, bool verbose) {
  arr dists;
  getkNN(dists, idx, x, k, eps, verbose);
}

void ANN::getkNN(arr& xx, const arr& x, uint k, double eps, bool verbose) {
  intA idx;
  arr dists;
  getkNN(dists, idx, x, k, eps, verbose);
  xx.resize(idx.N, X.d1);
  for(uint i=0; i<idx.N; i++) xx[i]=X[idx(i)];
}

#else //RAI_ANN

struct sANN {};

ANN::ANN() { NICO }
ANN::~ANN() { NICO }

void ANN::calculate() { NICO }
void ANN::setX(const arr& _XX) { NICO }
void ANN::append(const arr& x) { NICO }
uint ANN::getNN(const arr& x, double eps, bool verbose) { NICO }
void ANN::getkNN(intA& idx, const arr& x, uint k, double eps, bool verbose) { NICO }
void ANN::getkNN(arr& sqrDists, intA& idx, const arr& x, uint k, double eps, bool verbose) { NICO }
void ANN::getkNN(arr& X, const arr& x, uint k, double eps, bool verbose) { NICO }

#endif
