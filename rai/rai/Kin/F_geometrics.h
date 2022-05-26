/*  ------------------------------------------------------------------
    Copyright (c) 2011-2020 Marc Toussaint
    email: toussaint@tu-berlin.de

    This code is distributed under the MIT License.
    Please see <root-path>/LICENSE for details.
    --------------------------------------------------------------  */

#pragma once

#include "feature.h"

//===========================================================================

struct F_AboveBox : Feature {
  double margin=.0;
  virtual void phi2(arr& y, arr& J, const FrameL& F);
  virtual uint dim_phi2(const FrameL& F) { return 4; }
};

//===========================================================================

struct F_InsideBox : Feature {
  rai::Vector ivec;       ///< additional position or vector
  double margin;
  F_InsideBox(double _margin=.01) : margin(_margin) {}
  virtual void phi2(arr& y, arr& J, const FrameL& F);
  virtual uint dim_phi2(const FrameL& F) { return 6; }
};

//===========================================================================

struct TM_InsideLine : Feature {
  double margin;
  TM_InsideLine(double _margin=.03) : margin(_margin) {}
  virtual void phi2(arr& y, arr& J, const FrameL& F);
  virtual uint dim_phi2(const FrameL& F) { return 2; }
};

//===========================================================================

struct F_GraspOppose : Feature {
  bool centering=false;
  virtual void phi2(arr& y, arr& J, const FrameL& F);
  virtual uint dim_phi2(const FrameL& F) { if(centering) return 6; return 3; }
};

