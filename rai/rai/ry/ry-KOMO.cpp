/*  ------------------------------------------------------------------
    Copyright (c) 2011-2020 Marc Toussaint
    email: toussaint@tu-berlin.de

    This code is distributed under the MIT License.
    Please see <root-path>/LICENSE for details.
    --------------------------------------------------------------  */

#ifdef RAI_PYBIND

#include "types.h"
#include "ry-Config.h"
#include "../KOMO/komo.h"

//#include "../LGP/bounds.h"
#include "../Kin/viewer.h"
#include "../Kin/forceExchange.h"
#include "../Kin/F_FNO.h"
#include "../Kin/F_pose.h"


Skeleton list2skeleton(const pybind11::list& L) {
  Skeleton S;
  for(uint i=0; i<L.size(); i+=3) {
    std::vector<double> when = L[i].cast<std::vector<double>>();
    SkeletonSymbol symbol = L[i+1].cast<SkeletonSymbol>();
    ry::I_StringA frames = L[i+2].cast<ry::I_StringA>();
    S.append(SkeletonEntry(when[0], when[1], symbol, I_conv(frames)));
  }
  return S;
}

void checkView(shared_ptr<KOMO>& self){ if(self->pathConfig.hasView()) self->pathConfig.watch(); }

void init_KOMO(pybind11::module& m) {
  pybind11::class_<KOMO, std::shared_ptr<KOMO>>(m, "KOMO", "Constrained solver to optimize configurations or paths. (KOMO = k-order Markov Optimization)")

  .def(pybind11::init<>())

  .def("setModel", &KOMO::setModel)
  .def("setTiming", &KOMO::setTiming)

//  .def("makeObjectsFree", [](std::shared_ptr<KOMO>& self, const ry::I_StringA& objs) {
//    self->world.makeObjectsFree(I_conv(objs));
//  })

//  .def("activateCollisionPairs", [](std::shared_ptr<KOMO>& self, const std::vector<std::pair<std::string, std::string>>& collision_pairs) {
//    for(const auto&  pair : collision_pairs) {
//      self->activateCollisions(rai::String(pair.first), rai::String(pair.second));
//    }
//  })

//  .def("deactivateCollisionPairs", [](std::shared_ptr<KOMO>& self, const std::vector<std::pair<std::string, std::string>>& collision_pairs) {
//    for(const auto&  pair : collision_pairs) {
//      self->deactivateCollisions(rai::String(pair.first), rai::String(pair.second));
//    }
//  })

  .def("addTimeOptimization", &KOMO::addTimeOptimization)

  .def("clearObjectives", &KOMO::clearObjectives)

#if 0
      .def("addObjective",
       pybind11::overload_cast<const arr&, const FeatureSymbol&, const StringA&, ObjectiveType, const arr&, const arr&, int, int, int>
       (&KOMO::addObjective),
       "",
       pybind11::arg("times")=arr(),
       pybind11::arg("feature"),
       pybind11::arg("frames")=StringA(),
       pybind11::arg("type"),
       pybind11::arg("scale")=arr(),
       pybind11::arg("target")=arr(),
       pybind11::arg("order")=-1,
       pybind11::arg("deltaFromStep")=0,
       pybind11::arg("deltaToStep")=0
       )
#else
  .def("addObjective", [](std::shared_ptr<KOMO>& self, const arr& times, const FeatureSymbol& feature, const ry::I_StringA& frames, const ObjectiveType& type, const arr& scale, const arr& target, int order) {
        self->addObjective(times, feature, I_conv(frames), type, scale, target, order);
      }, "",
      pybind11::arg("times"),
      pybind11::arg("feature"),
      pybind11::arg("frames")=ry::I_StringA(),
      pybind11::arg("type"),
      pybind11::arg("scale")=arr(),
      pybind11::arg("target")=arr(),
      pybind11::arg("order")=-1)
#endif

//      .def("add_qControlObjective", [](std::shared_ptr<KOMO>& self, const std::vector<double>& time, uint order, double scale, const std::vector<double>& target) {
//        self->add_qControlObjective(arr(time, true), order, scale, arr(target, true));
//      }, "", pybind11::arg("time")=std::vector<double>(),
//      pybind11::arg("order"),
//      pybind11::arg("scale")=double(1.),
//      pybind11::arg("target")=std::vector<double>())

  .def("addSquaredQuaternionNorms",
       &KOMO::addSquaredQuaternionNorms,
       "",
       pybind11::arg("times")=arr(),
       pybind11::arg("scale")=3.
                              )

  .def("add_qControlObjective",
       &KOMO::add_qControlObjective,
       "",
       pybind11::arg("times"),
       pybind11::arg("order"),
       pybind11::arg("scale")=1.,
       pybind11::arg("target")=arr(),
       pybind11::arg("deltaFromStep")=0,
       pybind11::arg("deltaToStep")=0
                                        )
#if 0
  .def("addSwitch_stable",
       &KOMO::addSwitch_stable,
       "",
       pybind11::arg("startTime"),
       pybind11::arg("endTime"),
       pybind11::arg("prevFromFrame"),
       pybind11::arg("fromFrame"),
       pybind11::arg("toFrame")
       )
#endif
  .def("addSwitch_magic", &KOMO::addSwitch_magic)

  .def("addSwitch_dynamicTrans", &KOMO::addSwitch_dynamicTrans)

  .def("addInteraction_elasticBounce",
       &KOMO::addContact_elasticBounce,
       "",
       pybind11::arg("time"),
       pybind11::arg("from"),
       pybind11::arg("to"),
       pybind11::arg("elasticity") = .8,
       pybind11::arg("stickiness") = 0.)


  .def("addSkeleton", [](std::shared_ptr<KOMO>& self, const pybind11::list& L) {
    Skeleton S = list2skeleton(L);
    cout <<"SKELETON: " <<S <<endl;
    self->setSkeleton(S);
  //    skeleton2Bound(*self.komo, BD_path, S, self->world, self->world, false);
  })


//  .def("addFNO_PoI",
//       [](std::shared_ptr<KOMO>& self, const arr& times, const ry::I_StringA& frames, const uintA& ind1, const uintA& ind2, const ObjectiveType& type, const arr& scale) {
//         self->addSwitch(times, true, new rai::KinematicSwitch(rai::SW_addContact, rai::JT_none, frames[0].c_str(), frames[1].c_str(), self->world));
//         self->addObjective(times, make_shared<F_FNO_PoI>(ind1, ind2), I_conv(frames), type, scale);
//       },
//       "",
//       pybind11::arg("times"),
//       pybind11::arg("frames")=ry::I_StringA(),
//       pybind11::arg("ind1")=uintA(),
//       pybind11::arg("ind2")=uintA(),
//       pybind11::arg("type"),
//       pybind11::arg("scale")=arr()
//      )
   .def("addFNO_PoI",
        [](std::shared_ptr<KOMO>& self, const arr& times, const ry::I_StringA& frames, int ind2, const ObjectiveType& type, const arr& scale) {
          self->addFNO_PoI(times, I_conv(frames), ind2, type, scale);
        },
        "",
        pybind11::arg("times"),
        pybind11::arg("frames")=ry::I_StringA(),
        pybind11::arg("ind2")=0,
        pybind11::arg("type"),
        pybind11::arg("scale")=arr()
  )

    .def("addFNO_PoI2",
         [](std::shared_ptr<KOMO>& self, const arr& times, const ry::I_StringA& frames, int ind1, int ind2, const ObjectiveType& type, const arr& scale) {
 //          self->addSwitch(times, true, new rai::KinematicSwitch(rai::SW_addContact, rai::JT_none, frames[0].c_str(), frames[1].c_str(), self->world));
 //          self->addObjective(times, make_shared<F_FNO_PoI2>(ind1, ind2), I_conv(frames), type, scale);
           self->addFNO_PoI2(times, I_conv(frames), ind1, ind2, type, scale);
         },
         "",
         pybind11::arg("times"),
         pybind11::arg("frames")=ry::I_StringA(),
         pybind11::arg("ind1")=0,
         pybind11::arg("ind2")=0,
         pybind11::arg("type"),
         pybind11::arg("scale")=arr()
   )

//-- run

  .def("optimize", [](std::shared_ptr<KOMO>& self, double addInitializationNoise) {
    self->optimize(addInitializationNoise);
    checkView(self);
  }, "",
  pybind11::arg("addInitializationNoise")=0.01)

//-- reinitialize with configuration
  .def("setConfigurations", [](std::shared_ptr<KOMO>& self, shared_ptr<rai::Configuration>& C) {
    arr X = C->getFrameState();
    for(uint t=0;t<self->T;t++){
      self->pathConfig.setFrameState( X, self->timeSlices[t] );
    }
    checkView(self);
  })

//-- read out

  .def("getT", [](std::shared_ptr<KOMO>& self) {
    return self->T;
  })

  .def("getFrameState", &KOMO::getFrameState)

  .def("getPathFrames", &KOMO::getPath_frames)
//  .def("getPathFrames", [](std::shared_ptr<KOMO>& self, const ry::I_StringA& frames) {
//    arr X = self->getPath_frames(I_conv(frames));
//    return pybind11::array(X.dim(), X.p);
//  })

  .def("getPathTau", [](std::shared_ptr<KOMO>& self) {
    arr X = self->getPath_tau();
    return pybind11::array(X.dim(), X.p);
  })

  .def("getForceInteractions", [](std::shared_ptr<KOMO>& self) {
    rai::Graph G = self->pathConfig.reportForces();
    return graph2list(G);
  })

  .def("getReport", [](std::shared_ptr<KOMO>& self) {
//    rai::Graph G = self->getProblemGraph(true);
    rai::Graph R = self->getReport(true);
    return graph2dict(R);
  })

  .def("getProblemGraph", [](std::shared_ptr<KOMO>& self) {
    rai::Graph G = self->getProblemGraph(true);
    return graph2list(G);
  })

  .def("getConstraintViolations", [](std::shared_ptr<KOMO>& self) {
    rai::Graph R = self->getReport(false);
    return R.get<double>("ineq") + R.get<double>("eq");
  })

  .def("getCosts", [](std::shared_ptr<KOMO>& self) {
    rai::Graph R = self->getReport(false);
    return R.get<double>("sos");
  })

  .def("getPoI",
       [](std::shared_ptr<KOMO>& self, uint t, const char* from , const char* to) {
         pybind11::tuple tuple(2);
         FrameL frames = self->pathConfig.frames[self->k_order+t];
         rai::Frame* fromFrame = nullptr;
         for(rai::Frame* f : frames) {
           if(f->name == from) {
             fromFrame = f;
             break;
           }
         }
         rai::Frame* toFrame = nullptr;
         for(rai::Frame* f : frames) {
           if(f->name == to) {
             toFrame = f;
             break;
           }
         }

         Value pos = evalFeature<F_Position>({fromFrame});

         rai::Transformation toPose = toFrame->ensure_X();
         arr toRot = toPose.rot.getArr();
         arr to_x_rel = (~toRot)*(pos.y-conv_vec2arr(toPose.pos));

         arr JTo;
         toFrame->C.jacobian_pos(JTo, toFrame, pos.y);

         arr J2 = ~toRot*(pos.J - JTo);

         tuple[0] = to_x_rel;
         tuple[1] = rai::unpack(J2);
         return tuple;
       },
       ""
       )

  .def("getPoI2",
       [](std::shared_ptr<KOMO>& self, uint t, const char* from , const char* to) {
         pybind11::tuple tuple(4);
         FrameL frames = self->pathConfig.frames[self->k_order+t];
         rai::Frame* fromFrame = nullptr;
         for(rai::Frame* f : frames) {
           if(f->name == from) {
             fromFrame = f;
             break;
           }
         }
         rai::Frame* toFrame = nullptr;
         for(rai::Frame* f : frames) {
           if(f->name == to) {
             toFrame = f;
             break;
           }
         }
         rai::ForceExchange* forceExchange = rai::getContact(fromFrame, toFrame);

         arr poi, J_poi;
         forceExchange->kinPOA(poi, J_poi);

         rai::Transformation fromPose = fromFrame->ensure_X();
         arr fromRot = fromPose.rot.getArr();
         arr from_x_rel = (~fromRot)*(poi-conv_vec2arr(fromPose.pos));

         rai::Transformation toPose = toFrame->ensure_X();
         arr toRot = toPose.rot.getArr();
         arr to_x_rel = (~toRot)*(poi-conv_vec2arr(toPose.pos));

         arr JFrom, JTo;
         fromFrame->C.jacobian_pos(JFrom, fromFrame, poi);
         toFrame->C.jacobian_pos(JTo, toFrame, poi);

         arr J1 = ~fromRot*(J_poi - JFrom);
         arr J2 = ~toRot*(J_poi - JTo);

         tuple[0] = from_x_rel;
         tuple[1] = rai::unpack(J1);
         tuple[2] = to_x_rel;
         tuple[3] = rai::unpack(J2);
         return tuple;
       },
       ""
       )

//-- display

  .def("view", &KOMO::view)
  .def("view_play", &KOMO::view_play)
  .def("view_close", [](shared_ptr<KOMO>& self) {
    self->pathConfig.gl().reset();
  }, "close the view")

  ;


  //===========================================================================

  //  pybind11::class_<ry::ConfigViewer>(m, "ConfigViewer");
    pybind11::class_<Objective, shared_ptr<Objective>>(m, "KOMO_Objective");

  //===========================================================================

#define ENUMVAL(pre, x) .value(#x, pre##_##x)

  // pybind11::enum_<ObjectiveType>(m, "OT")
  // ENUMVAL(OT, none)
  // ENUMVAL(OT, f)
  // ENUMVAL(OT, sos)
  // ENUMVAL(OT, ineq)
  // ENUMVAL(OT, eq)
  // .export_values();

//pybind11::enum_<BoundType>(m, "BT")
//ENUMVAL(BD, all)
//ENUMVAL(BD, symbolic)
//ENUMVAL(BD, pose)
//ENUMVAL(BD, seq)
//ENUMVAL(BD, path)
//ENUMVAL(BD, seqPath)
//ENUMVAL(BD, max)
//.export_values();

  pybind11::enum_<SkeletonSymbol>(m, "SY")
  ENUMVAL(SY, touch)
  ENUMVAL(SY, above)
  ENUMVAL(SY, inside)
  ENUMVAL(SY, impulse)
  ENUMVAL(SY, stable)
  ENUMVAL(SY, stableOn)
  ENUMVAL(SY, dynamic)
  ENUMVAL(SY, dynamicOn)
  ENUMVAL(SY, dynamicTrans)
  //ENUMVAL(SY, liftDownUp)

  ENUMVAL(SY, contact)
  ENUMVAL(SY, bounce)

  ENUMVAL(SY, magic)

  ENUMVAL(SY, push)
  ENUMVAL(SY, graspSlide)
  .export_values();

}

#endif
