/*!
 * Copyright (c) 2015 by Contributors
 * \file cdropout.cc
 * \brief
 * \author Zehao Huang
*/

#include "./cdropout-inl.h"

namespace mxnet {
namespace op {
template<>
Operator *CreateOp<cpu>(CDropoutParam param) {
  return new CDropoutOp<cpu>(param);
}

// DO_BIND_DISPATCH comes from operator_common.h
Operator *CDropoutProp::CreateOperator(Context ctx) const {
  DO_BIND_DISPATCH(CreateOp, param_);
}

DMLC_REGISTER_PARAMETER(CDropoutParam);

MXNET_REGISTER_OP_PROPERTY(CDropout, CDropoutProp)
.describe("Apply cdropout to input")
.add_argument("data", "Symbol", "Input data to cdropout.")
.add_arguments(CDropoutParam::__FIELDS__());

}  // namespace op
}  // namespace mxnet


