/*!
 * Copyright (c) 2015 by Contributors
 * \file cdropout.cu
 * \brief
 * \author Zehao Huang
*/

#include "./cdropout-inl.h"

namespace mxnet {
namespace op {
template<>
Operator *CreateOp<gpu>(CDropoutParam param) {
  return new CDropoutOp<gpu>(param);
}
}  // namespace op
}  // namespace mxnet


