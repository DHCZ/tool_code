/*!
 * Copyright (c) 2015 by Contributors
 * \file cdropout-inl.h
 * \brief
 * \author Zehao Huang
*/

#ifndef MXNET_OPERATOR_CDROPOUT_INL_H_
#define MXNET_OPERATOR_CDROPOUT_INL_H_
#include <dmlc/logging.h>
#include <dmlc/parameter.h>
#include <mxnet/operator.h>
#include <map>
#include <vector>
#include <string>
#include <utility>
#include "./operator_common.h"
#include "./mshadow_op.h"

namespace cdropout {
enum CDropoutOpInputs {kData};
enum CDropoutOpOutputs {kOut, kMask};
enum CDropoutOpForwardResource {kRandom, kTempSpace};
}  // namespace dropout

namespace mxnet {
namespace op {

struct CDropoutParam : public dmlc::Parameter<CDropoutParam> {
  float p;
  DMLC_DECLARE_PARAMETER(CDropoutParam) {
    DMLC_DECLARE_FIELD(p).set_default(0.5)
    .set_range(0, 1)
    .describe("Fraction of the input that gets dropped out at training time");
  }
};  // struct DropoutParam

template<typename xpu>
class CDropoutOp : public Operator {
 public:
  explicit CDropoutOp(CDropoutParam param) {
    this->pkeep_ = 1.0f - param.p;
  }

  virtual void Forward(const OpContext &ctx,
                       const std::vector<TBlob> &in_data,
                       const std::vector<OpReqType> &req,
                       const std::vector<TBlob> &out_data,
                       const std::vector<TBlob> &aux_states) {
    using namespace mshadow;
    using namespace mshadow::expr;
    CHECK_EQ(in_data.size(), 1);
    if (ctx.is_train) {
      CHECK_EQ(out_data.size(), 2);
    }
    Stream<xpu> *s = ctx.get_stream<xpu>();
    Tensor<xpu, 2> data = in_data[cdropout::kData].FlatTo2D<xpu, real_t>(s);
    Tensor<xpu, 2> out = out_data[cdropout::kOut].FlatTo2D<xpu, real_t>(s);
    /*
    int batch_size    = in_data[cdropout::kData].size(0);
    int channel       = in_data[cdropout::kData].size(1);
    int featuremap_h  = in_data[cdropout::kData].size(2);
    int featuremap_w  = in_data[cdropout::kData].size(3);

    //std::cout << data.shape_ << std::endl;
    //std::cout << out.shape_ << std::endl;
    if (ctx.is_train) {
      Tensor<xpu, 2> mask = out_data[cdropout::kMask].FlatTo2D<xpu, real_t>(s);
      Random<xpu> *prnd = ctx.requested[cdropout::kRandom].get_random<xpu, real_t>(s);
      //std::cout << mask.shape_ << std::endl;
      Tensor<xpu, 1> cmask = NewTensor<xpu>(Shape1(channel),0.0f);
      cmask = F<mshadow_op::threshold>(prnd->uniform(cmask.shape_), pkeep_) * (1.0f / pkeep_);
      //for (index_t i = 0; i < cmask.size(0); ++i) {
      	//std::cout << cmask[i] << std::endl;
      //}
      Tensor<xpu, 4> mask4 = NewTensor<xpu>(Shape4(batch_size,channel,featuremap_h,featuremap_w),0.0f);
      mask4 = broadcast<1>(cmask, mask4.shape_);
      */
      /*
      for (index_t i = 0; i < mask4.size(1); ++i) {
      	for (index_t j = 0; j < mask4.size(0); ++j) {
      		for (index_t k = 0; k < mask4.size(2); ++k) {
      			for (index_t l = 0; l < mask4.size(3); ++l) {
      				std::cout << mask4[j][i][k][l] << std::endl;
      			}
      		}
      	}
      }
	  */
      //mask = reshape(mask4,mask.shape_);
      


    if (ctx.is_train) {
      int batch_size    = in_data[cdropout::kData].size(0);
      int channel       = in_data[cdropout::kData].size(1);
      int featuremap_h  = in_data[cdropout::kData].size(2);
      int featuremap_w  = in_data[cdropout::kData].size(3);

      Tensor<xpu, 2> mask = out_data[cdropout::kMask].FlatTo2D<xpu, real_t>(s);
      Random<xpu> *prnd = ctx.requested[cdropout::kRandom].get_random<xpu, real_t>(s);
      Tensor<xpu, 1> cmask = ctx.requested[cdropout::kTempSpace].get_space<xpu>(mshadow::Shape1(batch_size*channel), s);
      cmask = F<mshadow_op::threshold>(prnd->uniform(cmask.shape_), pkeep_) * (1.0f / pkeep_);
      //for (index_t i = 0; i < cmask.size(0); ++i) {
        //std::cout << cmask[i] << std::endl;
      //}
      mask = reshape(broadcast<0>(cmask, Shape3(batch_size*channel, featuremap_h, featuremap_w)),mask.shape_);
      //mask = F<mshadow_op::threshold>(prnd->uniform(mask.shape_), pkeep_) * (1.0f / pkeep_);
      Assign(out, req[cdropout::kOut], data * mask);
    } else {
      Assign(out, req[cdropout::kOut], F<mshadow_op::identity>(data));
    }
  }

  virtual void Backward(const OpContext &ctx,
                        const std::vector<TBlob> &out_grad,
                        const std::vector<TBlob> &in_data,
                        const std::vector<TBlob> &out_data,
                        const std::vector<OpReqType> &req,
                        const std::vector<TBlob> &in_grad,
                        const std::vector<TBlob> &aux_states) {
    using namespace mshadow;
    using namespace mshadow::expr;
    CHECK_EQ(out_grad.size(), 1);
    CHECK_EQ(in_grad.size(), 1);
    Stream<xpu> *s = ctx.get_stream<xpu>();
    Tensor<xpu, 2> grad = out_grad[cdropout::kOut].FlatTo2D<xpu, real_t>(s);
    Tensor<xpu, 2> mask = out_data[cdropout::kMask].FlatTo2D<xpu, real_t>(s);
    Tensor<xpu, 2> gdata = in_grad[cdropout::kData].FlatTo2D<xpu, real_t>(s);
    Assign(gdata, req[cdropout::kData], grad * mask);
    //Assign(gdata, req[cdropout::kData], grad);
  }

 private:
  real_t pkeep_;
};  // class CDropoutOp


template<typename xpu>
Operator *CreateOp(CDropoutParam param);

#if DMLC_USE_CXX11
class CDropoutProp : public OperatorProperty {
 public:
  void Init(const std::vector<std::pair<std::string, std::string> >& kwargs) override {
    param_.Init(kwargs);
  }

  std::map<std::string, std::string> GetParams() const override {
    return param_.__DICT__();
  }

  bool InferShape(std::vector<TShape> *in_shape,
                  std::vector<TShape> *out_shape,
                  std::vector<TShape> *aux_shape) const override {
    using namespace mshadow;
    CHECK_EQ(in_shape->size(), 1);
    const TShape &dshape = in_shape->at(0);
    if (dshape.ndim() == 0) return false;
    out_shape->clear();
    out_shape->push_back(dshape);
    out_shape->push_back(dshape);
    return true;
  }

  OperatorProperty* Copy() const override {
    auto ptr = new CDropoutProp();
    ptr->param_ = param_;
    return ptr;
  }

  std::string TypeString() const override {
    return "CDropout";
  }

  std::vector<int> DeclareBackwardDependency(
    const std::vector<int> &out_grad,
    const std::vector<int> &in_data,
    const std::vector<int> &out_data) const override {
    return {out_grad[cdropout::kOut], out_data[cdropout::kMask]};
  }

  std::vector<std::pair<int, void*> > BackwardInplaceOption(
    const std::vector<int> &out_grad,
    const std::vector<int> &in_data,
    const std::vector<int> &out_data,
    const std::vector<void*> &in_grad) const override {
    return {{out_grad[cdropout::kOut], in_grad[cdropout::kData]}};
  }

  std::vector<std::pair<int, void*> > ForwardInplaceOption(
    const std::vector<int> &in_data,
    const std::vector<void*> &out_data) const override {
    return {{in_data[cdropout::kData], out_data[cdropout::kOut]}};
  }

  std::vector<ResourceRequest> ForwardResource(
    const std::vector<TShape> &in_shape) const override {
    return {ResourceRequest::kRandom, ResourceRequest::kTempSpace};
  }

  int NumVisibleOutputs() const override {
    return 1;
  }

  int NumOutputs() const override {
    return 2;
  }

  std::vector<std::string> ListOutputs() const override {
    return {"output", "mask"};
  }

  Operator* CreateOperator(Context ctx) const override;

 private:
  CDropoutParam param_;
};  // class CDropoutProp
#endif  // DMLC_USE_CXX11
}  // namespace op
}  // namespace mxnet
#endif  // MXNET_OPERATOR_DROPOUT_INL_H_

