// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#include "core/framework/data_types.h"
#include "core/framework/execution_providers.h"
#include "core/framework/op_kernel.h"
#include "core/framework/session_state.h"
#include "core/graph/graph_viewer.h"
#include "core/graph/model.h"
#include "core/graph/op.h"
#include "core/providers/cpu/cpu_execution_provider.h"
#include "core/session/inference_session.h"
#include "gtest/gtest.h"
#include "test/providers/provider_test_utils.h"
#include "test/framework/test_utils.h"

#include "core/automl/featurizers/src/FeaturizerPrep/Featurizers/DateTimeFeaturizer.h"
namespace dtf = Microsoft::Featurizer::DateTimeFeaturizer;

using namespace ONNX_NAMESPACE;
using namespace onnxruntime::common;

//#define SAVE_MODEL 1

namespace onnxruntime {
namespace test {

//template<class T>
//void CreateCustomMLValue(OrtValue& ) {
//  auto element_type = DataTypeImpl::GetType<T>();
//std::unique_ptr<Tensor> p_tensor = std::make_unique<Tensor>(element_type,
//                                                            shape,
//                                                            alloc);
//if (value.size() > 0) {
//  CopyVectorToTensor(value, *p_tensor);
//}
//
//p_mlvalue->Init(p_tensor.release(),
//                DataTypeImpl::GetType<Tensor>(),
//                DataTypeImpl::GetType<Tensor>()->GetDeleteFunc());
//}

template<class T>
inline uint32_t Cast32(T v) {
  return static_cast<uint32_t>(v);
}


TEST(AutoMLModel, SaveModel) {
  SessionOptions so;
  so.enable_sequential_execution = true;
  so.session_logid = "AutoMLModel";
  so.session_log_verbosity_level = 1;

  InferenceSession session_object{so, &DefaultLoggingManager()};
  Model model("AutoMLModel", false);

  auto& graph = model.MainGraph();

  std::vector<onnxruntime::NodeArg*> inputs;
  std::vector<onnxruntime::NodeArg*> outputs;

  TypeProto input_tensor_proto(*DataTypeImpl::GetTensorType<int64_t>()->GetTypeProto());

  {
    TypeProto system_time_tensor(input_tensor_proto);
    // Find out the shape
    auto* mutable_system_time = system_time_tensor.mutable_tensor_type();
    mutable_system_time->set_elem_type(TensorProto_DataType_INT64);
    mutable_system_time->mutable_shape()->add_dim()->set_dim_value(1);
    auto& system_time_arg = graph.GetOrCreateNodeArg("From_TimeT", &system_time_tensor);
    inputs.push_back(&system_time_arg);

    //Output is our custom data type. This will return an Opaque type proto
    TypeProto output_dtf_timepoint(*DataTypeImpl::GetType<dtf::TimePoint>()->GetTypeProto());
    auto& time_point_output_arg = graph.GetOrCreateNodeArg("dtf_TimePoint", &output_dtf_timepoint);
    outputs.push_back(&time_point_output_arg);

    auto& node = graph.AddNode("DateTimeTransformer", "DateTimeTransformer", "Break time_t to dtf::TimePoint Components.",
                               inputs, outputs, nullptr, onnxruntime::kMSAutoMLDomain);
    node.SetExecutionProviderType(onnxruntime::kCpuExecutionProvider);
  }
  EXPECT_TRUE(graph.Resolve().IsOK());
  // Get a proto and load from it
  std::string serialized_model;
  auto model_proto = model.ToProto();
  EXPECT_TRUE(model_proto.SerializeToString(&serialized_model));

#ifdef SAVE_MODEL
  {
    std::ofstream os("d:\\Dev\\dft_mode.onnx", 
      std::ios_base::out | std::ios_base::binary | std::ios_base::trunc);
    os.write(serialized_model.data(), serialized_model.size());
    os.flush();
    EXPECT_FALSE(os.fail());
  }
#endif

  std::stringstream sstr(serialized_model);
  EXPECT_TRUE(session_object.Load(sstr).IsOK());
  EXPECT_TRUE(session_object.Initialize().IsOK());

  RunOptions run_options;

  // Prepare inputs
  // Create an input value
  const time_t date = 217081625;  // 1976_Nov_17__12_27_05
  std::vector<int64_t> val_dims = {1};
  std::vector<int64_t> values = {date};
  // prepare input
  // We use a utility function that would create a Tensor
  OrtValue ml_value;
  CreateMLValue<int64_t>(TestCPUExecutionProvider()->GetAllocator(0, OrtMemTypeDefault), val_dims, values, &ml_value);
  NameMLValMap feeds;
  feeds.insert(std::make_pair("From_TimeT", ml_value));

  // Prepare outputs
  // Output is a custom object
  std::vector<std::string> output_names;
  output_names.push_back("dtf_TimePoint");
  std::vector<OrtValue> fetches;

  EXPECT_TRUE(session_object.Run(run_options, feeds, output_names, &fetches).IsOK());
  ASSERT_EQ(1U, fetches.size());
  auto& tp = fetches.front().Get<dtf::TimePoint>();

  std::cout << Cast32(tp.month) << '/' << Cast32(tp.day) << '/' << tp.year 
    << ' ' << Cast32(tp.hour) << ':' << Cast32(tp.minute) << ':' << Cast32(tp.second)
    << std::endl;
}

}  // namespace test
}  // namespace onnxruntime