// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#include "core/session/onnxruntime_cxx_api.h"
#include "core/optimizer/graph_transformer_level.h"
#include "test_fixture.h"
using namespace onnxruntime;

TEST_F(CApiTest, session_options_graph_optimization_level) {
  // Test set optimization level succeeds when valid level is provided.
  Ort::SessionOptions options;
  options.SetGraphOptimizationLevel(ORT_ENABLE_EXTENDED);
}

#include "onnxruntime_c_api.h"
#include "core/providers/cuda/cuda_provider_factory.h"

TEST_F(CApiTest, TempModelTest) {
  const OrtApi* g_ort = OrtGetApiBase()->GetApi(ORT_API_VERSION);
  auto CheckStatus = [&](OrtStatus* status) {
    if (status != NULL) {
      const char* msg = g_ort->GetErrorMessage(status);
      fprintf(stderr, "%s\n", msg);
      g_ort->ReleaseStatus(status);
      ASSERT_EQ(status, nullptr);
    }
  };

  OrtEnv* env = NULL;
  CheckStatus(g_ort->CreateEnv(ORT_LOGGING_LEVEL_WARNING, "test", &env));
  // initialize session options if needed
  OrtSessionOptions* session_options;
  g_ort->CreateSessionOptions(&session_options);
  // If you have CUDA ONNXRuntime installed otherwise don't use this line
  OrtSessionOptionsAppendExecutionProvider_CUDA(session_options, 0);
  g_ort->SetIntraOpNumThreads(session_options, 8);
  g_ort->SetInterOpNumThreads(session_options, 8);
  g_ort->SetSessionGraphOptimizationLevel(session_options, ORT_DISABLE_ALL);
  g_ort->SetOptimizedModelFilePath(session_options, L"d:/temp/hrnet_w18_landmarks.optimized.onnx");
  OrtSession* session = NULL;
  CheckStatus(g_ort->CreateSession(env, L"d:/temp/hrnet_w18_landmarks.onnx", session_options, &session));
}
