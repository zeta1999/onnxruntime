// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#pragma once

#include "core/optimizer/graph_transformer.h"
#include "core/framework/ml_value.h"

namespace onnxruntime {

class CPUExecutionProvider;

/**
@class ConstantFolding

Transformer that traverses the graph top-down and performs constant folding, i.e.,
it statically computes parts of the graph that rely only on constant initializers.
*/
class ConstantFolding : public GraphTransformer {
 public:
  ConstantFolding(const std::unordered_set<std::string>& compatible_execution_providers = {}) noexcept;
  ~ConstantFolding();

 private:
  /** Constant folding will not be applied to nodes whose op_type is included in this set.
      All non-deterministic operators should be included in this set. */
  const std::unordered_set<std::string> excluded_op_types_ =
      {"RandomUniform", "RandomNormal", "RandomUniformLike", "RandomNormalLike", "Multinomial"};

  Status ApplyImpl(Graph& graph, bool& modified, int graph_level, const logging::Logger& logger) const override;

  // TODO: Ideally the EPs from the session would be provided here so we could use any of them. For now we only
  // utilize the CPU EP for constant folding so we create a temporary local one.
  std::unique_ptr<CPUExecutionProvider> cpu_execution_provider_;
};

}  // namespace onnxruntime
