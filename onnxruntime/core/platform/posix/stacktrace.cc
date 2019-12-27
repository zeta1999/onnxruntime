// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#include "core/common/common.h"

#include <execinfo.h>
#include <vector>

namespace onnxruntime {

std::vector<std::string> GetStackTrace() {
  constexpr int kCallstackLimit = 64;  // Maximum depth of callstack

  void *array[kCallstackLimit];
  char **strings = nullptr;

  size_t size = backtrace (array, kCallstackLimit);
  strings = backtrace_symbols (array, size);

  std::vector<std::string> stack;
  stack.reserve(size);

  // hide GetStackTrace so the output starts with the 'real' location
  constexpr size_t start_frame = 1;

  for (size_t i = start_frame; i < size; i++) {
    stack.push_back(strings[i]);
  }

  free(strings);

  return stack;
}
}  // namespace onnxruntime
