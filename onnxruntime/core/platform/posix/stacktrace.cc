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
  constexpr size_t frames_to_skip = 1;

  // we generally want to skip the first frame, but if something weird is going on (e.g. code coverage is
  // running) and we only have 1 frame, output it so there's at least something that may be meaningful
  const size_t start_frame = size > frames_to_skip ? frames_to_skip : 0;

  for (size_t i = start_frame; i < size; i++) {
    stack.push_back(strings[i]);
  }

  free(strings);

  return stack;
}
}  // namespace onnxruntime
