// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#pragma once

#include "core/framework/allocator.h"

namespace onnxruntime {

class CUDAAllocator : public IDeviceAllocator {
 public:
  CUDAAllocator(int device_id, const char* name, bool use_arena = true)
      : info_(name, OrtAllocatorType::OrtDeviceAllocator,
              OrtDevice(OrtDevice::GPU, OrtDevice::MemType::DEFAULT, device_id),
              device_id, OrtMemTypeDefault),
        use_arena_(use_arena) {}

  void* Alloc(size_t size) override;
  void Free(void* p) override;
  const OrtMemoryInfo& Info() const override;
  FencePtr CreateFence(const SessionState* session_state) override;
  bool AllowsArena() const override { return use_arena_; }

 private:
  void CheckDevice(bool throw_when_fail) const;

 private:
  const OrtMemoryInfo info_;
  bool use_arena_;
};

//TODO: add a default constructor
class CUDAPinnedAllocator : public IDeviceAllocator {
 public:
  CUDAPinnedAllocator(int device_id, const char* name, bool use_arena = true)
      : info_(name, OrtAllocatorType::OrtDeviceAllocator,
              OrtDevice(OrtDevice::CPU, OrtDevice::MemType::CUDA_PINNED, device_id),
              device_id, OrtMemTypeCPUOutput),
        use_arena_(use_arena) {}

  void* Alloc(size_t size) override;
  void Free(void* p) override;
  const OrtMemoryInfo& Info() const override;
  FencePtr CreateFence(const SessionState* session_state) override;
  bool AllowsArena() const override { return use_arena_; }

 private:
  const OrtMemoryInfo info_;
  bool use_arena_;
};

}  // namespace onnxruntime
