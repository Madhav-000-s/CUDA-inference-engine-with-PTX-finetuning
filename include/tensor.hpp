#pragma once
#include <cuda_fp16.h>
#include <cuda_runtime.h>

#include <cstdint>
#include <cstdio>
#include <cstdlib>

#define CUDA_CHECK(expr)                                                             \
    do {                                                                             \
        cudaError_t _err = (expr);                                                   \
        if (_err != cudaSuccess) {                                                   \
            fprintf(stderr, "CUDA error %s at %s:%d: %s\n", cudaGetErrorString(_err),\
                    __FILE__, __LINE__, #expr);                                      \
            std::exit(1);                                                            \
        }                                                                            \
    } while (0)

namespace tllm {

// Owning device buffer. Allocates in ctor, frees in dtor. No copy, moveable.
template <typename T>
class DeviceBuffer {
   public:
    DeviceBuffer() = default;
    explicit DeviceBuffer(size_t n) { resize(n); }
    DeviceBuffer(const DeviceBuffer&) = delete;
    DeviceBuffer& operator=(const DeviceBuffer&) = delete;
    DeviceBuffer(DeviceBuffer&& o) noexcept : ptr_(o.ptr_), n_(o.n_) {
        o.ptr_ = nullptr; o.n_ = 0;
    }
    DeviceBuffer& operator=(DeviceBuffer&& o) noexcept {
        if (this != &o) { free_(); ptr_ = o.ptr_; n_ = o.n_; o.ptr_ = nullptr; o.n_ = 0; }
        return *this;
    }
    ~DeviceBuffer() { free_(); }

    void resize(size_t n) {
        free_();
        n_ = n;
        if (n) CUDA_CHECK(cudaMalloc(&ptr_, n * sizeof(T)));
    }
    void copy_from_host(const void* src, size_t n_bytes) {
        CUDA_CHECK(cudaMemcpy(ptr_, src, n_bytes, cudaMemcpyHostToDevice));
    }
    void copy_to_host(void* dst, size_t n_bytes) const {
        CUDA_CHECK(cudaMemcpy(dst, ptr_, n_bytes, cudaMemcpyDeviceToHost));
    }

    T*       data()       { return ptr_; }
    const T* data() const { return ptr_; }
    size_t   size() const { return n_; }

   private:
    void free_() { if (ptr_) { cudaFree(ptr_); ptr_ = nullptr; } }
    T*     ptr_ = nullptr;
    size_t n_   = 0;
};

using half_t = __half;

}  // namespace tllm
