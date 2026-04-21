#define __NV_MODULE_ID _bc1937e1_11_qkv_proj_cu_93625e60
#define __NV_CUBIN_HANDLE_STORAGE__ extern
#if !defined(__CUDA_INCLUDE_COMPILER_INTERNAL_HEADERS__)
#define __CUDA_INCLUDE_COMPILER_INTERNAL_HEADERS__
#endif
#include "crt/host_runtime.h"
#include "qkv_proj.fatbin.c"
static void __device_stub__ZN4tllm7kernels44_GLOBAL__N__bc1937e1_11_qkv_proj_cu_93625e6021qkv_proj_kernel_tunedILi256EEEvP6__halfPKaPKS3_S8_ii(struct __half *__restrict__, const int8_t *__restrict__, const struct __half *__restrict__, const struct __half *__restrict__, int, int);
static void __nv_cudaEntityRegisterCallback(void **);
static void __sti____cudaRegisterAll(void);
#pragma section(".CRT$XCT",read)
__declspec(allocate(".CRT$XCT"))static void (*__dummy_static_init__sti____cudaRegisterAll[])(void) = {__sti____cudaRegisterAll};
static void __device_stub__ZN4tllm7kernels44_GLOBAL__N__bc1937e1_11_qkv_proj_cu_93625e6021qkv_proj_kernel_tunedILi256EEEvP6__halfPKaPKS3_S8_ii(
struct __half *__restrict__ __par0, 
const int8_t *__restrict__ __par1, 
const struct __half *__restrict__ __par2, 
const struct __half *__restrict__ __par3, 
int __par4, 
int __par5)
{
__cudaLaunchPrologue(6);
__cudaSetupArgSimple(__par0, 0Ui64);
__cudaSetupArgSimple(__par1, 8Ui64);
__cudaSetupArgSimple(__par2, 16Ui64);
__cudaSetupArgSimple(__par3, 24Ui64);
__cudaSetupArgSimple(__par4, 32Ui64);
__cudaSetupArgSimple(__par5, 36Ui64);
__cudaLaunch(((char *)((void ( *)(struct __half *__restrict__, const int8_t *__restrict__, const struct __half *__restrict__, const struct __half *__restrict__, int, int))tllm::kernels::_NV_ANON_NAMESPACE::qkv_proj_kernel_tuned<(int)256> )));
}namespace tllm{
namespace kernels{
namespace _NV_ANON_NAMESPACE{

template<> __specialization_static void __wrapper__device_stub_qkv_proj_kernel_tuned<256>( struct ::__half *__restrict__ &__cuda_0,const ::int8_t *__restrict__ &__cuda_1,const struct ::__half *__restrict__ &__cuda_2,const struct ::__half *__restrict__ &__cuda_3,int &__cuda_4,int &__cuda_5){__device_stub__ZN4tllm7kernels44_GLOBAL__N__bc1937e1_11_qkv_proj_cu_93625e6021qkv_proj_kernel_tunedILi256EEEvP6__halfPKaPKS3_S8_ii( (struct ::__half *&)__cuda_0,(const ::int8_t *&)__cuda_1,(const struct ::__half *&)__cuda_2,(const struct ::__half *&)__cuda_3,(int &)__cuda_4,(int &)__cuda_5);}}}}
static void __nv_cudaEntityRegisterCallback(
void **__T18)
{
__nv_dummy_param_ref(__T18);
__nv_save_fatbinhandle_for_managed_rt(__T18);
__cudaRegisterEntry(__T18, ((void ( *)(struct __half *__restrict__, const int8_t *__restrict__, const struct __half *__restrict__, const struct __half *__restrict__, int, int))tllm::kernels::_NV_ANON_NAMESPACE::qkv_proj_kernel_tuned<(int)256> ), _ZN4tllm7kernels44_GLOBAL__N__bc1937e1_11_qkv_proj_cu_93625e6021qkv_proj_kernel_tunedILi256EEEvP6__halfPKaPKS3_S8_ii, (-1));
}
static void __sti____cudaRegisterAll(void)
{
____cudaRegisterLinkedBinary(__nv_cudaEntityRegisterCallback);
}
