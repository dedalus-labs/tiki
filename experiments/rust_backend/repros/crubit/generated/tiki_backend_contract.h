// Automatically @generated C++ bindings for the following Rust crate:
// tiki_backend_contract
// Features: <none>

#pragma once

#pragma clang diagnostic push
#pragma clang diagnostic ignored "-Wreturn-type-c-linkage"
#pragma clang diagnostic ignored "-Wunused-private-field"
#pragma clang diagnostic ignored "-Wdeprecated-declarations"
#pragma clang diagnostic ignored "-Wignored-attributes"
#include<bit>
#include<cstddef>
#include<cstdint>
#include<cstring>
#include<utility>
#include<support/annotations_internal.h>
#include<support/internal/check.h>
#include<support/internal/memswap.h>
#include<support/internal/slot.h>
#include<support/movable.h>
#include<support/rs_std/vec.h>



namespace tiki_backend_contract{ 
struct Buffer;

// Generated from: src/lib.rs;l=36
[[nodiscard]]::std::uintptr_t consume(::rs::Movable<::tiki_backend_contract::Buffer>buffer);

 }


// clang-format off
#ifndef _CRUBIT_BINDINGS_FOR_rs_ustd_x00000020_x0000003a_x0000003a_x00000020Vec_x00000020_x0000003c_x00000020float_x00000020_x0000003e
#define _CRUBIT_BINDINGS_FOR_rs_ustd_x00000020_x0000003a_x0000003a_x00000020Vec_x00000020_x0000003c_x00000020float_x00000020_x0000003e
template<>
struct alignas(8)CRUBIT_INTERNAL_RUST_TYPE(":: alloc :: vec :: Vec < f32 >")rs_std::Vec<float>{ 
public: 
// Default::default
Vec();


// Clone::clone
Vec(const Vec&);

// Clone::clone_from
rs_std::Vec<float>&operator=(const Vec&);

Vec(Vec&&);
rs_std::Vec<float>&operator=(Vec&&);
Vec(::crubit::UnsafeRelocateTag,Vec&&value);

~Vec()noexcept;
float*data()noexcept;float const*data()const noexcept;std::size_t size()const noexcept;float&operator[](std::size_t index)noexcept;float const&operator[](std::size_t index)const noexcept;float*begin()noexcept;float const*begin()const noexcept;float*end()noexcept;float const*end()const noexcept;
private: unsigned char storage_[24];
 };
#endif




namespace tiki_backend_contract{ 

// Generated from: src/lib.rs;l=5
struct CRUBIT_INTERNAL_RUST_TYPE(":: tiki_backend_contract :: Buffer")alignas(8)[[clang::trivial_abi]]Buffer final{ public: 

// `tiki_backend_contract::Buffer` doesn't implement the `Default` trait
Buffer()=delete;

// Drop::drop
~Buffer();


// C++ move operations are unavailable for this type. See http://crubit.rs/rust/movable_types for an explanation of Rust types that are C++ movable.
Buffer(Buffer&&)=delete;
::tiki_backend_contract::Buffer&operator=(Buffer&&)=delete;
// `tiki_backend_contract::Buffer` doesn't implement the `Clone` trait
Buffer(const Buffer&)=delete;
Buffer&operator=(const Buffer&)=delete;Buffer(::crubit::UnsafeRelocateTag,Buffer&&value);


// Generated from: src/lib.rs;l=11
[[nodiscard]]static::tiki_backend_contract::Buffer new_(::std::uintptr_t size);


// Generated from: src/lib.rs;l=16
[[nodiscard]]::std::uintptr_t len()const;


// Generated from: src/lib.rs;l=21
[[nodiscard]]bool is_empty()const;


// Generated from: src/lib.rs;l=25
void fill(float value);


// Generated from: src/lib.rs;l=30
[[nodiscard]]float checksum()const;
private: 
union{ 
// Generated from: src/lib.rs;l=6
rs_std::Vec<float>values; };private: static void __crubit_field_offset_assertions(); };

static_assert(sizeof(Buffer)==24,"Verify that ADT layout didn't change since this header got generated");static_assert(alignof(Buffer)==8,"Verify that ADT layout didn't change since this header got generated");
namespace __crubit_internal{ extern "C" void __crubit_thunk_8f7d455e8f2f14fc__uRNvYNtCscjN0fHCQrTs_u21tiki_ubackend_ucontract6BufferNtNtNtCs4aqKKZZmviT_u4core3ops4drop4Drop4dropCsg0yVvhnDUop_u8rust_uout(::tiki_backend_contract::Buffer&); }inline Buffer::~Buffer(){ __crubit_internal::__crubit_thunk_8f7d455e8f2f14fc__uRNvYNtCscjN0fHCQrTs_u21tiki_ubackend_ucontract6BufferNtNtNtCs4aqKKZZmviT_u4core3ops4drop4Drop4dropCsg0yVvhnDUop_u8rust_uout(*this); }inline::tiki_backend_contract::Buffer::Buffer(::crubit::UnsafeRelocateTag,Buffer&&value){ ::std::memcpy(this,&value,sizeof(value)); }

namespace __crubit_internal{ extern "C" void __crubit_thunk_8f7d455e8f2f14fc__uRNvMCscjN0fHCQrTs_u21tiki_ubackend_ucontractNtB2_u6Buffer3new(::std::uintptr_t,::tiki_backend_contract::Buffer*__ret_ptr); }
inline::tiki_backend_contract::Buffer Buffer::new_(::std::uintptr_t size){ crubit::Slot<::tiki_backend_contract::Buffer>__return_value_ret_val_holder;auto*__return_value_storage=__return_value_ret_val_holder.Get();__crubit_internal::__crubit_thunk_8f7d455e8f2f14fc__uRNvMCscjN0fHCQrTs_u21tiki_ubackend_ucontractNtB2_u6Buffer3new(size,__return_value_storage);return::std::move(__return_value_ret_val_holder).AssumeInitAndTakeValue(); }

namespace __crubit_internal{ extern "C"::std::uintptr_t __crubit_thunk_8f7d455e8f2f14fc__uRNvMCscjN0fHCQrTs_u21tiki_ubackend_ucontractNtB2_u6Buffer3len(::tiki_backend_contract::Buffer const&); }
inline::std::uintptr_t Buffer::len()const{ auto&&self=*this;return __crubit_internal::__crubit_thunk_8f7d455e8f2f14fc__uRNvMCscjN0fHCQrTs_u21tiki_ubackend_ucontractNtB2_u6Buffer3len(self); }

namespace __crubit_internal{ extern "C" bool __crubit_thunk_8f7d455e8f2f14fc__uRNvMCscjN0fHCQrTs_u21tiki_ubackend_ucontractNtB2_u6Buffer8is_uempty(::tiki_backend_contract::Buffer const&); }
inline bool Buffer::is_empty()const{ auto&&self=*this;return __crubit_internal::__crubit_thunk_8f7d455e8f2f14fc__uRNvMCscjN0fHCQrTs_u21tiki_ubackend_ucontractNtB2_u6Buffer8is_uempty(self); }

namespace __crubit_internal{ extern "C" void __crubit_thunk_8f7d455e8f2f14fc__uRNvMCscjN0fHCQrTs_u21tiki_ubackend_ucontractNtB2_u6Buffer4fill(::tiki_backend_contract::Buffer&,float); }
inline void Buffer::fill(float value){ auto&&self=*this;return __crubit_internal::__crubit_thunk_8f7d455e8f2f14fc__uRNvMCscjN0fHCQrTs_u21tiki_ubackend_ucontractNtB2_u6Buffer4fill(self,value); }

namespace __crubit_internal{ extern "C" float __crubit_thunk_8f7d455e8f2f14fc__uRNvMCscjN0fHCQrTs_u21tiki_ubackend_ucontractNtB2_u6Buffer8checksum(::tiki_backend_contract::Buffer const&); }
inline float Buffer::checksum()const{ auto&&self=*this;return __crubit_internal::__crubit_thunk_8f7d455e8f2f14fc__uRNvMCscjN0fHCQrTs_u21tiki_ubackend_ucontractNtB2_u6Buffer8checksum(self); }
inline void Buffer::__crubit_field_offset_assertions(){ static_assert(0==offsetof(Buffer,values)); }
namespace __crubit_internal{ extern "C"::std::uintptr_t __crubit_thunk_8f7d455e8f2f14fc__uRNvCscjN0fHCQrTs_u21tiki_ubackend_ucontract7consume(::tiki_backend_contract::Buffer*); }
inline::std::uintptr_t consume(::rs::Movable<::tiki_backend_contract::Buffer>buffer){ crubit::Slot<::tiki_backend_contract::Buffer>buffer_slot;::std::move(buffer).MoveToSlot(buffer_slot);return __crubit_internal::__crubit_thunk_8f7d455e8f2f14fc__uRNvCscjN0fHCQrTs_u21tiki_ubackend_ucontract7consume(buffer_slot.Get()); }

 }

#ifndef _CRUBIT_BINDINGS_FOR_IMPL_rs_ustd_x00000020_x0000003a_x0000003a_x00000020Vec_x00000020_x0000003c_x00000020float_x00000020_x0000003e
#define _CRUBIT_BINDINGS_FOR_IMPL_rs_ustd_x00000020_x0000003a_x0000003a_x00000020Vec_x00000020_x0000003c_x00000020float_x00000020_x0000003e
namespace __crubit_internal{ extern "C" void __crubit_thunk_8f7d455e8f2f14fc__uRNvYINtNtCscU4bHatjcek_u5alloc3vec3VecfENtNtCs4aqKKZZmviT_u4core7default7Default7defaultCsg0yVvhnDUop_u8rust_uout(rs_std::Vec<float>*__ret_ptr); }inline rs_std::Vec<float>::Vec(){ __crubit_internal::__crubit_thunk_8f7d455e8f2f14fc__uRNvYINtNtCscU4bHatjcek_u5alloc3vec3VecfENtNtCs4aqKKZZmviT_u4core7default7Default7defaultCsg0yVvhnDUop_u8rust_uout(this); }namespace __crubit_internal{ extern "C" void __crubit_thunk_8f7d455e8f2f14fc__uRNvYINtNtCscU4bHatjcek_u5alloc3vec3VecfENtNtCs4aqKKZZmviT_u4core5clone5Clone5cloneCsg0yVvhnDUop_u8rust_uout(rs_std::Vec<float>const&,rs_std::Vec<float>*__ret_ptr); }namespace __crubit_internal{ extern "C" void __crubit_thunk_8f7d455e8f2f14fc__uRNvYINtNtCscU4bHatjcek_u5alloc3vec3VecfENtNtCs4aqKKZZmviT_u4core5clone5Clone10clone_ufromCsg0yVvhnDUop_u8rust_uout(rs_std::Vec<float>&,rs_std::Vec<float>const&); }inline rs_std::Vec<float>::Vec(const Vec&other){ __crubit_internal::__crubit_thunk_8f7d455e8f2f14fc__uRNvYINtNtCscU4bHatjcek_u5alloc3vec3VecfENtNtCs4aqKKZZmviT_u4core5clone5Clone5cloneCsg0yVvhnDUop_u8rust_uout(other,this); }inline rs_std::Vec<float>&rs_std::Vec<float>::operator=(const Vec&other){ if(this!=&other){ __crubit_internal::__crubit_thunk_8f7d455e8f2f14fc__uRNvYINtNtCscU4bHatjcek_u5alloc3vec3VecfENtNtCs4aqKKZZmviT_u4core5clone5Clone10clone_ufromCsg0yVvhnDUop_u8rust_uout(*this,other); }return*this; }inline rs_std::Vec<float>::Vec(Vec&&other): Vec(){ *this=::std::move(other); }inline rs_std::Vec<float>&rs_std::Vec<float>::operator=(Vec&&other){ crubit::MemSwap(*this,other);return*this; }inline rs_std::Vec<float>::Vec(::crubit::UnsafeRelocateTag,Vec&&value){ ::std::memcpy(this,&value,sizeof(value)); }

extern "C" void __crubit_thunk_8f7d455e8f2f14fc__uRNvYINtNtCscU4bHatjcek_u5alloc3vec3VecfENtNtNtCs4aqKKZZmviT_u4core3ops4drop4Drop4dropCsg0yVvhnDUop_u8rust_uout(void*vec)noexcept;inline rs_std::Vec<float>::~Vec()noexcept{ __crubit_thunk_8f7d455e8f2f14fc__uRNvYINtNtCscU4bHatjcek_u5alloc3vec3VecfENtNtNtCs4aqKKZZmviT_u4core3ops4drop4Drop4dropCsg0yVvhnDUop_u8rust_uout(this); }
inline float*rs_std::Vec<float>::data()noexcept{ return std::bit_cast<float*>(*reinterpret_cast<const std::uintptr_t*>(&storage_[8])); }inline float const*rs_std::Vec<float>::data()const noexcept{ return std::bit_cast<float*>(*reinterpret_cast<const std::uintptr_t*>(&storage_[8])); }inline std::size_t rs_std::Vec<float>::size()const noexcept{ return std::bit_cast<std::size_t>(*reinterpret_cast<const std::size_t*>(&storage_[16])); }inline float&rs_std::Vec<float>::operator[](std::size_t index)noexcept{ CRUBIT_CHECK(index<size());return data()[index]; }inline float const&rs_std::Vec<float>::operator[](std::size_t index)const noexcept{ CRUBIT_CHECK(index<size());return data()[index]; }inline float*rs_std::Vec<float>::begin()noexcept{ return data(); }inline float const*rs_std::Vec<float>::begin()const noexcept{ return data(); }inline float*rs_std::Vec<float>::end()noexcept{ return data()+size(); }inline float const*rs_std::Vec<float>::end()const noexcept{ return data()+size(); }
#endif


#pragma clang diagnostic pop
