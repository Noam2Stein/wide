#[cfg(all(target_feature = "neon", target_arch = "aarch64"))]
use core::arch::aarch64::*;
#[cfg(target_feature = "simd128")]
use core::arch::wasm32::*;

use super::*;

use crate::{i32x4, simd::SimdBackend, u64x4};

#[cfg(not(any(
  target_feature = "sse2",
  target_feature = "simd128",
  all(target_feature = "neon", target_arch = "aarch64"),
)))]
#[repr(C, align(16))]
#[derive(Default, Clone, Copy, PartialEq, Eq)]
pub(crate) struct Inner(pub [u32; 4]);

unsafe impl SimdBackend for u32x4 {
  pick! {
    if #[cfg(target_feature="sse2")] {
      type Inner = m128i;
    } else if #[cfg(target_feature="simd128")] {
      type Inner = v128;
    } else if #[cfg(all(target_feature="neon",target_arch="aarch64"))]{
      type Inner = uint32x4_t;
    } else {
      type Inner = Inner;
    }
  }

  #[inline]
  fn simd_eq(self, rhs: Self) -> Self {
    pick! {
      if #[cfg(target_feature="sse2")] {
        Self(cmp_eq_mask_i32_m128i(self.0, rhs.0))
      } else if #[cfg(target_feature="simd128")] {
        Self(u32x4_eq(self.0, rhs.0))
      } else if #[cfg(all(target_feature="neon",target_arch="aarch64"))]{
        unsafe { Self(vceqq_u32(self.0, rhs.0)) }
      } else {
        Self(Inner([
          if self.0.0[0] == rhs.0.0[0] { u32::MAX } else { 0 },
          if self.0.0[1] == rhs.0.0[1] { u32::MAX } else { 0 },
          if self.0.0[2] == rhs.0.0[2] { u32::MAX } else { 0 },
          if self.0.0[3] == rhs.0.0[3] { u32::MAX } else { 0 },
        ]))
      }
    }
  }

  #[inline]
  fn simd_ne(self, rhs: Self) -> Self {
    pick! {
      if #[cfg(target_feature="sse2")] {
        !self.simd_eq(rhs)
      } else if #[cfg(target_feature="simd128")] {
        Self(u32x4_ne(self.0, rhs.0))
      } else if #[cfg(all(target_feature="neon",target_arch="aarch64"))]{
        !self.simd_eq(rhs)
      } else {
        Self(Inner([
          if self.0.0[0] != rhs.0.0[0] { u32::MAX } else { 0 },
          if self.0.0[1] != rhs.0.0[1] { u32::MAX } else { 0 },
          if self.0.0[2] != rhs.0.0[2] { u32::MAX } else { 0 },
          if self.0.0[3] != rhs.0.0[3] { u32::MAX } else { 0 },
        ]))
      }
    }
  }

  #[inline]
  fn simd_lt(self, rhs: Self) -> Self {
    // lt is just gt the other way around
    rhs.simd_gt(self)
  }

  #[inline]
  fn simd_gt(self, rhs: Self) -> Self {
    pick! {
      if #[cfg(target_feature="sse2")] {
        // no unsigned less than so inverting the high bit will get the correct result
        let h = u32x4::splat(1 << 31);
        Self(cmp_gt_mask_i32_m128i((self ^ h).0, (rhs ^ h).0))
      } else if #[cfg(target_feature="simd128")] {
        Self(u32x4_gt(self.0, rhs.0))
      } else if #[cfg(all(target_feature="neon",target_arch="aarch64"))] {
        unsafe { Self(vcgtq_u32(self.0, rhs.0)) }
      } else {
        Self(Inner([
          if self.0.0[0] > rhs.0.0[0] { u32::MAX } else { 0 },
          if self.0.0[1] > rhs.0.0[1] { u32::MAX } else { 0 },
          if self.0.0[2] > rhs.0.0[2] { u32::MAX } else { 0 },
          if self.0.0[3] > rhs.0.0[3] { u32::MAX } else { 0 },
        ]))
      }
    }
  }

  #[inline]
  fn simd_le(self, rhs: Self) -> Self {
    pick! {
      if #[cfg(target_feature="sse2")] {
        !self.simd_gt(rhs)
      } else if #[cfg(target_feature="simd128")] {
        Self(u32x4_le(self.0, rhs.0))
      } else if #[cfg(all(target_feature="neon",target_arch="aarch64"))]{
        !self.simd_gt(rhs)
      } else {
        Self(Inner([
          if self.0.0[0] <= rhs.0.0[0] { u32::MAX } else { 0 },
          if self.0.0[1] <= rhs.0.0[1] { u32::MAX } else { 0 },
          if self.0.0[2] <= rhs.0.0[2] { u32::MAX } else { 0 },
          if self.0.0[3] <= rhs.0.0[3] { u32::MAX } else { 0 },
        ]))
      }
    }
  }

  #[inline]
  fn simd_ge(self, rhs: Self) -> Self {
    pick! {
      if #[cfg(target_feature="sse2")] {
        !self.simd_lt(rhs)
      } else if #[cfg(target_feature="simd128")] {
        Self(u32x4_ge(self.0, rhs.0))
      } else if #[cfg(all(target_feature="neon",target_arch="aarch64"))]{
        !self.simd_lt(rhs)
      } else {
        Self(Inner([
          if self.0.0[0] >= rhs.0.0[0] { u32::MAX } else { 0 },
          if self.0.0[1] >= rhs.0.0[1] { u32::MAX } else { 0 },
          if self.0.0[2] >= rhs.0.0[2] { u32::MAX } else { 0 },
          if self.0.0[3] >= rhs.0.0[3] { u32::MAX } else { 0 },
        ]))
      }
    }
  }

  #[inline]
  fn bitselect(self, if_one: Self, if_zero: Self) -> Self {
    pick! {
      if #[cfg(target_feature="sse2")] {
        Self(bitor_m128i(
          bitand_m128i(if_one.0, self.0),
          bitandnot_m128i(self.0, if_zero.0),
        ))
      } else if #[cfg(target_feature="simd128")] {
        Self(v128_bitselect(if_one.0, if_zero.0, self.0))
      } else if #[cfg(all(target_feature="neon",target_arch="aarch64"))]{
        unsafe { Self(vbslq_u32(self.0, if_one.0, if_zero.0)) }
      } else {
        generic_bit_blend(self, if_one, if_zero)
      }
    }
  }

  #[inline]
  fn select(self, if_true: Self, if_false: Self) -> Self {
    pick! {
      if #[cfg(target_feature="sse4.1")] {
        Self(blend_varying_i8_m128i(if_false.0, if_true.0, self.0))
      } else if #[cfg(target_feature="simd128")] {
        Self(v128_bitselect(if_true.0, if_false.0, self.0))
      } else if #[cfg(all(target_feature="neon",target_arch="aarch64"))]{
        unsafe { Self(vbslq_u32(self.0, if_true.0, if_false.0)) }
      } else {
        generic_bit_blend(self, if_true, if_false)
      }
    }
  }

  #[inline]
  fn to_bitmask(self) -> u32 {
    i32x4::to_bitmask(cast(self))
  }

  #[inline]
  fn any(self) -> bool {
    pick! {
      if #[cfg(target_feature="sse2")] {
        (move_mask_i8_m128i(self.0) & 0b1000100010001000) != 0
      } else if #[cfg(target_feature="simd128")] {
        u32x4_bitmask(self.0) != 0
      } else {
        let v : [u64;2] = cast(self);
        ((v[0] | v[1]) & 0x8000000080000000) != 0
      }
    }
  }

  #[inline]
  fn all(self) -> bool {
    pick! {
      if #[cfg(target_feature="sse2")] {
        (move_mask_i8_m128i(self.0) & 0b1000100010001000) == 0b1000100010001000
      } else if #[cfg(target_feature="simd128")] {
        u32x4_bitmask(self.0) == 0b1111
      } else {
        let v : [u64;2] = cast(self);
        (v[0] & v[1] & 0x8000000080000000) == 0x8000000080000000
      }
    }
  }

  #[inline]
  fn transpose(data: [u32x4; 4]) -> [u32x4; 4] {
    cast(i32x4::transpose(cast(data)))
  }
}

impl_simd_uint! {
  unsafe {
    T = u32,
    N = 4,
    Simd = u32x4,
    SignedSimd = i32x4,
    T_BITS = 32,
    T_BITS_MUL_2 = 64,
    [0, 1, 2, 3],
  }

  #[inline]
  fn not(self) -> Self::Output {
    self ^ cast::<u128, u32x4>(u128::MAX)
  }

  #[inline]
  fn add(self, rhs: Self) -> Self::Output {
    pick! {
      if #[cfg(target_feature="sse2")] {
        Self(add_i32_m128i(self.0, rhs.0))
      } else if #[cfg(target_feature="simd128")] {
        Self(u32x4_add(self.0, rhs.0))
      } else if #[cfg(all(target_feature="neon",target_arch="aarch64"))]{
        unsafe { Self(vaddq_u32(self.0, rhs.0)) }
      } else {
        Self(Inner([
          self.0.0[0].wrapping_add(rhs.0.0[0]),
          self.0.0[1].wrapping_add(rhs.0.0[1]),
          self.0.0[2].wrapping_add(rhs.0.0[2]),
          self.0.0[3].wrapping_add(rhs.0.0[3]),
        ]))
      }
    }
  }

  #[inline]
  fn sub(self, rhs: Self) -> Self::Output {
    pick! {
      if #[cfg(target_feature="sse2")] {
        Self(sub_i32_m128i(self.0, rhs.0))
      } else if #[cfg(target_feature="simd128")] {
        Self(u32x4_sub(self.0, rhs.0))
      } else if #[cfg(all(target_feature="neon",target_arch="aarch64"))]{
        unsafe { Self(vsubq_u32(self.0, rhs.0)) }
      } else {
        Self(Inner([
          self.0.0[0].wrapping_sub(rhs.0.0[0]),
          self.0.0[1].wrapping_sub(rhs.0.0[1]),
          self.0.0[2].wrapping_sub(rhs.0.0[2]),
          self.0.0[3].wrapping_sub(rhs.0.0[3]),
        ]))
      }
    }
  }

  #[inline]
  fn mul(self, rhs: Self) -> Self::Output {
    pick! {
      if #[cfg(target_feature="sse4.1")] {
        Self(mul_32_m128i(self.0, rhs.0))
      } else if #[cfg(target_feature="simd128")] {
        Self(u32x4_mul(self.0, rhs.0))
      } else if #[cfg(all(target_feature="neon",target_arch="aarch64"))]{
        unsafe { Self(vmulq_u32(self.0, rhs.0)) }
      } else {
        let arr1: [u32; 4] = cast(self);
        let arr2: [u32; 4] = cast(rhs);
        cast([
          arr1[0].wrapping_mul(arr2[0]),
          arr1[1].wrapping_mul(arr2[1]),
          arr1[2].wrapping_mul(arr2[2]),
          arr1[3].wrapping_mul(arr2[3]),
        ])
      }
    }
  }

  #[inline]
  fn shl(self, rhs: u32x4) -> Self::Output {
    pick! {
      if #[cfg(target_feature="avx2")] {
        // mask the shift count to 31 to have same behavior on all platforms
        let shift_by = bitand_m128i(rhs.0, set_splat_i32_m128i(31));
        Self(shl_each_u32_m128i(self.0, shift_by))
      } else if #[cfg(all(target_feature="neon",target_arch="aarch64"))]{
        unsafe {
          // mask the shift count to 31 to have same behavior on all platforms
          let shift_by = vreinterpretq_s32_u32(vandq_u32(rhs.0, vmovq_n_u32(31)));
          Self(vshlq_u32(self.0, shift_by))
        }
      } else {
        let arr: [u32; 4] = cast(self);
        let rhs: [u32; 4] = cast(rhs);
        cast([
          arr[0].wrapping_shl(rhs[0]),
          arr[1].wrapping_shl(rhs[1]),
          arr[2].wrapping_shl(rhs[2]),
          arr[3].wrapping_shl(rhs[3]),
        ])
      }
    }
  }

  #[inline]
  fn shl(self, rhs: u32) -> Self::Output {
    pick! {
      if #[cfg(target_feature="sse2")] {
        // Use `rhs % 32` to perform wrapping shift and not unbounded shift.
        #[expect(clippy::suspicious_arithmetic_impl)]
        let shift = cast([rhs as u64 & 31, 0]);
        Self(shl_all_u32_m128i(self.0, shift))
      } else if #[cfg(target_feature="simd128")] {
        Self(u32x4_shl(self.0, rhs))
      } else if #[cfg(all(target_feature="neon",target_arch="aarch64"))]{
        // Use `rhs % 32` to perform wrapping shift and not unbounded shift.
        #[expect(clippy::suspicious_arithmetic_impl)]
        unsafe { Self(vshlq_u32(self.0, vmovq_n_s32(rhs as i32 & 31))) }
      } else {
        Self(Inner([
          self.0.0[0].wrapping_shl(rhs),
          self.0.0[1].wrapping_shl(rhs),
          self.0.0[2].wrapping_shl(rhs),
          self.0.0[3].wrapping_shl(rhs),
        ]))
      }
    }
  }

  #[inline]
  fn shr(self, rhs: u32x4) -> Self::Output {
    pick! {
      if #[cfg(target_feature="avx2")] {
        // mask the shift count to 31 to have same behavior on all platforms
        let shift_by = bitand_m128i(rhs.0, set_splat_i32_m128i(31));
        Self(shr_each_u32_m128i(self.0, shift_by))
      } else if #[cfg(all(target_feature="neon",target_arch="aarch64"))]{
        unsafe {
          // mask the shift count to 31 to have same behavior on all platforms
          // no right shift, have to pass negative value to left shift on neon
          let shift_by = vnegq_s32(vreinterpretq_s32_u32(vandq_u32(rhs.0, vmovq_n_u32(31))));
          Self(vshlq_u32(self.0, shift_by))
        }
      } else {
        let arr: [u32; 4] = cast(self);
        let rhs: [u32; 4] = cast(rhs);
        cast([
          arr[0].wrapping_shr(rhs[0]),
          arr[1].wrapping_shr(rhs[1]),
          arr[2].wrapping_shr(rhs[2]),
          arr[3].wrapping_shr(rhs[3]),
        ])
      }
    }
  }

  #[inline]
  fn shr(self, rhs: u32) -> Self::Output {
    pick! {
      if #[cfg(target_feature="sse2")] {
        // Use `rhs % 32` to perform wrapping shift and not unbounded shift.
        #[expect(clippy::suspicious_arithmetic_impl)]
        let shift = cast([rhs as u64 & 31, 0]);
        Self(shr_all_u32_m128i(self.0, shift))
      } else if #[cfg(target_feature="simd128")] {
        Self(u32x4_shr(self.0, rhs))
      } else if #[cfg(all(target_feature="neon",target_arch="aarch64"))]{
        // Use `rhs % 32` to perform wrapping shift and not unbounded shift.
        #[expect(clippy::suspicious_arithmetic_impl)]
        unsafe { Self(vshlq_u32(self.0, vmovq_n_s32( -(rhs as i32 & 31)))) }
      } else {
        Self(Inner([
          self.0.0[0].wrapping_shr(rhs),
          self.0.0[1].wrapping_shr(rhs),
          self.0.0[2].wrapping_shr(rhs),
          self.0.0[3].wrapping_shr(rhs),
        ]))
      }
    }
  }

  #[inline]
  fn bitand(self, rhs: Self) -> Self::Output {
    pick! {
      if #[cfg(target_feature="sse2")] {
        Self(bitand_m128i(self.0, rhs.0))
      } else if #[cfg(target_feature="simd128")] {
        Self(v128_and(self.0, rhs.0))
      } else if #[cfg(all(target_feature="neon",target_arch="aarch64"))]{
        unsafe { Self(vandq_u32(self.0, rhs.0)) }
      } else {
        Self(Inner([
          self.0.0[0].bitand(rhs.0.0[0]),
          self.0.0[1].bitand(rhs.0.0[1]),
          self.0.0[2].bitand(rhs.0.0[2]),
          self.0.0[3].bitand(rhs.0.0[3]),
        ]))
      }
    }
  }

  #[inline]
  fn bitor(self, rhs: Self) -> Self::Output {
    pick! {
      if #[cfg(target_feature="sse2")] {
        Self(bitor_m128i(self.0, rhs.0))
      } else if #[cfg(target_feature="simd128")] {
        Self(v128_or(self.0, rhs.0))
      } else if #[cfg(all(target_feature="neon",target_arch="aarch64"))]{
        unsafe { Self(vorrq_u32(self.0, rhs.0)) }
      } else {
        Self(Inner([
          self.0.0[0].bitor(rhs.0.0[0]),
          self.0.0[1].bitor(rhs.0.0[1]),
          self.0.0[2].bitor(rhs.0.0[2]),
          self.0.0[3].bitor(rhs.0.0[3]),
        ]))
      }
    }
  }

  #[inline]
  fn bitxor(self, rhs: Self) -> Self::Output {
    pick! {
      if #[cfg(target_feature="sse2")] {
        Self(bitxor_m128i(self.0, rhs.0))
      } else if #[cfg(target_feature="simd128")] {
        Self(v128_xor(self.0, rhs.0))
      } else if #[cfg(all(target_feature="neon",target_arch="aarch64"))]{
        unsafe { Self(veorq_u32(self.0, rhs.0)) }
      } else {
        Self(Inner([
          self.0.0[0].bitxor(rhs.0.0[0]),
          self.0.0[1].bitxor(rhs.0.0[1]),
          self.0.0[2].bitxor(rhs.0.0[2]),
          self.0.0[3].bitxor(rhs.0.0[3]),
        ]))
      }
    }
  }

  #[inline]
  pub fn max(self, rhs: Self) -> Self {
    pick! {
      if #[cfg(target_feature="sse4.1")] {
        Self(max_u32_m128i(self.0, rhs.0))
      } else if #[cfg(target_feature="simd128")] {
        Self(u32x4_max(self.0, rhs.0))
      } else if #[cfg(all(target_feature="neon",target_arch="aarch64"))]{
        unsafe { Self(vmaxq_u32(self.0, rhs.0)) }
      } else if #[cfg(all(target_feature="neon",target_arch="aarch64"))]{
        unsafe { Self(vmaxq_u16(self.0, rhs.0)) }
      } else {
        let arr: [u32; 4] = cast(self);
        let rhs: [u32; 4] = cast(rhs);
        cast([
          arr[0].max(rhs[0]),
          arr[1].max(rhs[1]),
          arr[2].max(rhs[2]),
          arr[3].max(rhs[3]),
        ])
      }
    }
  }

  #[inline]
  pub fn min(self, rhs: Self) -> Self {
    pick! {
      if #[cfg(target_feature="sse4.1")] {
        Self(min_u32_m128i(self.0, rhs.0))
      } else if #[cfg(target_feature="simd128")] {
        Self(u32x4_min(self.0, rhs.0))
      } else if #[cfg(all(target_feature="neon",target_arch="aarch64"))]{
        unsafe { Self(vminq_u32(self.0, rhs.0)) }
      } else {
        let arr: [u32; 4] = cast(self);
        let rhs: [u32; 4] = cast(rhs);
        cast([
          arr[0].min(rhs[0]),
          arr[1].min(rhs[1]),
          arr[2].min(rhs[2]),
          arr[3].min(rhs[3]),
        ])
      }
    }
  }

  #[inline]
  pub fn reduce_add(self) -> u32 {
    pick! {
      if #[cfg(target_feature="sse2")] {
        let hi64  = unpack_high_i64_m128i(self.0, self.0);
        let sum64 = add_i32_m128i(hi64, self.0);
        let hi32  = shuffle_ai_f32_all_m128i::<0b10_11_00_01>(sum64);    // Swap the low two elements
        let sum32 = add_i32_m128i(sum64, hi32);
        get_i32_from_m128i_s(sum32).cast_unsigned()
      } else {
        let arr: [u32; 4] = cast(self);
        arr[0].wrapping_add(arr[1]).wrapping_add(
        arr[2].wrapping_add(arr[3]))
      }
    }
  }

  #[inline]
  pub fn reduce_mul(self) -> u32 {
    pick! {
      if #[cfg(target_feature="sse4.1")] {
        let high_64  = unpack_high_i64_m128i(self.0, self.0);
        let reduce_64 = mul_32_m128i(high_64, self.0);
        let high_32  = shuffle_ai_f32_all_m128i::<0b10_11_00_01>(reduce_64);
        let reduce_32 = mul_32_m128i(reduce_64, high_32);
        get_i32_from_m128i_s(reduce_32).cast_unsigned()
      } else if #[cfg(target_feature="simd128")] {
        let high_64 = u64x2_shuffle::<1, 0>(self.0, self.0);
        let reduce_64 = u32x4_mul(self.0, high_64);
        let high_32 = u32x4_shuffle::<1, 0, 0, 0>(reduce_64, reduce_64);
        let reduce_32 = u32x4_mul(reduce_64, high_32);
        u32x4_extract_lane::<0>(reduce_32)
      } else if #[cfg(all(target_feature="neon", target_arch="aarch64"))] {
        unsafe {
          let high_64 = vextq_u32::<2>(self.0, self.0);
          let reduce_64 = vmulq_u32(self.0, high_64);
          let high_32 = vrev64q_u32(reduce_64);
          let reduce_32 = vmulq_u32(reduce_64, high_32);
          vgetq_lane_u32::<0>(reduce_32)
        }
      } else {
        let array = self.to_array();
        array[0].wrapping_mul(array[1]).wrapping_mul(array[2].wrapping_mul(array[3]))
      }
    }
  }

  #[inline]
  pub fn reduce_max(self) -> u32 {
    let arr: [u32; 4] = cast(self);
    arr[0].max(arr[1]).max(arr[2].max(arr[3]))
  }

  #[inline]
  pub fn reduce_min(self) -> u32 {
    let arr: [u32; 4] = cast(self);
    arr[0].min(arr[1]).min(arr[2].min(arr[3]))
  }

  #[inline]
  pub fn unbounded_shl(self, rhs: Self) -> Self {
    pick! {
      if #[cfg(target_feature="avx2")] {
        Self(shl_each_u32_m128i(self.0, rhs.0))
      } else if #[cfg(all(target_feature="neon",target_arch="aarch64"))]{
        unsafe {
          // The intrinsic has different semantics so we need to mask ourselves.
          Self(vshlq_u32(self.0, vreinterpretq_s32_u32(rhs.0))) & rhs.simd_lt(32)
        }
      } else {
        let self_array = self.to_array();
        let rhs_array = rhs.to_array();

        cast([
          self_array[0].unbounded_shl(rhs_array[0]),
          self_array[1].unbounded_shl(rhs_array[1]),
          self_array[2].unbounded_shl(rhs_array[2]),
          self_array[3].unbounded_shl(rhs_array[3]),
        ])
      }
    }
  }

  #[inline]
  pub fn unbounded_shl_scalar(self, rhs: u32) -> Self {
    pick! {
      if #[cfg(target_feature="sse2")] {
        Self(shl_all_u32_m128i(self.0, cast([rhs as u64, 0])))
      } else if #[cfg(target_feature="simd128")] {
        // The intrinsic performs wrapping shift so we need to mask the result.
        Self(u32x4_shl(self.0, rhs)) & Self::splat(rhs).simd_lt(32)
      } else if #[cfg(all(target_feature="neon",target_arch="aarch64"))]{
        // The intrinsic has different semantics so we need to saturate `rhs`.
        unsafe { Self(vshlq_u32(self.0, vmovq_n_s32(rhs.min(32) as i32))) }
      } else {
        Self(Inner([
          self.0.0[0].unbounded_shl(rhs),
          self.0.0[1].unbounded_shl(rhs),
          self.0.0[2].unbounded_shl(rhs),
          self.0.0[3].unbounded_shl(rhs),
        ]))
      }
    }
  }

  #[inline]
  pub fn unbounded_shr(self, rhs: Self) -> Self {
    pick! {
      if #[cfg(target_feature="avx2")] {
        Self(shr_each_u32_m128i(self.0, rhs.0))
      } else if #[cfg(all(target_feature="neon",target_arch="aarch64"))]{
        unsafe {
          // Negate `rhs` because there is no direct shift-right intrinsic, and
          // mask to hide `rhs` overflow.
          Self(vshlq_u32(self.0, vnegq_s32(vreinterpretq_s32_u32(rhs.0)))) & rhs.simd_lt(32)
        }
      } else {
        let self_array = self.to_array();
        let rhs_array = rhs.to_array();

        Self::new([
          self_array[0].unbounded_shr(rhs_array[0]),
          self_array[1].unbounded_shr(rhs_array[1]),
          self_array[2].unbounded_shr(rhs_array[2]),
          self_array[3].unbounded_shr(rhs_array[3]),
        ])
      }
    }
  }

  #[inline]
  pub fn unbounded_shr_scalar(self, rhs: u32) -> Self {
    pick! {
      if #[cfg(target_feature="sse2")] {
        Self(shr_all_u32_m128i(self.0, cast([rhs as u64, 0])))
      } else if #[cfg(target_feature="simd128")] {
        if rhs < 32 { Self(u32x4_shr(self.0, rhs)) } else { Self::ZERO }
      } else if #[cfg(all(target_feature="neon",target_arch="aarch64"))]{
        unsafe {
          // Negate `rhs` because there is no direct shift-right intrinsic, and
          // restrict it to prevent overflow.
          Self(vshlq_u32(self.0, vmovq_n_s32(-rhs.min(32).cast_signed())))
        }
      } else {
        Self(Inner([
          self.0.0[0].unbounded_shr(rhs),
          self.0.0[1].unbounded_shr(rhs),
          self.0.0[2].unbounded_shr(rhs),
          self.0.0[3].unbounded_shr(rhs),
        ]))
      }
    }
  }

  #[inline]
  pub fn saturating_add(self, rhs: Self) -> Self {
    pick! {
      if #[cfg(any(target_feature="sse2", target_feature="simd128"))] {
        let result = self + rhs;
        let overflow = result.simd_lt(self);
        // Return `MAX` (all bits set) if overflow occurs.
        result | overflow
      } else if #[cfg(all(target_feature="neon",target_arch="aarch64"))]{
        unsafe { Self(vqaddq_u32(self.0, rhs.0)) }
      } else {
        Self(Inner([
          self.0.0[0].saturating_add(rhs.0.0[0]),
          self.0.0[1].saturating_add(rhs.0.0[1]),
          self.0.0[2].saturating_add(rhs.0.0[2]),
          self.0.0[3].saturating_add(rhs.0.0[3]),
        ]))
      }
    }
  }

  #[inline]
  pub fn saturating_sub(self, rhs: Self) -> Self {
    pick! {
      if #[cfg(any(target_feature="sse2", target_feature="simd128"))] {
        let result = self - rhs;
        let no_overflow = result.simd_le(self);
        // Return `0` (no bits set) if overflow occurs.
        result & no_overflow
      } else if #[cfg(all(target_feature="neon",target_arch="aarch64"))]{
        unsafe { Self(vqsubq_u32(self.0, rhs.0)) }
      } else {
        Self(Inner([
          self.0.0[0].saturating_sub(rhs.0.0[0]),
          self.0.0[1].saturating_sub(rhs.0.0[1]),
          self.0.0[2].saturating_sub(rhs.0.0[2]),
          self.0.0[3].saturating_sub(rhs.0.0[3]),
        ]))
      }
    }
  }

  #[inline]
  pub fn overflowing_mul(self, rhs: Self) -> (Self, Self) {
    let (low, high) = self.mul_keep_low_high(rhs);
    let overflow = high.simd_ne(Self::ZERO);
    (low, overflow)
  }

  optional_fn_widening_mul {
    #[inline]
    pub fn widening_mul(self, rhs: Self) -> u64x4 {
      pick! {
        if #[cfg(target_feature="avx2")] {
          // ok to sign extend since we are throwing away the high half of the result anyway
          let a = convert_to_i64_m256i_from_i32_m128i(self.0);
          let b = convert_to_i64_m256i_from_i32_m128i(rhs.0);
          cast(mul_u64_low_bits_m256i(a, b))
        } else if #[cfg(target_feature="sse2")] {
          let evenp = mul_widen_u32_odd_m128i(self.0, rhs.0);

          let oddp = mul_widen_u32_odd_m128i(
            shr_imm_u64_m128i::<32>(self.0),
            shr_imm_u64_m128i::<32>(rhs.0));

          Simd(crate::u64x4_::Inner(
            Simd(unpack_low_i64_m128i(evenp, oddp)),
            Simd(unpack_high_i64_m128i(evenp, oddp)),
          ))
        } else if #[cfg(target_feature="simd128")] {
          Simd(crate::u64x4_::Inner(
            Simd(u64x2_extmul_low_u32x4(self.0, rhs.0)),
            Simd(u64x2_extmul_high_u32x4(self.0, rhs.0)),
          ))
        } else if #[cfg(all(target_feature="neon",target_arch="aarch64"))] {
          unsafe {
            Simd(crate::u64x4_::Inner(
              Simd(vmull_u32(vget_low_u32(self.0), vget_low_u32(rhs.0))),
              Simd(vmull_u32(vget_high_u32(self.0), vget_high_u32(rhs.0))),
            ))
          }
        } else {
          let a: [u32; 4] = cast(self);
          let b: [u32; 4] = cast(rhs);
          cast([
            u64::from(a[0]) * u64::from(b[0]),
            u64::from(a[1]) * u64::from(b[1]),
            u64::from(a[2]) * u64::from(b[2]),
            u64::from(a[3]) * u64::from(b[3]),
          ])
        }
      }
    }
  }

  #[inline]
  pub fn mul_keep_low_high(self, rhs: Self) -> (Self, Self) {
    pick! {
      if #[cfg(target_feature="sse4.1")] {
        let even_wide_mul = mul_widen_u32_odd_m128i(self.0, rhs.0);
        let odd_wide_mul = mul_widen_u32_odd_m128i(
          shuffle_ai_f32_all_m128i::<0b_00_11_00_01>(self.0),
          shuffle_ai_f32_all_m128i::<0b_00_11_00_01>(rhs.0),
        );

        let ll_hh_1 = unpack_low_i32_m128i(even_wide_mul, odd_wide_mul);
        let ll_hh_2 = unpack_high_i32_m128i(even_wide_mul, odd_wide_mul);
        (
          Self(unpack_low_i64_m128i(ll_hh_1, ll_hh_2)),
          Self(unpack_high_i64_m128i(ll_hh_1, ll_hh_2)),
        )
      } else if #[cfg(target_feature="simd128")] {
        let low_wide_mul = u64x2_extmul_low_u32x4(self.0, rhs.0);
        let high_wide_mul = u64x2_extmul_high_u32x4(self.0, rhs.0);
        (
          Self(u32x4_shuffle::<0, 2, 4, 6>(low_wide_mul, high_wide_mul)),
          Self(u32x4_shuffle::<1, 3, 5, 7>(low_wide_mul, high_wide_mul)),
        )
      } else if #[cfg(all(target_feature="neon", target_arch="aarch64"))] {
        unsafe {
          let low_wide_mul = vreinterpretq_u32_u64(
            vmull_u32(vget_low_u32(self.0), vget_low_u32(rhs.0)),
          );
          let high_wide_mul = vreinterpretq_u32_u64(
            vmull_u32(vget_high_u32(self.0), vget_high_u32(rhs.0)),
          );
          let low_high = vuzpq_u32(low_wide_mul, high_wide_mul);
          (
            Self(low_high.0),
            Self(low_high.1),
          )
        }
      } else {
        // TODO(perf): This implementation looks quite bad. Is there a better
        // one?

        let self_array = self.to_array();
        let rhs_array = rhs.to_array();

        let widening_mul = [
          (self_array[0] as u64).wrapping_mul(rhs_array[0] as u64),
          (self_array[1] as u64).wrapping_mul(rhs_array[1] as u64),
          (self_array[2] as u64).wrapping_mul(rhs_array[2] as u64),
          (self_array[3] as u64).wrapping_mul(rhs_array[3] as u64),
        ];

        (
          Self::new([
            widening_mul[0] as u32,
            widening_mul[1] as u32,
            widening_mul[2] as u32,
            widening_mul[3] as u32,
          ]),
          Self::new([
            (widening_mul[0] >> 32) as u32,
            (widening_mul[1] >> 32) as u32,
            (widening_mul[2] >> 32) as u32,
            (widening_mul[3] >> 32) as u32,
          ]),
        )
      }
    }
  }

  #[inline]
  pub fn mul_keep_high(self, rhs: Self) -> Self {
    pick! {
      if #[cfg(target_feature="avx2")] {
        let a = convert_to_i64_m256i_from_u32_m128i(self.0);
        let b = convert_to_i64_m256i_from_u32_m128i(rhs.0);
        let r = mul_u64_low_bits_m256i(a, b);

        // the compiler does a good job shuffling the lanes around
        let b : [u32;8] = cast(r);
        cast([b[1],b[3],b[5],b[7]])
      } else if #[cfg(target_feature="sse2")] {
        let evenp = mul_widen_u32_odd_m128i(self.0, rhs.0);

        let oddp = mul_widen_u32_odd_m128i(
          shr_imm_u64_m128i::<32>(self.0),
          shr_imm_u64_m128i::<32>(rhs.0));

        // the compiler does a good job shuffling the lanes around
        let a : [u32;4]= cast(evenp);
        let b : [u32;4]= cast(oddp);
        cast([a[1],b[1],a[3],b[3]])

      } else if #[cfg(target_feature="simd128")] {
        let low =  u64x2_extmul_low_u32x4(self.0, rhs.0);
        let high = u64x2_extmul_high_u32x4(self.0, rhs.0);

        Self(u32x4_shuffle::<1, 3, 5, 7>(low, high))
      } else if #[cfg(all(target_feature="neon",target_arch="aarch64"))] {
        unsafe {
          let l = vmull_u32(vget_low_u32(self.0), vget_low_u32(rhs.0));
          let h = vmull_u32(vget_high_u32(self.0), vget_high_u32(rhs.0));
          u32x4(vcombine_u32(vshrn_n_u64(l,32), vshrn_n_u64(h,32)))
        }
      } else {
        let a: [u32; 4] = cast(self);
        let b: [u32; 4] = cast(rhs);
        cast([
          ((u64::from(a[0]) * u64::from(b[0])) >> 32) as u32,
          ((u64::from(a[1]) * u64::from(b[1])) >> 32) as u32,
          ((u64::from(a[2]) * u64::from(b[2])) >> 32) as u32,
          ((u64::from(a[3]) * u64::from(b[3])) >> 32) as u32,
        ])
      }
    }
  }
}

/// The following functionality exists only for [`u32x4`], or only for
/// particular types inconsistently.
impl u32x4 {
  /// Widening multiplication. Computes `self * rhs`, widening to a SIMD
  /// vector of larger integers.
  ///
  /// The returned value is always exact and can never overflow.
  ///
  /// This function has been renamed to [`widening_mul`].
  ///
  /// [`widening_mul`]: Self::widening_mul
  #[inline]
  #[must_use]
  #[deprecated(since = "1.6.0", note = "renamed to `widening_mul`")]
  pub fn mul_widen(self, rhs: Self) -> u64x4 {
    self.widening_mul(rhs)
  }
}
