#[cfg(all(target_feature = "neon", target_arch = "aarch64"))]
use core::arch::aarch64::*;
#[cfg(target_feature = "simd128")]
use core::arch::wasm32::*;

use super::*;

use crate::{i64x2, simd::SimdBackend};

#[cfg(not(any(
  target_feature = "sse2",
  target_feature = "simd128",
  all(target_feature = "neon", target_arch = "aarch64"),
)))]
#[repr(C, align(16))]
#[derive(Default, Clone, Copy, PartialEq, Eq)]
pub(crate) struct Inner(pub [u64; 2]);

unsafe impl SimdBackend for u64x2 {
  pick! {
    if #[cfg(target_feature="sse2")] {
      type Inner = m128i;
    } else if #[cfg(target_feature="simd128")] {
      type Inner = v128;
    } else if #[cfg(all(target_feature="neon",target_arch="aarch64"))]{
      type Inner = uint64x2_t;
    } else {
      type Inner = Inner;
    }
  }

  #[inline]
  fn simd_eq(self, rhs: Self) -> Self {
    pick! {
      if #[cfg(target_feature="sse4.1")] {
        Self(cmp_eq_mask_i64_m128i(self.0, rhs.0))
      } else if #[cfg(target_feature="simd128")] {
        Self(u64x2_eq(self.0, rhs.0))
      } else if #[cfg(all(target_feature="neon",target_arch="aarch64"))]{
        unsafe { Self(vceqq_u64(self.0, rhs.0)) }
      } else {
        let s: [u64;2] = cast(self);
        let r: [u64;2] = cast(rhs);
        cast([
          if s[0] == r[0] { -1_i64 } else { 0 },
          if s[1] == r[1] { -1_i64 } else { 0 },
        ])
      }
    }
  }

  #[inline]
  fn simd_ne(self, rhs: Self) -> Self {
    pick! {
      if #[cfg(target_feature="sse4.1")] {
        !self.simd_eq(rhs)
      } else if #[cfg(target_feature="simd128")] {
        Self(u64x2_ne(self.0, rhs.0))
      } else if #[cfg(all(target_feature="neon",target_arch="aarch64"))]{
        !self.simd_eq(rhs)
      } else {
        let s: [u64;2] = cast(self);
        let r: [u64;2] = cast(rhs);
        cast([
          if s[0] != r[0] { -1_i64 } else { 0 },
          if s[1] != r[1] { -1_i64 } else { 0 },
        ])
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
      if #[cfg(target_feature="sse4.2")] {
        // no unsigned gt so inverting the high bit will get the correct result
        let highbit = u64x2::splat(1 << 63);
        Self(cmp_gt_mask_i64_m128i((self ^ highbit).0, (rhs ^ highbit).0))
      } else if #[cfg(all(target_feature="neon",target_arch="aarch64"))]{
        unsafe { Self(vcgtq_u64(self.0, rhs.0)) }
      } else {
        // u64x2_gt on WASM is not a thing. https://github.com/WebAssembly/simd/pull/414
        let s: [u64;2] = cast(self);
        let r: [u64;2] = cast(rhs);
        cast([
          if s[0] > r[0] { u64::MAX } else { 0 },
          if s[1] > r[1] { u64::MAX } else { 0 },
        ])
      }
    }
  }

  #[inline]
  fn simd_le(self, rhs: Self) -> Self {
    pick! {
      if #[cfg(target_feature="sse4.1")] {
        !self.simd_gt(rhs)
      } else if #[cfg(all(target_feature="neon",target_arch="aarch64"))]{
        !self.simd_gt(rhs)
      } else {
        let s: [u64;2] = cast(self);
        let r: [u64;2] = cast(rhs);
        cast([
          if s[0] <= r[0] { -1_i64 } else { 0 },
          if s[1] <= r[1] { -1_i64 } else { 0 },
        ])
      }
    }
  }

  #[inline]
  fn simd_ge(self, rhs: Self) -> Self {
    pick! {
      if #[cfg(target_feature="sse4.1")] {
        !self.simd_lt(rhs)
      } else if #[cfg(all(target_feature="neon",target_arch="aarch64"))]{
        !self.simd_lt(rhs)
      } else {
        let s: [u64;2] = cast(self);
        let r: [u64;2] = cast(rhs);
        cast([
          if s[0] >= r[0] { -1_i64 } else { 0 },
          if s[1] >= r[1] { -1_i64 } else { 0 },
        ])
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
        unsafe { Self(vbslq_u64(self.0, if_one.0, if_zero.0)) }
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
        unsafe { Self(vbslq_u64(self.0, if_true.0, if_false.0)) }
      } else {
        generic_bit_blend(self, if_true, if_false)
      }
    }
  }

  #[inline]
  fn to_bitmask(self) -> u32 {
    i64x2::to_bitmask(cast(self))
  }

  #[inline]
  fn any(self) -> bool {
    i64x2::any(cast(self))
  }

  #[inline]
  fn all(self) -> bool {
    i64x2::all(cast(self))
  }

  #[inline]
  fn transpose(data: [u64x2; 2]) -> [u64x2; 2] {
    cast(i64x2::transpose(cast(data)))
  }
}

impl_simd_uint! {
  unsafe {
    T = u64,
    N = 2,
    Simd = u64x2,
    SignedSimd = i64x2,
    T_BITS = 64,
    T_BITS_MUL_2 = 128,
    [0, 1],
  }

  #[inline]
  fn not(self) -> Self::Output {
    self ^ cast::<u128, u64x2>(u128::MAX)
  }

  #[inline]
  fn add(self, rhs: Self) -> Self::Output {
    pick! {
      if #[cfg(target_feature="sse2")] {
        Self(add_i64_m128i(self.0, rhs.0))
      } else if #[cfg(target_feature="simd128")] {
        Self(u64x2_add(self.0, rhs.0))
      } else if #[cfg(all(target_feature="neon",target_arch="aarch64"))]{
        unsafe { Self(vaddq_u64(self.0, rhs.0)) }
      } else {
        Self(Inner([
          self.0.0[0].wrapping_add(rhs.0.0[0]),
          self.0.0[1].wrapping_add(rhs.0.0[1]),
        ]))
      }
    }
  }

  #[inline]
  fn sub(self, rhs: Self) -> Self::Output {
    pick! {
      if #[cfg(target_feature="sse2")] {
        Self(sub_i64_m128i(self.0, rhs.0))
      } else if #[cfg(target_feature="simd128")] {
        Self(u64x2_sub(self.0, rhs.0))
      } else if #[cfg(all(target_feature="neon",target_arch="aarch64"))]{
        unsafe { Self(vsubq_u64(self.0, rhs.0)) }
      } else {
        Self(Inner([
          self.0.0[0].wrapping_sub(rhs.0.0[0]),
          self.0.0[1].wrapping_sub(rhs.0.0[1]),
        ]))
      }
    }
  }

  #[inline]
  fn mul(self, rhs: Self) -> Self::Output {
    //we should try to implement this on sse2
    pick! {
      if #[cfg(target_feature="simd128")] {
        Self(u64x2_mul(self.0, rhs.0))
      } else {
        let arr1: [u64; 2] = cast(self);
        let arr2: [u64; 2] = cast(rhs);
        cast([
          arr1[0].wrapping_mul(arr2[0]),
          arr1[1].wrapping_mul(arr2[1]),
        ])
      }
    }
  }

  #[inline]
  fn shl(self, rhs: Self) -> Self::Output {
    pick! {
      if #[cfg(target_feature="avx2")] {
        // mask the shift count to 63 to have same behavior on all platforms
        let shift_by = rhs & Self::splat(63);
        Self(shl_each_u64_m128i(self.0, shift_by.0))
      } else if #[cfg(all(target_feature="neon", target_arch="aarch64"))] {
        unsafe {
          // mask the shift count to 63 to have same behavior on all platforms
          let shift_by = vreinterpretq_s64_u64(vandq_u64(rhs.0, vmovq_n_u64(63)));
          Self(vshlq_u64(self.0, shift_by))
        }
      } else {
        let arr: [u64; 2] = cast(self);
        let rhs: [u64; 2] = cast(rhs);
        cast([
          arr[0].wrapping_shl(rhs[0] as u32),
          arr[1].wrapping_shl(rhs[1] as u32),
        ])
      }
    }
  }

  #[inline]
  fn shl(self, rhs: u32) -> Self::Output {
    pick! {
      if #[cfg(target_feature="sse2")] {
        // Use `rhs % 64` to perform wrapping shift and not unbounded shift.
        #[expect(clippy::suspicious_arithmetic_impl)]
        let shift = cast([rhs as u64 & 63, 0]);
        Self(shl_all_u64_m128i(self.0, shift))
      } else if #[cfg(target_feature="simd128")] {
        Self(u64x2_shl(self.0, rhs))
      } else if #[cfg(all(target_feature="neon",target_arch="aarch64"))]{
        // Use `rhs % 64` to perform wrapping shift and not unbounded shift.
        #[expect(clippy::suspicious_arithmetic_impl)]
        unsafe { Self(vshlq_u64(self.0, vmovq_n_s64(rhs as i64 & 63))) }
      } else {
        Self(Inner([
          self.0.0[0].wrapping_shl(rhs),
          self.0.0[1].wrapping_shl(rhs),
        ]))
      }
    }
  }

  #[inline]
  fn shr(self, rhs: Self) -> Self::Output {
    pick! {
      if #[cfg(target_feature="avx2")] {
        // mask the shift count to 63 to have same behavior on all platforms
        let shift_by = rhs & Self::splat(63);
        Self(shr_each_u64_m128i(self.0, shift_by.0))
      } else if #[cfg(all(target_feature="neon", target_arch="aarch64"))] {
        unsafe {
          // mask the shift count to 63 to have same behavior on all platforms
          // no right shift, have to pass negative value to left shift on neon
          let shift_by = vnegq_s64(vreinterpretq_s64_u64(vandq_u64(rhs.0, vmovq_n_u64(63))));
          Self(vshlq_u64(self.0, shift_by))
        }
      } else {
        let arr: [u64; 2] = cast(self);
        let rhs: [u64; 2] = cast(rhs);
        cast([
          arr[0].wrapping_shr(rhs[0] as u32),
          arr[1].wrapping_shr(rhs[1] as u32),
        ])
      }
    }
  }

  #[inline]
  fn shr(self, rhs: u32) -> Self::Output {
    pick! {
      if #[cfg(target_feature="sse2")] {
        // Use `rhs % 64` to perform wrapping shift and not unbounded shift.
        #[expect(clippy::suspicious_arithmetic_impl)]
        let shift = cast([rhs as u64 & 63, 0]);
        Self(shr_all_u64_m128i(self.0, shift))
      } else if #[cfg(target_feature="simd128")] {
        Self(u64x2_shr(self.0, rhs))
      } else if #[cfg(all(target_feature="neon",target_arch="aarch64"))]{
        // Use `rhs % 64` to perform wrapping shift and not unbounded shift.
        #[expect(clippy::suspicious_arithmetic_impl)]
        unsafe { Self(vshlq_u64(self.0, vmovq_n_s64(-(rhs as i64 & 63)))) }
      } else {
        Self(Inner([
          self.0.0[0].wrapping_shr(rhs),
          self.0.0[1].wrapping_shr(rhs),
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
        unsafe { Self(vandq_u64(self.0, rhs.0)) }
      } else {
        Self(Inner([
          self.0.0[0].bitand(rhs.0.0[0]),
          self.0.0[1].bitand(rhs.0.0[1]),
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
        unsafe { Self(vorrq_u64(self.0, rhs.0)) }
      } else {
        Self(Inner([
          self.0.0[0].bitor(rhs.0.0[0]),
          self.0.0[1].bitor(rhs.0.0[1]),
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
        unsafe { Self(veorq_u64(self.0, rhs.0)) }
      } else {
        Self(Inner([
          self.0.0[0].bitxor(rhs.0.0[0]),
          self.0.0[1].bitxor(rhs.0.0[1]),
        ]))
      }
    }
  }

  #[inline]
  pub fn max(self, rhs: Self) -> Self {
    self.simd_gt(rhs).select(self, rhs)
  }

  #[inline]
  pub fn min(self, rhs: Self) -> Self {
    self.simd_lt(rhs).select(self, rhs)
  }

  #[inline]
  pub fn reduce_add(self) -> u64 {
    pick! {
      if #[cfg(any(target_feature="sse2", target_feature="simd128"))] {
        let array: [u64; 2] = cast(self);
        array[0].wrapping_add(array[1])
      } else if #[cfg(all(target_feature="neon",target_arch="aarch64"))]{
        unsafe { vgetq_lane_u64(self.0, 0).wrapping_add(vgetq_lane_u64(self.0, 1)) }
      } else {
        self.0.0[0].wrapping_add(self.0.0[1])
      }
    }
  }

  #[inline]
  pub fn reduce_mul(self) -> u64 {
    pick! {
      if #[cfg(any(target_feature="sse2", target_feature="simd128"))] {
        let array: [u64; 2] = cast(self);
        array[0].wrapping_mul(array[1])
      } else if #[cfg(all(target_feature="neon",target_arch="aarch64"))]{
        unsafe { vgetq_lane_u64(self.0, 0).wrapping_mul(vgetq_lane_u64(self.0, 1)) }
      } else {
        self.0.0[0].wrapping_mul(self.0.0[1])
      }
    }
  }

  #[inline]
  pub fn reduce_max(self) -> u64 {
    pick! {
      if #[cfg(any(target_feature="sse2", target_feature="simd128"))] {
        let array: [u64; 2] = cast(self);
        array[0].max(array[1])
      } else if #[cfg(all(target_feature="neon",target_arch="aarch64"))]{
        unsafe { vgetq_lane_u64(self.0, 0).max(vgetq_lane_u64(self.0, 1)) }
      } else {
        self.0.0[0].max(self.0.0[1])
      }
    }
  }

  #[inline]
  pub fn reduce_min(self) -> u64 {
    pick! {
      if #[cfg(any(target_feature="sse2", target_feature="simd128"))] {
        let array: [u64; 2] = cast(self);
        array[0].min(array[1])
      } else if #[cfg(all(target_feature="neon",target_arch="aarch64"))]{
        unsafe { vgetq_lane_u64(self.0, 0).min(vgetq_lane_u64(self.0, 1)) }
      } else {
        self.0.0[0].min(self.0.0[1])
      }
    }
  }

  #[inline]
  pub fn unbounded_shl(self, rhs: Self) -> Self {
    pick! {
      if #[cfg(target_feature="avx2")] {
        Self(shl_each_u64_m128i(self.0, rhs.0))
      } else if #[cfg(all(target_feature="neon", target_arch="aarch64"))] {
        unsafe {
          // The intrinsic has different semantics so we need to mask ourselves.
          Self(vshlq_u64(self.0, vreinterpretq_s64_u64(rhs.0))) & rhs.simd_lt(64)
        }
      } else {
        // Cannot use scalar `unbounded_shl` because it takes `u32`, which is
        // smaller than `u64`.
        (self << rhs) & rhs.simd_lt(64)
      }
    }
  }

  #[inline]
  pub fn unbounded_shl_scalar(self, rhs: u32) -> Self {
    pick! {
      if #[cfg(target_feature="sse2")] {
        Self(shl_all_u64_m128i(self.0, cast([rhs as u64, 0])))
      } else if #[cfg(target_feature="simd128")] {
        // The intrinsic performs wrapping shift so we need to mask the result.
        Self(u64x2_shl(self.0, rhs)) & Self::splat(rhs as u64).simd_lt(64)
      } else if #[cfg(all(target_feature="neon",target_arch="aarch64"))]{
        unsafe { Self(vshlq_u64(self.0, vmovq_n_s64(rhs.min(64) as i64))) }
      } else {
        Self(Inner([
          self.0.0[0].unbounded_shl(rhs),
          self.0.0[1].unbounded_shl(rhs),
        ]))
      }
    }
  }

  #[inline]
  pub fn unbounded_shr(self, rhs: Self) -> Self {
    pick! {
      if #[cfg(target_feature="avx2")] {
        Self(shr_each_u64_m128i(self.0, rhs.0))
      } else if #[cfg(all(target_feature="neon", target_arch="aarch64"))] {
        unsafe {
          // Negate `rhs` because there is no direct shift-right intrinsic, and
          // mask to hide `rhs` overflow.
          Self(vshlq_u64(self.0, vnegq_s64(vreinterpretq_s64_u64(rhs.0)))) & rhs.simd_lt(64)
        }
      } else {
        // Cannot use scalar `unbounded_shr` because it takes `u32`, which is
        // smaller than `u64`.
        (self >> rhs) & rhs.simd_lt(64)
      }
    }
  }

  #[inline]
  pub fn unbounded_shr_scalar(self, rhs: u32) -> Self {
    pick! {
      if #[cfg(target_feature="sse2")] {
        Self(shr_all_u64_m128i(self.0, cast([rhs as u64, 0])))
      } else if #[cfg(target_feature="simd128")] {
        if rhs < 64 { Self(u64x2_shr(self.0, rhs)) } else { Self::ZERO }
      } else if #[cfg(all(target_feature="neon",target_arch="aarch64"))]{
        unsafe {
          // Negate `rhs` because there is no direct shift-right intrinsic, and
          // restrict it to prevent overflow.
          Self(vshlq_u64(self.0, vmovq_n_s64(-rhs.min(64).cast_signed() as i64)))
        }
      } else {
        Self(Inner([
          self.0.0[0].unbounded_shr(rhs),
          self.0.0[1].unbounded_shr(rhs),
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
        unsafe { Self(vqaddq_u64(self.0, rhs.0)) }
      } else {
        Self(Inner([
          self.0.0[0].saturating_add(rhs.0.0[0]),
          self.0.0[1].saturating_add(rhs.0.0[1]),
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
        unsafe { Self(vqsubq_u64(self.0, rhs.0)) }
      } else {
        Self(Inner([
          self.0.0[0].saturating_sub(rhs.0.0[0]),
          self.0.0[1].saturating_sub(rhs.0.0[1]),
        ]))
      }
    }
  }

  #[inline]
  pub fn overflowing_mul(self, rhs: Self) -> (Self, Self) {
    // TODO(perf): This implementation looks quite bad. Is there a better
    // one? This intentionally avoids `mul_keep_low_high` because getting the
    // high bits of 64-bit multiplication could be slow.

    let self_array = self.to_array();
    let rhs_array = rhs.to_array();

    let result = [
      self_array[0].overflowing_mul(rhs_array[0]),
      self_array[1].overflowing_mul(rhs_array[1]),
    ];
    (
      Self::new([result[0].0, result[1].0]),
      Self::new([-(result[0].1 as i64) as u64, -(result[1].1 as i64) as u64]),
    )
  }

  optional_fn_widening_mul {
    // Cannot have `widening_mul` because there is no `u128x2` type.
  }

  #[inline]
  pub fn mul_keep_low_high(self, rhs: Self) -> (Self, Self) {
    // TODO(perf): This implementation looks quite bad. Is there a better
    // one?

    let self_array = self.to_array();
    let rhs_array = rhs.to_array();

    let widening_mul = [
      (self_array[0] as u128).wrapping_mul(rhs_array[0] as u128),
      (self_array[1] as u128).wrapping_mul(rhs_array[1] as u128),
    ];

    (
      Self::new([
        widening_mul[0] as u64,
        widening_mul[1] as u64,
      ]),
      Self::new([
        (widening_mul[0] >> 64) as u64,
        (widening_mul[1] >> 64) as u64,
      ]),
    )
  }

  #[inline]
  pub fn mul_keep_high(self, rhs: Self) -> Self {
    let arr1: [u64; 2] = cast(self);
    let arr2: [u64; 2] = cast(rhs);
    cast([
      ((arr1[0] as u128 * arr2[0] as u128) >> 64) as u64,
      ((arr1[1] as u128 * arr2[1] as u128) >> 64) as u64,
    ])
  }
}
