#[cfg(all(target_feature = "neon", target_arch = "aarch64"))]
use core::arch::aarch64::*;
#[cfg(target_feature = "simd128")]
use core::arch::wasm32::*;

use super::*;

use crate::{i8x16, i16x8, simd::SimdBackend, u16x16};

#[cfg(not(any(
  target_feature = "sse2",
  target_feature = "simd128",
  all(target_feature = "neon", target_arch = "aarch64"),
)))]
#[repr(C, align(16))]
#[derive(Default, Clone, Copy, PartialEq, Eq)]
pub(crate) struct Inner(pub [u8; 16]);

unsafe impl SimdBackend for u8x16 {
  pick! {
    if #[cfg(target_feature="sse2")] {
      type Inner = m128i;
    } else if #[cfg(target_feature="simd128")] {
      type Inner = v128;
    } else if #[cfg(all(target_feature="neon",target_arch="aarch64"))]{
      type Inner = uint8x16_t;
    } else {
      type Inner = Inner;
    }
  }
}

impl_simd! {
  unsafe {
    T = u8,
    N = 16,
    Simd = u8x16,
    optional_type_x86_inner { X86Inner = __m128i },
    optional_type_arm_inner { ArmInner = uint8x16_t },
    optional_type_wasm_inner { WasmInner = v128 },
  }

  #[inline]
  fn simd_eq(self, rhs: Self) -> Self::Output {
    pick! {
      if #[cfg(target_feature="sse2")] {
        Self(cmp_eq_mask_i8_m128i(self.0, rhs.0))
      } else if #[cfg(target_feature="simd128")] {
        Self(u8x16_eq(self.0, rhs.0))
      } else if #[cfg(all(target_feature="neon",target_arch="aarch64"))]{
        unsafe { Self(vceqq_u8(self.0, rhs.0)) }
      } else {
        Self(Inner([
          if self.0.0[0] == rhs.0.0[0] { u8::MAX } else { 0 },
          if self.0.0[1] == rhs.0.0[1] { u8::MAX } else { 0 },
          if self.0.0[2] == rhs.0.0[2] { u8::MAX } else { 0 },
          if self.0.0[3] == rhs.0.0[3] { u8::MAX } else { 0 },
          if self.0.0[4] == rhs.0.0[4] { u8::MAX } else { 0 },
          if self.0.0[5] == rhs.0.0[5] { u8::MAX } else { 0 },
          if self.0.0[6] == rhs.0.0[6] { u8::MAX } else { 0 },
          if self.0.0[7] == rhs.0.0[7] { u8::MAX } else { 0 },
          if self.0.0[8] == rhs.0.0[8] { u8::MAX } else { 0 },
          if self.0.0[9] == rhs.0.0[9] { u8::MAX } else { 0 },
          if self.0.0[10] == rhs.0.0[10] { u8::MAX } else { 0 },
          if self.0.0[11] == rhs.0.0[11] { u8::MAX } else { 0 },
          if self.0.0[12] == rhs.0.0[12] { u8::MAX } else { 0 },
          if self.0.0[13] == rhs.0.0[13] { u8::MAX } else { 0 },
          if self.0.0[14] == rhs.0.0[14] { u8::MAX } else { 0 },
          if self.0.0[15] == rhs.0.0[15] { u8::MAX } else { 0 },
        ]))
      }
    }
  }

  #[inline]
  fn simd_ne(self, rhs: Self) -> Self::Output {
    pick! {
      if #[cfg(target_feature="sse2")] {
        !self.simd_eq(rhs)
      } else if #[cfg(target_feature="simd128")] {
        Self(u8x16_ne(self.0, rhs.0))
      } else if #[cfg(all(target_feature="neon",target_arch="aarch64"))]{
        !self.simd_eq(rhs)
      } else {
        Self(Inner([
          if self.0.0[0] != rhs.0.0[0] { u8::MAX } else { 0 },
          if self.0.0[1] != rhs.0.0[1] { u8::MAX } else { 0 },
          if self.0.0[2] != rhs.0.0[2] { u8::MAX } else { 0 },
          if self.0.0[3] != rhs.0.0[3] { u8::MAX } else { 0 },
          if self.0.0[4] != rhs.0.0[4] { u8::MAX } else { 0 },
          if self.0.0[5] != rhs.0.0[5] { u8::MAX } else { 0 },
          if self.0.0[6] != rhs.0.0[6] { u8::MAX } else { 0 },
          if self.0.0[7] != rhs.0.0[7] { u8::MAX } else { 0 },
          if self.0.0[8] != rhs.0.0[8] { u8::MAX } else { 0 },
          if self.0.0[9] != rhs.0.0[9] { u8::MAX } else { 0 },
          if self.0.0[10] != rhs.0.0[10] { u8::MAX } else { 0 },
          if self.0.0[11] != rhs.0.0[11] { u8::MAX } else { 0 },
          if self.0.0[12] != rhs.0.0[12] { u8::MAX } else { 0 },
          if self.0.0[13] != rhs.0.0[13] { u8::MAX } else { 0 },
          if self.0.0[14] != rhs.0.0[14] { u8::MAX } else { 0 },
          if self.0.0[15] != rhs.0.0[15] { u8::MAX } else { 0 },
        ]))
      }
    }
  }

  #[inline]
  fn simd_lt(self, rhs: Self) -> Self::Output {
    pick! {
      if #[cfg(target_feature="sse2")] {
        // Convert from u8 to i8.
        let offset = Self::splat(0x80);
        let self_i8 = self.bitxor(offset).0;
        let rhs_i8 = rhs.bitxor(offset).0;
        Self(cmp_lt_mask_i8_m128i(self_i8, rhs_i8))
      } else if #[cfg(target_feature="simd128")] {
        Self(u8x16_lt(self.0, rhs.0))
      } else if #[cfg(all(target_feature="neon",target_arch="aarch64"))]{
        unsafe { Self(vcltq_u8(self.0, rhs.0)) }
      } else {
        Self(Inner([
          if self.0.0[0] < rhs.0.0[0] { u8::MAX } else { 0 },
          if self.0.0[1] < rhs.0.0[1] { u8::MAX } else { 0 },
          if self.0.0[2] < rhs.0.0[2] { u8::MAX } else { 0 },
          if self.0.0[3] < rhs.0.0[3] { u8::MAX } else { 0 },
          if self.0.0[4] < rhs.0.0[4] { u8::MAX } else { 0 },
          if self.0.0[5] < rhs.0.0[5] { u8::MAX } else { 0 },
          if self.0.0[6] < rhs.0.0[6] { u8::MAX } else { 0 },
          if self.0.0[7] < rhs.0.0[7] { u8::MAX } else { 0 },
          if self.0.0[8] < rhs.0.0[8] { u8::MAX } else { 0 },
          if self.0.0[9] < rhs.0.0[9] { u8::MAX } else { 0 },
          if self.0.0[10] < rhs.0.0[10] { u8::MAX } else { 0 },
          if self.0.0[11] < rhs.0.0[11] { u8::MAX } else { 0 },
          if self.0.0[12] < rhs.0.0[12] { u8::MAX } else { 0 },
          if self.0.0[13] < rhs.0.0[13] { u8::MAX } else { 0 },
          if self.0.0[14] < rhs.0.0[14] { u8::MAX } else { 0 },
          if self.0.0[15] < rhs.0.0[15] { u8::MAX } else { 0 },
        ]))
      }
    }
  }

  #[inline]
  fn simd_gt(self, rhs: Self) -> Self::Output {
    pick! {
      if #[cfg(target_feature="sse2")] {
        // Convert from u8 to i8.
        let offset = Self::splat(0x80);
        let self_i8 = self.bitxor(offset).0;
        let rhs_i8 = rhs.bitxor(offset).0;
        Self(cmp_gt_mask_i8_m128i(self_i8, rhs_i8))
      } else if #[cfg(target_feature="simd128")] {
        Self(u8x16_gt(self.0, rhs.0))
      } else if #[cfg(all(target_feature="neon",target_arch="aarch64"))]{
        unsafe { Self(vcgtq_u8(self.0, rhs.0)) }
      } else {
        Self(Inner([
          if self.0.0[0] > rhs.0.0[0] { u8::MAX } else { 0 },
          if self.0.0[1] > rhs.0.0[1] { u8::MAX } else { 0 },
          if self.0.0[2] > rhs.0.0[2] { u8::MAX } else { 0 },
          if self.0.0[3] > rhs.0.0[3] { u8::MAX } else { 0 },
          if self.0.0[4] > rhs.0.0[4] { u8::MAX } else { 0 },
          if self.0.0[5] > rhs.0.0[5] { u8::MAX } else { 0 },
          if self.0.0[6] > rhs.0.0[6] { u8::MAX } else { 0 },
          if self.0.0[7] > rhs.0.0[7] { u8::MAX } else { 0 },
          if self.0.0[8] > rhs.0.0[8] { u8::MAX } else { 0 },
          if self.0.0[9] > rhs.0.0[9] { u8::MAX } else { 0 },
          if self.0.0[10] > rhs.0.0[10] { u8::MAX } else { 0 },
          if self.0.0[11] > rhs.0.0[11] { u8::MAX } else { 0 },
          if self.0.0[12] > rhs.0.0[12] { u8::MAX } else { 0 },
          if self.0.0[13] > rhs.0.0[13] { u8::MAX } else { 0 },
          if self.0.0[14] > rhs.0.0[14] { u8::MAX } else { 0 },
          if self.0.0[15] > rhs.0.0[15] { u8::MAX } else { 0 },
        ]))
      }
    }
  }

  #[inline]
  fn simd_le(self, rhs: Self) -> Self::Output {
    pick! {
      if #[cfg(target_feature="sse2")] {
        // Convert from u8 to i8.
        let offset = Self::splat(0x80);
        let self_i8 = self.bitxor(offset).0;
        let rhs_i8 = rhs.bitxor(offset).0;
        // a <= b  is equivalent to  !(b < a)  or  !(a > b)
        let gt_mask = Self(cmp_gt_mask_i8_m128i(self_i8, rhs_i8));
        Self(gt_mask.bitxor(Self::splat(0xFF)).0)
      } else if #[cfg(target_feature="simd128")] {
        Self(u8x16_le(self.0, rhs.0))
      } else if #[cfg(all(target_feature="neon",target_arch="aarch64"))]{
        unsafe { Self(vcleq_u8(self.0, rhs.0)) }
      } else {
        Self(Inner([
          if self.0.0[0] <= rhs.0.0[0] { u8::MAX } else { 0 },
          if self.0.0[1] <= rhs.0.0[1] { u8::MAX } else { 0 },
          if self.0.0[2] <= rhs.0.0[2] { u8::MAX } else { 0 },
          if self.0.0[3] <= rhs.0.0[3] { u8::MAX } else { 0 },
          if self.0.0[4] <= rhs.0.0[4] { u8::MAX } else { 0 },
          if self.0.0[5] <= rhs.0.0[5] { u8::MAX } else { 0 },
          if self.0.0[6] <= rhs.0.0[6] { u8::MAX } else { 0 },
          if self.0.0[7] <= rhs.0.0[7] { u8::MAX } else { 0 },
          if self.0.0[8] <= rhs.0.0[8] { u8::MAX } else { 0 },
          if self.0.0[9] <= rhs.0.0[9] { u8::MAX } else { 0 },
          if self.0.0[10] <= rhs.0.0[10] { u8::MAX } else { 0 },
          if self.0.0[11] <= rhs.0.0[11] { u8::MAX } else { 0 },
          if self.0.0[12] <= rhs.0.0[12] { u8::MAX } else { 0 },
          if self.0.0[13] <= rhs.0.0[13] { u8::MAX } else { 0 },
          if self.0.0[14] <= rhs.0.0[14] { u8::MAX } else { 0 },
          if self.0.0[15] <= rhs.0.0[15] { u8::MAX } else { 0 },
        ]))
      }
    }
  }

  #[inline]
  fn simd_ge(self, rhs: Self) -> Self::Output {
    pick! {
      if #[cfg(target_feature="sse2")] {
        // Convert from u8 to i8.
        let offset = Self::splat(0x80);
        let self_i8 = self.bitxor(offset).0;
        let rhs_i8 = rhs.bitxor(offset).0;
        // a >= b  is equivalent to  !(b > a)  or  !(a < b)
        let lt_mask = Self(cmp_lt_mask_i8_m128i(self_i8, rhs_i8));
        Self(lt_mask.bitxor(Self::splat(0xFF)).0)
      } else if #[cfg(target_feature="simd128")] {
        Self(u8x16_ge(self.0, rhs.0))
      } else if #[cfg(all(target_feature="neon",target_arch="aarch64"))]{
        unsafe { Self(vcgeq_u8(self.0, rhs.0)) }
      } else {
        Self(Inner([
          if self.0.0[0] >= rhs.0.0[0] { u8::MAX } else { 0 },
          if self.0.0[1] >= rhs.0.0[1] { u8::MAX } else { 0 },
          if self.0.0[2] >= rhs.0.0[2] { u8::MAX } else { 0 },
          if self.0.0[3] >= rhs.0.0[3] { u8::MAX } else { 0 },
          if self.0.0[4] >= rhs.0.0[4] { u8::MAX } else { 0 },
          if self.0.0[5] >= rhs.0.0[5] { u8::MAX } else { 0 },
          if self.0.0[6] >= rhs.0.0[6] { u8::MAX } else { 0 },
          if self.0.0[7] >= rhs.0.0[7] { u8::MAX } else { 0 },
          if self.0.0[8] >= rhs.0.0[8] { u8::MAX } else { 0 },
          if self.0.0[9] >= rhs.0.0[9] { u8::MAX } else { 0 },
          if self.0.0[10] >= rhs.0.0[10] { u8::MAX } else { 0 },
          if self.0.0[11] >= rhs.0.0[11] { u8::MAX } else { 0 },
          if self.0.0[12] >= rhs.0.0[12] { u8::MAX } else { 0 },
          if self.0.0[13] >= rhs.0.0[13] { u8::MAX } else { 0 },
          if self.0.0[14] >= rhs.0.0[14] { u8::MAX } else { 0 },
          if self.0.0[15] >= rhs.0.0[15] { u8::MAX } else { 0 },
        ]))
      }
    }
  }

  #[inline]
  pub fn bitselect(self, if_one: Self, if_zero: Self) -> Self {
    pick! {
      if #[cfg(target_feature="sse2")] {
        Self(bitor_m128i(
          bitand_m128i(if_one.0, self.0),
          bitandnot_m128i(self.0, if_zero.0),
        ))
      } else if #[cfg(target_feature="simd128")] {
        Self(v128_bitselect(if_one.0, if_zero.0, self.0))
      } else if #[cfg(all(target_feature="neon",target_arch="aarch64"))]{
        unsafe { Self(vbslq_u8(self.0, if_one.0, if_zero.0)) }
      } else {
        generic_bit_blend(self, if_one, if_zero)
      }
    }
  }

  #[inline]
  pub fn select(self, if_true: Self, if_false: Self) -> Self {
    pick! {
      if #[cfg(target_feature="sse4.1")] {
        Self(blend_varying_i8_m128i(if_false.0, if_true.0, self.0))
      } else if #[cfg(target_feature="simd128")] {
        Self(v128_bitselect(if_true.0, if_false.0, self.0))
      } else if #[cfg(all(target_feature="neon",target_arch="aarch64"))]{
        unsafe { Self(vbslq_u8(self.0, if_true.0, if_false.0)) }
      } else {
        generic_bit_blend(self, if_true, if_false)
      }
    }
  }

  #[inline]
  pub fn to_bitmask(self) -> u32 {
    i8x16::to_bitmask(cast(self)) as u32
  }

  #[inline]
  pub fn any(self) -> bool {
    i8x16::any(cast(self))
  }

  #[inline]
  pub fn all(self) -> bool {
    i8x16::all(cast(self))
  }

  ///
  /// Currently this function is never accelerated.
  #[inline]
  pub fn transpose(data: [u8x16; 16]) -> [u8x16; 16] {
    cast(i8x16::transpose(cast(data)))
  }
}

impl_simd_uint! {
  unsafe {
    T = u8,
    N = 16,
    Simd = u8x16,
    SignedSimd = i8x16,
    T_BITS = 8,
    T_BITS_MUL_2 = 16,
    [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15],
  }

  #[inline]
  fn not(self) -> Self::Output {
    self ^ cast::<u128, u8x16>(u128::MAX)
  }

  #[inline]
  fn add(self, rhs: Self) -> Self::Output {
    pick! {
      if #[cfg(target_feature="sse2")] {
        Self(add_i8_m128i(self.0, rhs.0))
      } else if #[cfg(target_feature="simd128")] {
        Self(u8x16_add(self.0, rhs.0))
      } else if #[cfg(all(target_feature="neon",target_arch="aarch64"))]{
        unsafe { Self(vaddq_u8(self.0, rhs.0)) }
      } else {
        Self(Inner([
          self.0.0[0].wrapping_add(rhs.0.0[0]),
          self.0.0[1].wrapping_add(rhs.0.0[1]),
          self.0.0[2].wrapping_add(rhs.0.0[2]),
          self.0.0[3].wrapping_add(rhs.0.0[3]),
          self.0.0[4].wrapping_add(rhs.0.0[4]),
          self.0.0[5].wrapping_add(rhs.0.0[5]),
          self.0.0[6].wrapping_add(rhs.0.0[6]),
          self.0.0[7].wrapping_add(rhs.0.0[7]),
          self.0.0[8].wrapping_add(rhs.0.0[8]),
          self.0.0[9].wrapping_add(rhs.0.0[9]),
          self.0.0[10].wrapping_add(rhs.0.0[10]),
          self.0.0[11].wrapping_add(rhs.0.0[11]),
          self.0.0[12].wrapping_add(rhs.0.0[12]),
          self.0.0[13].wrapping_add(rhs.0.0[13]),
          self.0.0[14].wrapping_add(rhs.0.0[14]),
          self.0.0[15].wrapping_add(rhs.0.0[15]),
        ]))
      }
    }
  }

  #[inline]
  fn sub(self, rhs: Self) -> Self::Output {
    pick! {
      if #[cfg(target_feature="sse2")] {
        Self(sub_i8_m128i(self.0, rhs.0))
      } else if #[cfg(target_feature="simd128")] {
        Self(u8x16_sub(self.0, rhs.0))
      } else if #[cfg(all(target_feature="neon",target_arch="aarch64"))]{
        unsafe { Self(vsubq_u8(self.0, rhs.0)) }
      } else {
        Self(Inner([
          self.0.0[0].wrapping_sub(rhs.0.0[0]),
          self.0.0[1].wrapping_sub(rhs.0.0[1]),
          self.0.0[2].wrapping_sub(rhs.0.0[2]),
          self.0.0[3].wrapping_sub(rhs.0.0[3]),
          self.0.0[4].wrapping_sub(rhs.0.0[4]),
          self.0.0[5].wrapping_sub(rhs.0.0[5]),
          self.0.0[6].wrapping_sub(rhs.0.0[6]),
          self.0.0[7].wrapping_sub(rhs.0.0[7]),
          self.0.0[8].wrapping_sub(rhs.0.0[8]),
          self.0.0[9].wrapping_sub(rhs.0.0[9]),
          self.0.0[10].wrapping_sub(rhs.0.0[10]),
          self.0.0[11].wrapping_sub(rhs.0.0[11]),
          self.0.0[12].wrapping_sub(rhs.0.0[12]),
          self.0.0[13].wrapping_sub(rhs.0.0[13]),
          self.0.0[14].wrapping_sub(rhs.0.0[14]),
          self.0.0[15].wrapping_sub(rhs.0.0[15]),
        ]))
      }
    }
  }

  #[inline]
  fn mul(self, rhs: Self) -> Self::Output {
    // For x86 and wasm, this technically can be done explicitly by converting
    // to `i16` then converting back after multiplication, but that may not
    // actually be faster than auto-vectorization.
    pick! {
      if #[cfg(all(target_feature="neon",target_arch="aarch64"))]{
        unsafe { Self(vmulq_u8(self.0, rhs.0)) }
      } else {
        let self_array: [u8; 16] = cast(self);
        let rhs_array: [u8; 16] = cast(rhs);

        Self::new([
          self_array[0].wrapping_mul(rhs_array[0]),
          self_array[1].wrapping_mul(rhs_array[1]),
          self_array[2].wrapping_mul(rhs_array[2]),
          self_array[3].wrapping_mul(rhs_array[3]),
          self_array[4].wrapping_mul(rhs_array[4]),
          self_array[5].wrapping_mul(rhs_array[5]),
          self_array[6].wrapping_mul(rhs_array[6]),
          self_array[7].wrapping_mul(rhs_array[7]),
          self_array[8].wrapping_mul(rhs_array[8]),
          self_array[9].wrapping_mul(rhs_array[9]),
          self_array[10].wrapping_mul(rhs_array[10]),
          self_array[11].wrapping_mul(rhs_array[11]),
          self_array[12].wrapping_mul(rhs_array[12]),
          self_array[13].wrapping_mul(rhs_array[13]),
          self_array[14].wrapping_mul(rhs_array[14]),
          self_array[15].wrapping_mul(rhs_array[15]),
        ])
      }
    }
  }

  #[inline]
  fn shl(self, rhs: Self) -> Self::Output {
    // For x86, this technically can be done explicitly by converting
    // to `u16` or `u32` then converting back after multiplication, but that may
    // not actually be faster than auto-vectorization.
    pick! {
      if #[cfg(all(target_feature="neon", target_arch="aarch64"))] {
        unsafe {
          // Mask `rhs` to 7 to match `wrapping_shl`.
          let shift_by = vreinterpretq_s8_u8(vandq_u8(rhs.0, vmovq_n_u8(7)));
          Self(vshlq_u8(self.0, shift_by))
        }
      } else {
        let self_array: [u8; 16] = cast(self);
        let rhs_array: [u8; 16] = cast(rhs);

        Self::new([
          self_array[0].wrapping_shl(rhs_array[0] as u32),
          self_array[1].wrapping_shl(rhs_array[1] as u32),
          self_array[2].wrapping_shl(rhs_array[2] as u32),
          self_array[3].wrapping_shl(rhs_array[3] as u32),
          self_array[4].wrapping_shl(rhs_array[4] as u32),
          self_array[5].wrapping_shl(rhs_array[5] as u32),
          self_array[6].wrapping_shl(rhs_array[6] as u32),
          self_array[7].wrapping_shl(rhs_array[7] as u32),
          self_array[8].wrapping_shl(rhs_array[8] as u32),
          self_array[9].wrapping_shl(rhs_array[9] as u32),
          self_array[10].wrapping_shl(rhs_array[10] as u32),
          self_array[11].wrapping_shl(rhs_array[11] as u32),
          self_array[12].wrapping_shl(rhs_array[12] as u32),
          self_array[13].wrapping_shl(rhs_array[13] as u32),
          self_array[14].wrapping_shl(rhs_array[14] as u32),
          self_array[15].wrapping_shl(rhs_array[15] as u32),
        ])
      }
    }
  }

  #[inline]
  fn shl(self, rhs: u32) -> Self::Output {
    // For x86, this technically can be done explicitly by converting
    // to `u16` or `u32` then converting back after multiplication, but that
    // may not actually be faster than auto-vectorization.
    pick! {
      if #[cfg(target_feature="simd128")] {
        // Mask `rhs` to 7 to match `wrapping_shl`.
        Self(u8x16_shl(self.0, rhs & 7))
      } else if #[cfg(all(target_feature="neon",target_arch="aarch64"))]{
        // Mask `rhs` to 7 to match `wrapping_shl`.
        unsafe { Self(vshlq_u8(self.0, vmovq_n_s8(rhs as i8 & 7))) }
      } else {
        let self_array = self.to_array();

        cast([
          self_array[0].wrapping_shl(rhs),
          self_array[1].wrapping_shl(rhs),
          self_array[2].wrapping_shl(rhs),
          self_array[3].wrapping_shl(rhs),
          self_array[4].wrapping_shl(rhs),
          self_array[5].wrapping_shl(rhs),
          self_array[6].wrapping_shl(rhs),
          self_array[7].wrapping_shl(rhs),
          self_array[8].wrapping_shl(rhs),
          self_array[9].wrapping_shl(rhs),
          self_array[10].wrapping_shl(rhs),
          self_array[11].wrapping_shl(rhs),
          self_array[12].wrapping_shl(rhs),
          self_array[13].wrapping_shl(rhs),
          self_array[14].wrapping_shl(rhs),
          self_array[15].wrapping_shl(rhs),
        ])
      }
    }
  }

  #[inline]
  fn shr(self, rhs: Self) -> Self::Output {
    // For x86, this technically can be done explicitly by converting
    // to `u16` or `u32` then converting back after multiplication, but that may
    // not actually be faster than auto-vectorization.
    pick! {
      if #[cfg(all(target_feature="neon", target_arch="aarch64"))] {
        unsafe {
          // Mask `rhs` to 7 to match `wrapping_shr`, and negate it because
          // there is no shift-right intrinsic.
          let neg_rhs = vnegq_s8(vreinterpretq_s8_u8(vandq_u8(rhs.0, vmovq_n_u8(7))));
          Self(vshlq_u8(self.0, neg_rhs))
        }
      } else {
        let self_array: [u8; 16] = cast(self);
        let rhs_array: [u8; 16] = cast(rhs);

        Self::new([
          self_array[0].wrapping_shr(rhs_array[0] as u32),
          self_array[1].wrapping_shr(rhs_array[1] as u32),
          self_array[2].wrapping_shr(rhs_array[2] as u32),
          self_array[3].wrapping_shr(rhs_array[3] as u32),
          self_array[4].wrapping_shr(rhs_array[4] as u32),
          self_array[5].wrapping_shr(rhs_array[5] as u32),
          self_array[6].wrapping_shr(rhs_array[6] as u32),
          self_array[7].wrapping_shr(rhs_array[7] as u32),
          self_array[8].wrapping_shr(rhs_array[8] as u32),
          self_array[9].wrapping_shr(rhs_array[9] as u32),
          self_array[10].wrapping_shr(rhs_array[10] as u32),
          self_array[11].wrapping_shr(rhs_array[11] as u32),
          self_array[12].wrapping_shr(rhs_array[12] as u32),
          self_array[13].wrapping_shr(rhs_array[13] as u32),
          self_array[14].wrapping_shr(rhs_array[14] as u32),
          self_array[15].wrapping_shr(rhs_array[15] as u32),
        ])
      }
    }
  }

  #[inline]
  fn shr(self, rhs: u32) -> Self::Output {
    // For x86, this technically can be done explicitly by converting
    // to `u16` or `u32` then converting back after multiplication, but that
    // may not actually be faster than auto-vectorization.
    pick! {
      if #[cfg(target_feature="simd128")] {
        // Mask `rhs` to 7 to match `wrapping_shr`.
        Self(u8x16_shr(self.0, rhs & 7))
      } else if #[cfg(all(target_feature="neon",target_arch="aarch64"))]{
        // Mask `rhs` to 7 to match `wrapping_shr`, and negate it because
        // there is no shift-right intrinsic.
        unsafe { Self(vshlq_u8(self.0, vmovq_n_s8(-(rhs as i8 & 7)))) }
      } else {
        let self_array = self.to_array();

        cast([
          self_array[0].wrapping_shr(rhs),
          self_array[1].wrapping_shr(rhs),
          self_array[2].wrapping_shr(rhs),
          self_array[3].wrapping_shr(rhs),
          self_array[4].wrapping_shr(rhs),
          self_array[5].wrapping_shr(rhs),
          self_array[6].wrapping_shr(rhs),
          self_array[7].wrapping_shr(rhs),
          self_array[8].wrapping_shr(rhs),
          self_array[9].wrapping_shr(rhs),
          self_array[10].wrapping_shr(rhs),
          self_array[11].wrapping_shr(rhs),
          self_array[12].wrapping_shr(rhs),
          self_array[13].wrapping_shr(rhs),
          self_array[14].wrapping_shr(rhs),
          self_array[15].wrapping_shr(rhs),
        ])
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
        unsafe { Self(vandq_u8(self.0, rhs.0)) }
      } else {
        Self(Inner([
          self.0.0[0].bitand(rhs.0.0[0]),
          self.0.0[1].bitand(rhs.0.0[1]),
          self.0.0[2].bitand(rhs.0.0[2]),
          self.0.0[3].bitand(rhs.0.0[3]),
          self.0.0[4].bitand(rhs.0.0[4]),
          self.0.0[5].bitand(rhs.0.0[5]),
          self.0.0[6].bitand(rhs.0.0[6]),
          self.0.0[7].bitand(rhs.0.0[7]),
          self.0.0[8].bitand(rhs.0.0[8]),
          self.0.0[9].bitand(rhs.0.0[9]),
          self.0.0[10].bitand(rhs.0.0[10]),
          self.0.0[11].bitand(rhs.0.0[11]),
          self.0.0[12].bitand(rhs.0.0[12]),
          self.0.0[13].bitand(rhs.0.0[13]),
          self.0.0[14].bitand(rhs.0.0[14]),
          self.0.0[15].bitand(rhs.0.0[15]),
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
        unsafe { Self(vorrq_u8(self.0, rhs.0)) }
      } else {
        Self(Inner([
          self.0.0[0].bitor(rhs.0.0[0]),
          self.0.0[1].bitor(rhs.0.0[1]),
          self.0.0[2].bitor(rhs.0.0[2]),
          self.0.0[3].bitor(rhs.0.0[3]),
          self.0.0[4].bitor(rhs.0.0[4]),
          self.0.0[5].bitor(rhs.0.0[5]),
          self.0.0[6].bitor(rhs.0.0[6]),
          self.0.0[7].bitor(rhs.0.0[7]),
          self.0.0[8].bitor(rhs.0.0[8]),
          self.0.0[9].bitor(rhs.0.0[9]),
          self.0.0[10].bitor(rhs.0.0[10]),
          self.0.0[11].bitor(rhs.0.0[11]),
          self.0.0[12].bitor(rhs.0.0[12]),
          self.0.0[13].bitor(rhs.0.0[13]),
          self.0.0[14].bitor(rhs.0.0[14]),
          self.0.0[15].bitor(rhs.0.0[15]),
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
        unsafe { Self(veorq_u8(self.0, rhs.0)) }
      } else {
        Self(Inner([
          self.0.0[0].bitxor(rhs.0.0[0]),
          self.0.0[1].bitxor(rhs.0.0[1]),
          self.0.0[2].bitxor(rhs.0.0[2]),
          self.0.0[3].bitxor(rhs.0.0[3]),
          self.0.0[4].bitxor(rhs.0.0[4]),
          self.0.0[5].bitxor(rhs.0.0[5]),
          self.0.0[6].bitxor(rhs.0.0[6]),
          self.0.0[7].bitxor(rhs.0.0[7]),
          self.0.0[8].bitxor(rhs.0.0[8]),
          self.0.0[9].bitxor(rhs.0.0[9]),
          self.0.0[10].bitxor(rhs.0.0[10]),
          self.0.0[11].bitxor(rhs.0.0[11]),
          self.0.0[12].bitxor(rhs.0.0[12]),
          self.0.0[13].bitxor(rhs.0.0[13]),
          self.0.0[14].bitxor(rhs.0.0[14]),
          self.0.0[15].bitxor(rhs.0.0[15]),
        ]))
      }
    }
  }

  #[inline]
  pub fn max(self, rhs: Self) -> Self {
    pick! {
      if #[cfg(target_feature="sse2")] {
        Self(max_u8_m128i(self.0, rhs.0))
      } else if #[cfg(target_feature="simd128")] {
        Self(u8x16_max(self.0, rhs.0))
      } else if #[cfg(all(target_feature="neon",target_arch="aarch64"))]{
        unsafe { Self(vmaxq_u8(self.0, rhs.0)) }
      } else {
        Self(Inner([
          self.0.0[0].max(rhs.0.0[0]),
          self.0.0[1].max(rhs.0.0[1]),
          self.0.0[2].max(rhs.0.0[2]),
          self.0.0[3].max(rhs.0.0[3]),
          self.0.0[4].max(rhs.0.0[4]),
          self.0.0[5].max(rhs.0.0[5]),
          self.0.0[6].max(rhs.0.0[6]),
          self.0.0[7].max(rhs.0.0[7]),
          self.0.0[8].max(rhs.0.0[8]),
          self.0.0[9].max(rhs.0.0[9]),
          self.0.0[10].max(rhs.0.0[10]),
          self.0.0[11].max(rhs.0.0[11]),
          self.0.0[12].max(rhs.0.0[12]),
          self.0.0[13].max(rhs.0.0[13]),
          self.0.0[14].max(rhs.0.0[14]),
          self.0.0[15].max(rhs.0.0[15]),
        ]))
      }
    }
  }

  #[inline]
  pub fn min(self, rhs: Self) -> Self {
    pick! {
      if #[cfg(target_feature="sse2")] {
        Self(min_u8_m128i(self.0, rhs.0))
      } else if #[cfg(target_feature="simd128")] {
        Self(u8x16_min(self.0, rhs.0))
      } else if #[cfg(all(target_feature="neon",target_arch="aarch64"))]{
        unsafe { Self(vminq_u8(self.0, rhs.0)) }
      } else {
        Self(Inner([
          self.0.0[0].min(rhs.0.0[0]),
          self.0.0[1].min(rhs.0.0[1]),
          self.0.0[2].min(rhs.0.0[2]),
          self.0.0[3].min(rhs.0.0[3]),
          self.0.0[4].min(rhs.0.0[4]),
          self.0.0[5].min(rhs.0.0[5]),
          self.0.0[6].min(rhs.0.0[6]),
          self.0.0[7].min(rhs.0.0[7]),
          self.0.0[8].min(rhs.0.0[8]),
          self.0.0[9].min(rhs.0.0[9]),
          self.0.0[10].min(rhs.0.0[10]),
          self.0.0[11].min(rhs.0.0[11]),
          self.0.0[12].min(rhs.0.0[12]),
          self.0.0[13].min(rhs.0.0[13]),
          self.0.0[14].min(rhs.0.0[14]),
          self.0.0[15].min(rhs.0.0[15]),
        ]))
      }
    }
  }

  #[inline]
  pub fn reduce_add(self) -> u8 {
    #[allow(dead_code)]
    const SHUFFLE_1: [u8; 16] =
      [8, 9, 10, 11, 12, 13, 14, 15, 0, 0, 0, 0, 0, 0, 0, 0];
    #[allow(dead_code)]
    const SHUFFLE_2: [u8; 16] =
      [4, 5, 6, 7, 0, 1, 2, 3, 0, 0, 0, 0, 0, 0, 0, 0];
    #[allow(dead_code)]
    const SHUFFLE_3: [u8; 16] =
      [2, 3, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0];
    #[allow(dead_code)]
    const SHUFFLE_4: [u8; 16] =
      [1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0];

    pick! {
      if #[cfg(target_feature="ssse3")] {
        let rhs = shuffle_av_i8z_all_m128i(self.0, m128i::from(SHUFFLE_1));
        let sum = add_i8_m128i(self.0, rhs);
        let rhs = shuffle_av_i8z_all_m128i(sum, m128i::from(SHUFFLE_2));
        let sum = add_i8_m128i(sum, rhs);
        let rhs = shuffle_av_i8z_all_m128i(sum, m128i::from(SHUFFLE_3));
        let sum = add_i8_m128i(sum, rhs);
        let rhs = shuffle_av_i8z_all_m128i(sum, m128i::from(SHUFFLE_4));
        let sum = add_i8_m128i(sum, rhs);
        get_i32_from_m128i_s(sum) as u8
      } else if #[cfg(target_feature="simd128")] {
        let rhs = u8x16_shuffle::<8, 9, 10, 11, 12, 13, 14, 15, 0, 1, 2, 3, 4, 5, 6, 7>(self.0, self.0);
        let sum = u8x16_add(self.0, rhs);
        let rhs = u8x16_shuffle::<4, 5, 6, 7, 0, 1, 2, 3, 0, 0, 0, 0, 0, 0, 0, 0>(sum, sum);
        let sum = u8x16_add(sum, rhs);
        let rhs = u8x16_shuffle::<2, 3, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0>(sum, sum);
        let sum = u8x16_add(sum, rhs);
        let rhs = u8x16_shuffle::<1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0>(sum, sum);
        let sum = u8x16_add(sum, rhs);
        u8x16_extract_lane::<0>(sum)
      } else if #[cfg(all(target_feature="neon",target_arch="aarch64"))]{
        unsafe {
          // Use `transmute` instead of `cast` because `uint8x16_t` does not
          // implement `bytemuck::Pod`.
          let rhs = vqtbl1q_u8(self.0, core::mem::transmute(SHUFFLE_1));
          let sum = vaddq_u8(self.0, rhs);
          let rhs = vqtbl1q_u8(sum, core::mem::transmute(SHUFFLE_2));
          let sum = vaddq_u8(sum, rhs);
          let rhs = vqtbl1q_u8(sum, core::mem::transmute(SHUFFLE_3));
          let sum = vaddq_u8(sum, rhs);
          let rhs = vqtbl1q_u8(sum, core::mem::transmute(SHUFFLE_4));
          let sum = vaddq_u8(sum, rhs);
          vgetq_lane_u8(sum, 0)
        }
      } else {
        let array: [u8; 16] = cast(self);
        array.into_iter().reduce(u8::wrapping_add).unwrap()
      }
    }
  }

  #[inline]
  pub fn reduce_mul(self) -> u8 {
    pick! {
      if #[cfg(all(target_feature="neon", target_arch="aarch64"))] {
        const HIGH_64: [u8; 16] = [8, 9, 10, 11, 12, 13, 14, 15, 0, 0, 0, 0, 0, 0, 0, 0];
        const HIGH_32: [u8; 16] = [4, 5, 6, 7, 0, 1, 2, 3, 0, 0, 0, 0, 0, 0, 0, 0];
        const HIGH_16: [u8; 16] = [2, 3, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0];
        const HIGH_8: [u8; 16] = [1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0];

        unsafe {
          // Use `transmute` instead of `cast` because `int8x16_t` does not
          // implement `bytemuck::Pod`.
          let high_64 = vqtbl1q_u8(self.0, core::mem::transmute(HIGH_64));
          let reduce_64 = vmulq_u8(self.0, high_64);
          let high_32 = vqtbl1q_u8(reduce_64, core::mem::transmute(HIGH_32));
          let reduce_32 = vmulq_u8(reduce_64, high_32);
          let high_16 = vqtbl1q_u8(reduce_32, core::mem::transmute(HIGH_16));
          let reduce_16 = vmulq_u8(reduce_32, high_16);
          let high_8 = vqtbl1q_u8(reduce_16, core::mem::transmute(HIGH_8));
          let reduce_8 = vmulq_u8(reduce_16, high_8);
          vgetq_lane_u8::<0>(reduce_8)
        }
      } else {
        self.to_array().into_iter().reduce(u8::wrapping_mul).unwrap()
      }
    }
  }

  #[inline]
  pub fn reduce_max(self) -> u8 {
    #[allow(dead_code)]
    const SHUFFLE_1: [i8; 16] =
      [8, 9, 10, 11, 12, 13, 14, 15, 0, 0, 0, 0, 0, 0, 0, 0];
    #[allow(dead_code)]
    const SHUFFLE_2: [i8; 16] =
      [4, 5, 6, 7, 0, 1, 2, 3, 0, 0, 0, 0, 0, 0, 0, 0];
    #[allow(dead_code)]
    const SHUFFLE_3: [i8; 16] =
      [2, 3, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0];
    #[allow(dead_code)]
    const SHUFFLE_4: [i8; 16] =
      [1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0];

    pick! {
      if #[cfg(target_feature="ssse3")] {
        let rhs = shuffle_av_i8z_all_m128i(self.0, m128i::from(SHUFFLE_1));
        let max = max_u8_m128i(self.0, rhs);
        let rhs = shuffle_av_i8z_all_m128i(max, m128i::from(SHUFFLE_2));
        let max = max_u8_m128i(max, rhs);
        let rhs = shuffle_av_i8z_all_m128i(max, m128i::from(SHUFFLE_3));
        let max = max_u8_m128i(max, rhs);
        let rhs = shuffle_av_i8z_all_m128i(max, m128i::from(SHUFFLE_4));
        let max = max_u8_m128i(max, rhs);
        get_i32_from_m128i_s(max) as u8
      } else if #[cfg(target_feature="simd128")] {
        let rhs = u8x16_shuffle::<8, 9, 10, 11, 12, 13, 14, 15, 0, 1, 2, 3, 4, 5, 6, 7>(self.0, self.0);
        let max = u8x16_max(self.0, rhs);
        let rhs = u8x16_shuffle::<4, 5, 6, 7, 0, 1, 2, 3, 0, 0, 0, 0, 0, 0, 0, 0>(max, max);
        let max = u8x16_max(max, rhs);
        let rhs = u8x16_shuffle::<2, 3, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0>(max, max);
        let max = u8x16_max(max, rhs);
        let rhs = u8x16_shuffle::<1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0>(max, max);
        let max = u8x16_max(max, rhs);
        u8x16_extract_lane::<0>(max)
      } else if #[cfg(all(target_feature="neon",target_arch="aarch64"))]{
        unsafe {
          // Use `transmute` instead of `cast` because `uint8x16_t` does not
          // implement `bytemuck::Pod`.
          let rhs = vqtbl1q_u8(self.0, core::mem::transmute(SHUFFLE_1));
          let max = vmaxq_u8(self.0, rhs);
          let rhs = vqtbl1q_u8(max, core::mem::transmute(SHUFFLE_2));
          let max = vmaxq_u8(max, rhs);
          let rhs = vqtbl1q_u8(max, core::mem::transmute(SHUFFLE_3));
          let max = vmaxq_u8(max, rhs);
          let rhs = vqtbl1q_u8(max, core::mem::transmute(SHUFFLE_4));
          let max = vmaxq_u8(max, rhs);
          vgetq_lane_u8(max, 0)
        }
      } else {
        let array: [u8; 16] = cast(self);
        array.into_iter().reduce(u8::max).unwrap()
      }
    }
  }

  #[inline]
  pub fn reduce_min(self) -> u8 {
    #[allow(dead_code)]
    const SHUFFLE_1: [i8; 16] =
      [8, 9, 10, 11, 12, 13, 14, 15, 0, 0, 0, 0, 0, 0, 0, 0];
    #[allow(dead_code)]
    const SHUFFLE_2: [i8; 16] =
      [4, 5, 6, 7, 0, 1, 2, 3, 0, 0, 0, 0, 0, 0, 0, 0];
    #[allow(dead_code)]
    const SHUFFLE_3: [i8; 16] =
      [2, 3, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0];
    #[allow(dead_code)]
    const SHUFFLE_4: [i8; 16] =
      [1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0];

    pick! {
      if #[cfg(target_feature="ssse3")] {
        let rhs = shuffle_av_i8z_all_m128i(self.0, m128i::from(SHUFFLE_1));
        let min = min_u8_m128i(self.0, rhs);
        let rhs = shuffle_av_i8z_all_m128i(min, m128i::from(SHUFFLE_2));
        let min = min_u8_m128i(min, rhs);
        let rhs = shuffle_av_i8z_all_m128i(min, m128i::from(SHUFFLE_3));
        let min = min_u8_m128i(min, rhs);
        let rhs = shuffle_av_i8z_all_m128i(min, m128i::from(SHUFFLE_4));
        let min = min_u8_m128i(min, rhs);
        get_i32_from_m128i_s(min) as u8
      } else if #[cfg(target_feature="simd128")] {
        let rhs = u8x16_shuffle::<8, 9, 10, 11, 12, 13, 14, 15, 0, 1, 2, 3, 4, 5, 6, 7>(self.0, self.0);
        let min = u8x16_min(self.0, rhs);
        let rhs = u8x16_shuffle::<4, 5, 6, 7, 0, 1, 2, 3, 0, 0, 0, 0, 0, 0, 0, 0>(min, min);
        let min = u8x16_min(min, rhs);
        let rhs = u8x16_shuffle::<2, 3, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0>(min, min);
        let min = u8x16_min(min, rhs);
        let rhs = u8x16_shuffle::<1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0>(min, min);
        let min = u8x16_min(min, rhs);
        u8x16_extract_lane::<0>(min)
      } else if #[cfg(all(target_feature="neon",target_arch="aarch64"))]{
        unsafe {
          // Use `transmute` instead of `cast` because `uint8x16_t` does not
          // implement `bytemuck::Pod`.
          let rhs = vqtbl1q_u8(self.0, core::mem::transmute(SHUFFLE_1));
          let min = vminq_u8(self.0, rhs);
          let rhs = vqtbl1q_u8(min, core::mem::transmute(SHUFFLE_2));
          let min = vminq_u8(min, rhs);
          let rhs = vqtbl1q_u8(min, core::mem::transmute(SHUFFLE_3));
          let min = vminq_u8(min, rhs);
          let rhs = vqtbl1q_u8(min, core::mem::transmute(SHUFFLE_4));
          let min = vminq_u8(min, rhs);
          vgetq_lane_u8(min, 0)
        }
      } else {
        let array: [u8; 16] = cast(self);
        array.into_iter().reduce(u8::min).unwrap()
      }
    }
  }

  #[inline]
  pub fn unbounded_shl(self, rhs: Self) -> Self {
    // For x86, this technically can be done explicitly by converting to `u16`
    // or `u32` then converting back after multiplication, but that may not
    // actually be faster than auto-vectorization.
    pick! {
      if #[cfg(all(target_feature="neon", target_arch="aarch64"))] {
        unsafe {
          Self(vshlq_u8(self.0, vreinterpretq_s8_u8(rhs.0))) & rhs.simd_lt(8)
        }
      } else {
        let self_array = self.to_array();
        let rhs_array = rhs.to_array();

        Self::new([
          self_array[0].unbounded_shl(rhs_array[0] as u32),
          self_array[1].unbounded_shl(rhs_array[1] as u32),
          self_array[2].unbounded_shl(rhs_array[2] as u32),
          self_array[3].unbounded_shl(rhs_array[3] as u32),
          self_array[4].unbounded_shl(rhs_array[4] as u32),
          self_array[5].unbounded_shl(rhs_array[5] as u32),
          self_array[6].unbounded_shl(rhs_array[6] as u32),
          self_array[7].unbounded_shl(rhs_array[7] as u32),
          self_array[8].unbounded_shl(rhs_array[8] as u32),
          self_array[9].unbounded_shl(rhs_array[9] as u32),
          self_array[10].unbounded_shl(rhs_array[10] as u32),
          self_array[11].unbounded_shl(rhs_array[11] as u32),
          self_array[12].unbounded_shl(rhs_array[12] as u32),
          self_array[13].unbounded_shl(rhs_array[13] as u32),
          self_array[14].unbounded_shl(rhs_array[14] as u32),
          self_array[15].unbounded_shl(rhs_array[15] as u32),
        ])
      }
    }
  }

  #[inline]
  pub fn unbounded_shl_scalar(self, rhs: u32) -> Self {
    // For x86, this technically can be done explicitly by converting
    // to `u16` or `u32` then converting back after multiplication, but that
    // may not actually be faster than auto-vectorization.
    pick! {
      if #[cfg(target_feature="simd128")] {
        // The intrinsic performs wrapping shift so we need to mask the result.
        if rhs >= 8 { Self::ZERO } else { Self(u8x16_shl(self.0, rhs)) }
      } else if #[cfg(all(target_feature="neon",target_arch="aarch64"))]{
        // The intrinsic has different semantics so we need to saturate `rhs`.
        unsafe { Self(vshlq_u8(self.0, vmovq_n_s8(rhs.min(i8::MAX as u32) as i8))) }
      } else {
        let self_array = self.to_array();

        cast([
          self_array[0].unbounded_shl(rhs),
          self_array[1].unbounded_shl(rhs),
          self_array[2].unbounded_shl(rhs),
          self_array[3].unbounded_shl(rhs),
          self_array[4].unbounded_shl(rhs),
          self_array[5].unbounded_shl(rhs),
          self_array[6].unbounded_shl(rhs),
          self_array[7].unbounded_shl(rhs),
          self_array[8].unbounded_shl(rhs),
          self_array[9].unbounded_shl(rhs),
          self_array[10].unbounded_shl(rhs),
          self_array[11].unbounded_shl(rhs),
          self_array[12].unbounded_shl(rhs),
          self_array[13].unbounded_shl(rhs),
          self_array[14].unbounded_shl(rhs),
          self_array[15].unbounded_shl(rhs),
        ])
      }
    }
  }

  #[inline]
  pub fn unbounded_shr(self, rhs: Self) -> Self {
    // For x86, this technically can be done explicitly by converting
    // to `u16` or `u32` then converting back after multiplication, but that may
    // not actually be faster than auto-vectorization.
    pick! {
      if #[cfg(all(target_feature="neon", target_arch="aarch64"))] {
        unsafe {
          // Negate `rhs` because there is no direct shift-right intrinsic, and
          // mask to hide `rhs` overflow.
          Self(vshlq_u8(self.0, vnegq_s8(vreinterpretq_s8_u8(rhs.0)))) & rhs.simd_lt(8)
        }
      } else {
        let self_array = self.to_array();
        let rhs_array = rhs.to_array();

        Self::new([
          self_array[0].unbounded_shr(rhs_array[0] as u32),
          self_array[1].unbounded_shr(rhs_array[1] as u32),
          self_array[2].unbounded_shr(rhs_array[2] as u32),
          self_array[3].unbounded_shr(rhs_array[3] as u32),
          self_array[4].unbounded_shr(rhs_array[4] as u32),
          self_array[5].unbounded_shr(rhs_array[5] as u32),
          self_array[6].unbounded_shr(rhs_array[6] as u32),
          self_array[7].unbounded_shr(rhs_array[7] as u32),
          self_array[8].unbounded_shr(rhs_array[8] as u32),
          self_array[9].unbounded_shr(rhs_array[9] as u32),
          self_array[10].unbounded_shr(rhs_array[10] as u32),
          self_array[11].unbounded_shr(rhs_array[11] as u32),
          self_array[12].unbounded_shr(rhs_array[12] as u32),
          self_array[13].unbounded_shr(rhs_array[13] as u32),
          self_array[14].unbounded_shr(rhs_array[14] as u32),
          self_array[15].unbounded_shr(rhs_array[15] as u32),
        ])
      }
    }
  }

  #[inline]
  pub fn unbounded_shr_scalar(self, rhs: u32) -> Self {
    // For x86, this technically can be done explicitly by converting
    // to `u16` or `u32` then converting back after multiplication, but that
    // may not actually be faster than auto-vectorization.
    pick! {
      if #[cfg(target_feature="simd128")] {
        if rhs < 8 { Self(u8x16_shr(self.0, rhs)) } else { Self::ZERO }
      } else if #[cfg(all(target_feature="neon",target_arch="aarch64"))]{
        unsafe {
          // Negate `rhs` because there is no direct shift-right intrinsic, and
          // restrict it to prevent overflow.
          Self(vshlq_u8(self.0, vmovq_n_s8(-rhs.min(8).cast_signed() as i8)))
        }
      } else {
        let self_array = self.to_array();

        cast([
          self_array[0].unbounded_shr(rhs),
          self_array[1].unbounded_shr(rhs),
          self_array[2].unbounded_shr(rhs),
          self_array[3].unbounded_shr(rhs),
          self_array[4].unbounded_shr(rhs),
          self_array[5].unbounded_shr(rhs),
          self_array[6].unbounded_shr(rhs),
          self_array[7].unbounded_shr(rhs),
          self_array[8].unbounded_shr(rhs),
          self_array[9].unbounded_shr(rhs),
          self_array[10].unbounded_shr(rhs),
          self_array[11].unbounded_shr(rhs),
          self_array[12].unbounded_shr(rhs),
          self_array[13].unbounded_shr(rhs),
          self_array[14].unbounded_shr(rhs),
          self_array[15].unbounded_shr(rhs),
        ])
      }
    }
  }

  #[inline]
  pub fn saturating_add(self, rhs: Self) -> Self {
    pick! {
      if #[cfg(target_feature="sse2")] {
        Self(add_saturating_u8_m128i(self.0, rhs.0))
      } else if #[cfg(target_feature="simd128")] {
        Self(u8x16_add_sat(self.0, rhs.0))
      } else if #[cfg(all(target_feature="neon",target_arch="aarch64"))]{
        unsafe { Self(vqaddq_u8(self.0, rhs.0)) }
      } else {
        Self(Inner([
          self.0.0[0].saturating_add(rhs.0.0[0]),
          self.0.0[1].saturating_add(rhs.0.0[1]),
          self.0.0[2].saturating_add(rhs.0.0[2]),
          self.0.0[3].saturating_add(rhs.0.0[3]),
          self.0.0[4].saturating_add(rhs.0.0[4]),
          self.0.0[5].saturating_add(rhs.0.0[5]),
          self.0.0[6].saturating_add(rhs.0.0[6]),
          self.0.0[7].saturating_add(rhs.0.0[7]),
          self.0.0[8].saturating_add(rhs.0.0[8]),
          self.0.0[9].saturating_add(rhs.0.0[9]),
          self.0.0[10].saturating_add(rhs.0.0[10]),
          self.0.0[11].saturating_add(rhs.0.0[11]),
          self.0.0[12].saturating_add(rhs.0.0[12]),
          self.0.0[13].saturating_add(rhs.0.0[13]),
          self.0.0[14].saturating_add(rhs.0.0[14]),
          self.0.0[15].saturating_add(rhs.0.0[15]),
        ]))
      }
    }
  }

  #[inline]
  pub fn saturating_sub(self, rhs: Self) -> Self {
    pick! {
      if #[cfg(target_feature="sse2")] {
        Self(sub_saturating_u8_m128i(self.0, rhs.0))
      } else if #[cfg(target_feature="simd128")] {
        Self(u8x16_sub_sat(self.0, rhs.0))
      } else if #[cfg(all(target_feature="neon",target_arch="aarch64"))]{
        unsafe { Self(vqsubq_u8(self.0, rhs.0)) }
      } else {
        Self(Inner([
          self.0.0[0].saturating_sub(rhs.0.0[0]),
          self.0.0[1].saturating_sub(rhs.0.0[1]),
          self.0.0[2].saturating_sub(rhs.0.0[2]),
          self.0.0[3].saturating_sub(rhs.0.0[3]),
          self.0.0[4].saturating_sub(rhs.0.0[4]),
          self.0.0[5].saturating_sub(rhs.0.0[5]),
          self.0.0[6].saturating_sub(rhs.0.0[6]),
          self.0.0[7].saturating_sub(rhs.0.0[7]),
          self.0.0[8].saturating_sub(rhs.0.0[8]),
          self.0.0[9].saturating_sub(rhs.0.0[9]),
          self.0.0[10].saturating_sub(rhs.0.0[10]),
          self.0.0[11].saturating_sub(rhs.0.0[11]),
          self.0.0[12].saturating_sub(rhs.0.0[12]),
          self.0.0[13].saturating_sub(rhs.0.0[13]),
          self.0.0[14].saturating_sub(rhs.0.0[14]),
          self.0.0[15].saturating_sub(rhs.0.0[15]),
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
    pub fn widening_mul(self, rhs: Self) -> u16x16 {
      pick! {
        if #[cfg(all(target_feature="neon", target_arch="aarch64"))] {
          unsafe {
            let low_wide_mul = vmull_u8(vget_low_u8(self.0), vget_low_u8(rhs.0));
            let high_wide_mul = vmull_u8(vget_high_u8(self.0), vget_high_u8(rhs.0));

            u16x16(crate::u16x16_::Inner(u16x8(low_wide_mul), u16x8(high_wide_mul)))
          }
        } else {
          let self_array = self.to_array();
          let rhs_array = rhs.to_array();

          u16x16::new([
            (self_array[0] as u16).wrapping_mul(rhs_array[0] as u16),
            (self_array[1] as u16).wrapping_mul(rhs_array[1] as u16),
            (self_array[2] as u16).wrapping_mul(rhs_array[2] as u16),
            (self_array[3] as u16).wrapping_mul(rhs_array[3] as u16),
            (self_array[4] as u16).wrapping_mul(rhs_array[4] as u16),
            (self_array[5] as u16).wrapping_mul(rhs_array[5] as u16),
            (self_array[6] as u16).wrapping_mul(rhs_array[6] as u16),
            (self_array[7] as u16).wrapping_mul(rhs_array[7] as u16),
            (self_array[8] as u16).wrapping_mul(rhs_array[8] as u16),
            (self_array[9] as u16).wrapping_mul(rhs_array[9] as u16),
            (self_array[10] as u16).wrapping_mul(rhs_array[10] as u16),
            (self_array[11] as u16).wrapping_mul(rhs_array[11] as u16),
            (self_array[12] as u16).wrapping_mul(rhs_array[12] as u16),
            (self_array[13] as u16).wrapping_mul(rhs_array[13] as u16),
            (self_array[14] as u16).wrapping_mul(rhs_array[14] as u16),
            (self_array[15] as u16).wrapping_mul(rhs_array[15] as u16),
          ])
        }
      }
    }
  }

  #[inline]
  pub fn mul_keep_low_high(self, rhs: Self) -> (Self, Self) {
    pick! {
      if #[cfg(all(target_feature="neon", target_arch="aarch64"))] {
        unsafe {
          let low_wide_mul = vreinterpretq_u8_u16(
            vmull_u8(vget_low_u8(self.0), vget_low_u8(rhs.0)),
          );
          let high_wide_mul = vreinterpretq_u8_u16(
            vmull_u8(vget_high_u8(self.0), vget_high_u8(rhs.0)),
          );
          let low_high = vuzpq_u8(low_wide_mul, high_wide_mul);
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
          (self_array[0] as u16).wrapping_mul(rhs_array[0] as u16),
          (self_array[1] as u16).wrapping_mul(rhs_array[1] as u16),
          (self_array[2] as u16).wrapping_mul(rhs_array[2] as u16),
          (self_array[3] as u16).wrapping_mul(rhs_array[3] as u16),
          (self_array[4] as u16).wrapping_mul(rhs_array[4] as u16),
          (self_array[5] as u16).wrapping_mul(rhs_array[5] as u16),
          (self_array[6] as u16).wrapping_mul(rhs_array[6] as u16),
          (self_array[7] as u16).wrapping_mul(rhs_array[7] as u16),
          (self_array[8] as u16).wrapping_mul(rhs_array[8] as u16),
          (self_array[9] as u16).wrapping_mul(rhs_array[9] as u16),
          (self_array[10] as u16).wrapping_mul(rhs_array[10] as u16),
          (self_array[11] as u16).wrapping_mul(rhs_array[11] as u16),
          (self_array[12] as u16).wrapping_mul(rhs_array[12] as u16),
          (self_array[13] as u16).wrapping_mul(rhs_array[13] as u16),
          (self_array[14] as u16).wrapping_mul(rhs_array[14] as u16),
          (self_array[15] as u16).wrapping_mul(rhs_array[15] as u16),
        ];

        (
          Self::new([
            widening_mul[0] as u8,
            widening_mul[1] as u8,
            widening_mul[2] as u8,
            widening_mul[3] as u8,
            widening_mul[4] as u8,
            widening_mul[5] as u8,
            widening_mul[6] as u8,
            widening_mul[7] as u8,
            widening_mul[8] as u8,
            widening_mul[9] as u8,
            widening_mul[10] as u8,
            widening_mul[11] as u8,
            widening_mul[12] as u8,
            widening_mul[13] as u8,
            widening_mul[14] as u8,
            widening_mul[15] as u8,
          ]),
          Self::new([
            (widening_mul[0] >> 8) as u8,
            (widening_mul[1] >> 8) as u8,
            (widening_mul[2] >> 8) as u8,
            (widening_mul[3] >> 8) as u8,
            (widening_mul[4] >> 8) as u8,
            (widening_mul[5] >> 8) as u8,
            (widening_mul[6] >> 8) as u8,
            (widening_mul[7] >> 8) as u8,
            (widening_mul[8] >> 8) as u8,
            (widening_mul[9] >> 8) as u8,
            (widening_mul[10] >> 8) as u8,
            (widening_mul[11] >> 8) as u8,
            (widening_mul[12] >> 8) as u8,
            (widening_mul[13] >> 8) as u8,
            (widening_mul[14] >> 8) as u8,
            (widening_mul[15] >> 8) as u8,
          ]),
        )
      }
    }
  }

  #[inline]
  pub fn mul_keep_high(self, rhs: Self) -> Self {
    pick! {
      if #[cfg(all(target_feature="neon", target_arch="aarch64"))] {
        unsafe {
          let low_wide_mul = vreinterpretq_u8_u16(
            vmull_u8(vget_low_u8(self.0), vget_low_u8(rhs.0)),
          );
          let high_wide_mul = vreinterpretq_u8_u16(
            vmull_u8(vget_high_u8(self.0), vget_high_u8(rhs.0)),
          );
          Self(vuzpq_u8(low_wide_mul, high_wide_mul).1)
        }
      } else {
        let self_array = self.to_array();
        let rhs_array = rhs.to_array();

        Self::new([
          ((self_array[0] as u16).wrapping_mul(rhs_array[0] as u16) >> 8) as u8,
          ((self_array[1] as u16).wrapping_mul(rhs_array[1] as u16) >> 8) as u8,
          ((self_array[2] as u16).wrapping_mul(rhs_array[2] as u16) >> 8) as u8,
          ((self_array[3] as u16).wrapping_mul(rhs_array[3] as u16) >> 8) as u8,
          ((self_array[4] as u16).wrapping_mul(rhs_array[4] as u16) >> 8) as u8,
          ((self_array[5] as u16).wrapping_mul(rhs_array[5] as u16) >> 8) as u8,
          ((self_array[6] as u16).wrapping_mul(rhs_array[6] as u16) >> 8) as u8,
          ((self_array[7] as u16).wrapping_mul(rhs_array[7] as u16) >> 8) as u8,
          ((self_array[8] as u16).wrapping_mul(rhs_array[8] as u16) >> 8) as u8,
          ((self_array[9] as u16).wrapping_mul(rhs_array[9] as u16) >> 8) as u8,
          ((self_array[10] as u16).wrapping_mul(rhs_array[10] as u16) >> 8) as u8,
          ((self_array[11] as u16).wrapping_mul(rhs_array[11] as u16) >> 8) as u8,
          ((self_array[12] as u16).wrapping_mul(rhs_array[12] as u16) >> 8) as u8,
          ((self_array[13] as u16).wrapping_mul(rhs_array[13] as u16) >> 8) as u8,
          ((self_array[14] as u16).wrapping_mul(rhs_array[14] as u16) >> 8) as u8,
          ((self_array[15] as u16).wrapping_mul(rhs_array[15] as u16) >> 8) as u8,
        ])
      }
    }
  }
}

/// The following functionality exists only for [`u8x16`], or only for
/// particular types inconsistently.
impl u8x16 {
  /// Returns `[lhs[0], rhs[0], lhs[1], rhs[1], ...]`, taking the first 8
  /// elements of each input and dropping their last 8 elements.
  #[inline]
  #[must_use]
  pub fn unpack_low(lhs: u8x16, rhs: u8x16) -> u8x16 {
    pick! {
        if #[cfg(target_feature = "sse2")] {
            Self(unpack_low_i8_m128i(lhs.0, rhs.0))
        } else if #[cfg(target_feature = "simd128")] {
          Self(u8x16_shuffle::<0, 16, 1, 17, 2, 18, 3, 19, 4, 20, 5, 21, 6, 22, 7, 23>(lhs.0, rhs.0))
        } else if #[cfg(all(target_feature = "neon", target_arch = "aarch64"))] {
            let lhs = unsafe { vget_low_u8(lhs.0) };
            let rhs = unsafe { vget_low_u8(rhs.0) };

            let zipped = unsafe { vzip_u8(lhs, rhs) };
            Self(unsafe { vcombine_u8(zipped.0, zipped.1) })
        } else {
            u8x16::new([
                lhs.as_array()[0], rhs.as_array()[0],
                lhs.as_array()[1], rhs.as_array()[1],
                lhs.as_array()[2], rhs.as_array()[2],
                lhs.as_array()[3], rhs.as_array()[3],
                lhs.as_array()[4], rhs.as_array()[4],
                lhs.as_array()[5], rhs.as_array()[5],
                lhs.as_array()[6], rhs.as_array()[6],
                lhs.as_array()[7], rhs.as_array()[7],
            ])
        }
    }
  }

  /// Returns `[lhs[8], rhs[8], lhs[9], rhs[9], ...]`, taking the last 8
  /// elements of each input and dropping their first 8 elements.
  #[inline]
  #[must_use]
  pub fn unpack_high(lhs: u8x16, rhs: u8x16) -> u8x16 {
    pick! {
        if #[cfg(target_feature = "sse2")] {
            Self(unpack_high_i8_m128i(lhs.0, rhs.0))
        } else if #[cfg(target_feature = "simd128")] {
            Self(u8x16_shuffle::<8, 24, 9, 25, 10, 26, 11, 27, 12, 28, 13, 29, 14, 30, 15, 31>(lhs.0, rhs.0))
        } else if #[cfg(all(target_feature = "neon", target_arch = "aarch64"))] {
            let lhs = unsafe { vget_high_u8(lhs.0) };
            let rhs = unsafe { vget_high_u8(rhs.0) };

            let zipped = unsafe { vzip_u8(lhs, rhs) };
            Self(unsafe { vcombine_u8(zipped.0, zipped.1) })
        } else {
            u8x16::new([
                lhs.as_array()[8], rhs.as_array()[8],
                lhs.as_array()[9], rhs.as_array()[9],
                lhs.as_array()[10], rhs.as_array()[10],
                lhs.as_array()[11], rhs.as_array()[11],
                lhs.as_array()[12], rhs.as_array()[12],
                lhs.as_array()[13], rhs.as_array()[13],
                lhs.as_array()[14], rhs.as_array()[14],
                lhs.as_array()[15], rhs.as_array()[15],
            ])
        }
    }
  }

  /// Treats two [`i16x8`] values as a single [`i16x16`] value, then converts
  /// each element from [`i16`] to [`u8`], saturating out of range values.
  #[inline]
  #[must_use]
  pub fn narrow_i16x8(lhs: i16x8, rhs: i16x8) -> Self {
    pick! {
        if #[cfg(target_feature = "sse2")] {
            Self(pack_i16_to_u8_m128i(lhs.0, rhs.0))
        } else if #[cfg(target_feature = "simd128")] {
            Self(u8x16_narrow_i16x8(lhs.0, rhs.0))
        } else if #[cfg(all(target_feature = "neon", target_arch = "aarch64"))] {
            let lhs = unsafe { vqmovun_s16(lhs.0) };
            let rhs = unsafe { vqmovun_s16(rhs.0) };
            Self(unsafe { vcombine_u8(lhs, rhs) })
        } else {
            fn clamp(a: i16) -> u8 {
                  if a < u8::MIN as i16 {
                      u8::MIN
                  } else if a > u8::MAX as i16 {
                      u8::MAX
                  } else {
                      a as u8
                  }
            }

            Self(Inner([
                clamp(lhs.as_array()[0]),
                clamp(lhs.as_array()[1]),
                clamp(lhs.as_array()[2]),
                clamp(lhs.as_array()[3]),
                clamp(lhs.as_array()[4]),
                clamp(lhs.as_array()[5]),
                clamp(lhs.as_array()[6]),
                clamp(lhs.as_array()[7]),
                clamp(rhs.as_array()[0]),
                clamp(rhs.as_array()[1]),
                clamp(rhs.as_array()[2]),
                clamp(rhs.as_array()[3]),
                clamp(rhs.as_array()[4]),
                clamp(rhs.as_array()[5]),
                clamp(rhs.as_array()[6]),
                clamp(rhs.as_array()[7]),
            ]))
        }
    }
  }

  /// Returns a new vector where each element is based on the index values in
  /// `rhs`.
  ///
  /// * Index values in the range `[0, 15]` select the i-th element of `self`.
  /// * Index values that are out of range will cause that output lane to be
  ///   `0`.
  #[inline]
  pub fn swizzle(self, rhs: i8x16) -> i8x16 {
    cast(i8x16::swizzle(cast(self), rhs))
  }

  /// Works like [`swizzle`](Self::swizzle) with the following additional
  /// details
  ///
  /// * Indices in the range `[0, 15]` will select the i-th element of `self`.
  /// * If the high bit of any index is set (meaning that the index is
  ///   negative), then the corresponding output lane is guaranteed to be zero.
  /// * Otherwise the output lane is either `0` or `self[rhs[i] % 16]`,
  ///   depending on the implementation.
  #[inline]
  pub fn swizzle_relaxed(self, rhs: u8x16) -> u8x16 {
    cast(i8x16::swizzle_relaxed(cast(self), cast(rhs)))
  }
}
