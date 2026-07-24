use super::*;

use crate::{i8x32, simd::SimdBackend, u8x16, u16x32};

#[cfg(not(target_feature = "avx2"))]
#[repr(C, align(32))]
#[derive(Default, Clone, Copy, PartialEq, Eq)]
pub(crate) struct Inner(pub u8x16, pub u8x16);

unsafe impl SimdBackend for u8x32 {
  pick! {
    if #[cfg(target_feature="avx2")] {
      type Inner = m256i;
    } else {
      type Inner = Inner;
    }
  }
}

impl_simd! {
  unsafe {
    T = u8,
    N = 32,
    Simd = u8x32,
    optional_type_x86_inner { X86Inner = __m256i },
    optional_type_arm_inner {},
    optional_type_wasm_inner {},
  }

  #[inline]
  fn simd_eq(self, rhs: Self) -> Self::Output {
    pick! {
      if #[cfg(target_feature="avx2")] {
        Self(cmp_eq_mask_i8_m256i(self.0,rhs.0))
      } else {
        Self(Inner(self.0.0.simd_eq(rhs.0.0), self.0.1.simd_eq(rhs.0.1)))
      }
    }
  }

  #[inline]
  fn simd_ne(self, rhs: Self) -> Self::Output {
    pick! {
      if #[cfg(target_feature="avx2")] {
        !self.simd_eq(rhs)
      } else {
        Self(Inner(self.0.0.simd_ne(rhs.0.0), self.0.1.simd_ne(rhs.0.1)))
      }
    }
  }

  #[inline]
  fn simd_lt(self, rhs: Self) -> Self::Output {
    pick! {
      if #[cfg(target_feature="avx2")] {
        // Convert from u8 to i8.
        let offset = Self::splat(0x80);
        let self_i8 = self.bitxor(offset).0;
        let rhs_i8 = rhs.bitxor(offset).0;
        Self(cmp_gt_mask_i8_m256i(rhs_i8, self_i8))
      } else {
        Self(Inner(self.0.0.simd_lt(rhs.0.0), self.0.1.simd_lt(rhs.0.1)))
      }
    }
  }

  #[inline]
  fn simd_gt(self, rhs: Self) -> Self::Output {
    pick! {
      if #[cfg(target_feature="avx2")] {
        // Convert from u8 to i8.
        let offset = Self::splat(0x80);
        let self_i8 = self.bitxor(offset).0;
        let rhs_i8 = rhs.bitxor(offset).0;
        Self(cmp_gt_mask_i8_m256i(self_i8,rhs_i8))
      } else {
        Self(Inner(self.0.0.simd_gt(rhs.0.0), self.0.1.simd_gt(rhs.0.1)))
      }
    }
  }

  #[inline]
  fn simd_le(self, rhs: Self) -> Self::Output {
    pick! {
      if #[cfg(target_feature="avx2")] {
        // Convert from u8 to i8.
        let offset = Self::splat(0x80);
        let self_i8 = self.bitxor(offset).0;
        let rhs_i8 = rhs.bitxor(offset).0;
        let gt_mask = Self(cmp_gt_mask_i8_m256i(self_i8,rhs_i8));
        Self(gt_mask.bitxor(Self::splat(0xFF)).0)
      } else {
        Self(Inner(self.0.0.simd_le(rhs.0.0), self.0.1.simd_le(rhs.0.1)))
      }
    }
  }

  #[inline]
  fn simd_ge(self, rhs: Self) -> Self::Output {
    pick! {
      if #[cfg(target_feature="avx2")] {
        // Convert from u8 to i8.
        let offset = Self::splat(0x80);
        let self_i8 = self.bitxor(offset).0;
        let rhs_i8 = rhs.bitxor(offset).0;
        let lt_mask = Self(cmp_gt_mask_i8_m256i(rhs_i8, self_i8));
        Self(lt_mask.bitxor(Self::splat(0xFF)).0)
      } else {
        Self(Inner(self.0.0.simd_ge(rhs.0.0), self.0.1.simd_ge(rhs.0.1)))
      }
    }
  }

  #[inline]
  pub fn bitselect(self, if_one: Self, if_zero: Self) -> Self {
    pick! {
      if #[cfg(target_feature="avx2")] {
        Self(bitor_m256i(
          bitand_m256i(if_one.0, self.0),
          bitandnot_m256i(self.0, if_zero.0),
        ))
      } else {
        Self(Inner(
          self.0.0.bitselect(if_one.0.0, if_zero.0.0),
          self.0.1.bitselect(if_one.0.1, if_zero.0.1),
        ))
      }
    }
  }

  #[inline]
  pub fn select(self, if_true: Self, if_false: Self) -> Self {
    pick! {
      if #[cfg(target_feature="avx2")] {
        Self(blend_varying_i8_m256i(if_false.0, if_true.0, self.0))
      } else {
        Self(Inner(
          self.0.0.select(if_true.0.0, if_false.0.0),
          self.0.1.select(if_true.0.1, if_false.0.1),
        ))
      }
    }
  }

  #[inline]
  pub fn to_bitmask(self) -> u32 {
    i8x32::to_bitmask(cast(self)) as u32
  }

  #[inline]
  pub fn any(self) -> bool {
    i8x32::any(cast(self))
  }

  #[inline]
  pub fn all(self) -> bool {
    i8x32::all(cast(self))
  }

  ///
  /// Currently this function is never accelerated.
  #[inline]
  pub fn transpose(data: [u8x32; 32]) -> [u8x32; 32] {
    cast(i8x32::transpose(cast(data)))
  }
}

impl_simd_uint! {
  unsafe {
    T = u8,
    N = 32,
    Simd = u8x32,
    SignedSimd = i8x32,
    T_BITS = 8,
    T_BITS_MUL_2 = 16,
    [
      0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20,
      21, 22, 23, 24, 25, 26, 27, 28, 29, 30, 31
    ],
  }

  #[inline]
  fn not(self) -> Self {
    pick! {
      if #[cfg(target_feature="avx2")] {
        Self(self.0.not())
      } else {
        Self(Inner(self.0.0.not(), self.0.1.not()))
      }
    }
  }

  #[inline]
  fn add(self, rhs: Self) -> Self::Output {
    pick! {
      if #[cfg(target_feature="avx2")] {
        Self(add_i8_m256i(self.0,rhs.0))
      } else {
        Self(Inner(self.0.0.add(rhs.0.0), self.0.1.add(rhs.0.1)))
      }
    }
  }

  #[inline]
  fn sub(self, rhs: Self) -> Self::Output {
    pick! {
      if #[cfg(target_feature="avx2")] {
        Self(sub_i8_m256i(self.0,rhs.0))
      } else {
        Self(Inner(self.0.0.sub(rhs.0.0), self.0.1.sub(rhs.0.1)))
      }
    }
  }

  #[inline]
  fn mul(self, rhs: Self) -> Self::Output {
    // For x86, this technically can be done explicitly by converting to `i16`
    // then converting back after multiplication, but that may not actually be
    // faster than auto-vectorization.
    let [self_a, self_b]: [u8x16; 2] = cast(self);
    let [rhs_a, rhs_b]: [u8x16; 2] = cast(rhs);
    cast([self_a * rhs_a, self_b * rhs_b])
  }

  #[inline]
  fn shl(self, rhs: Self) -> Self::Output {
    // For x86, this technically can be done explicitly by converting to `u16`
    // or `u32` then converting back after multiplication, but that may not
    // actually be faster than auto-vectorization.
    let [self_a, self_b]: [u8x16; 2] = cast(self);
    let [rhs_a, rhs_b]: [u8x16; 2] = cast(rhs);
    cast([self_a << rhs_a, self_b << rhs_b])
  }

  #[inline]
  fn shl(self, rhs: u32) -> Self::Output {
    // For x86, this technically can be done explicitly by converting
    // to `u16` or `u32` then converting back after multiplication, but that
    // may not actually be faster than auto-vectorization.
    let [self_a, self_b]: [u8x16; 2] = cast(self);
    cast([self_a << rhs, self_b << rhs])
  }

  #[inline]
  fn shr(self, rhs: Self) -> Self::Output {
    // For x86, this technically can be done explicitly by converting to `u16`
    // or `u32` then converting back after multiplication, but that may not
    // actually be faster than auto-vectorization.
    let [self_a, self_b]: [u8x16; 2] = cast(self);
    let [rhs_a, rhs_b]: [u8x16; 2] = cast(rhs);
    cast([self_a >> rhs_a, self_b >> rhs_b])
  }

  #[inline]
  fn shr(self, rhs: u32) -> Self::Output {
    // For x86, this technically can be done explicitly by converting
    // to `u16` or `u32` then converting back after multiplication, but that
    // may not actually be faster than auto-vectorization.
    let [self_a, self_b]: [u8x16; 2] = cast(self);
    cast([self_a >> rhs, self_b >> rhs])
  }

  #[inline]
  fn bitand(self, rhs: Self) -> Self::Output {
    pick! {
      if #[cfg(target_feature="avx2")] {
        Self(bitand_m256i(self.0,rhs.0))
      } else {
        Self(Inner(self.0.0.bitand(rhs.0.0), self.0.1.bitand(rhs.0.1)))
      }
    }
  }

  #[inline]
  fn bitor(self, rhs: Self) -> Self::Output {
    pick! {
      if #[cfg(target_feature="avx2")] {
        Self(bitor_m256i(self.0,rhs.0))
      } else {
        Self(Inner(self.0.0.bitor(rhs.0.0), self.0.1.bitor(rhs.0.1)))
      }
    }
  }

  #[inline]
  fn bitxor(self, rhs: Self) -> Self::Output {
    pick! {
      if #[cfg(target_feature="avx2")] {
        Self(bitxor_m256i(self.0,rhs.0))
      } else {
        Self(Inner(self.0.0.bitxor(rhs.0.0), self.0.1.bitxor(rhs.0.1)))
      }
    }
  }

  #[inline]
  pub fn max(self, rhs: Self) -> Self {
    pick! {
      if #[cfg(target_feature="avx2")] {
        Self(max_u8_m256i(self.0,rhs.0))
      } else {
        Self(Inner(self.0.0.max(rhs.0.0), self.0.1.max(rhs.0.1)))
      }
    }
  }

  #[inline]
  pub fn min(self, rhs: Self) -> Self {
    pick! {
      if #[cfg(target_feature="avx2")] {
        Self(min_u8_m256i(self.0,rhs.0))
      } else {
        Self(Inner(self.0.0.min(rhs.0.0), self.0.1.min(rhs.0.1)))
      }
    }
  }

  #[inline]
  pub fn reduce_add(self) -> u8 {
    let array: [u8x16; 2] = cast(self);
    (array[0] + array[1]).reduce_add()
  }

  #[inline]
  pub fn reduce_mul(self) -> u8 {
    let array: [u8x16; 2] = cast(self);
    (array[0] * array[1]).reduce_mul()
  }

  #[inline]
  pub fn reduce_max(self) -> u8 {
    let array: [u8x16; 2] = cast(self);
    array[0].max(array[1]).reduce_max()
  }

  #[inline]
  pub fn reduce_min(self) -> u8 {
    let array: [u8x16; 2] = cast(self);
    array[0].min(array[1]).reduce_min()
  }

  #[inline]
  pub fn unbounded_shl(self, rhs: Self) -> Self {
    // For x86, this technically can be done explicitly by converting to `u16`
    // or `u32` then converting back after multiplication, but that may not
    // actually be faster than auto-vectorization.
    let [self_a, self_b] = cast::<u8x32, [u8x16; 2]>(self);
    let [rhs_a, rhs_b] = cast::<u8x32, [u8x16; 2]>(rhs);
    cast([self_a.unbounded_shl(rhs_a), self_b.unbounded_shl(rhs_b)])
  }

  #[inline]
  pub fn unbounded_shl_scalar(self, rhs: u32) -> Self {
    // For x86, this technically can be done explicitly by converting
    // to `u16` or `u32` then converting back after multiplication, but that
    // may not actually be faster than auto-vectorization.
    let [self_a, self_b] = cast::<u8x32, [u8x16; 2]>(self);
    cast([self_a.unbounded_shl_scalar(rhs), self_b.unbounded_shl_scalar(rhs)])
  }

  #[inline]
  pub fn unbounded_shr(self, rhs: Self) -> Self {
    // For x86, this technically can be done explicitly by converting to `u16`
    // or `u32` then converting back after multiplication, but that may not
    // actually be faster than auto-vectorization.
    let [self_a, self_b] = cast::<u8x32, [u8x16; 2]>(self);
    let [rhs_a, rhs_b] = cast::<u8x32, [u8x16; 2]>(rhs);
    cast([self_a.unbounded_shr(rhs_a), self_b.unbounded_shr(rhs_b)])
  }

  #[inline]
  pub fn unbounded_shr_scalar(self, rhs: u32) -> Self {
    // For x86, this technically can be done explicitly by converting
    // to `u16` or `u32` then converting back after multiplication, but that
    // may not actually be faster than auto-vectorization.
    let [self_a, self_b] = cast::<u8x32, [u8x16; 2]>(self);
    cast([self_a.unbounded_shr_scalar(rhs), self_b.unbounded_shr_scalar(rhs)])
  }

  #[inline]
  pub fn saturating_add(self, rhs: Self) -> Self {
    pick! {
      if #[cfg(target_feature="avx2")] {
        Self(add_saturating_u8_m256i(self.0, rhs.0))
      } else {
        Self(Inner(self.0.0.saturating_add(rhs.0.0), self.0.1.saturating_add(rhs.0.1)))
      }
    }
  }

  #[inline]
  pub fn saturating_sub(self, rhs: Self) -> Self {
    pick! {
      if #[cfg(target_feature="avx2")] {
        Self(sub_saturating_u8_m256i(self.0, rhs.0))
      } else {
        Self(Inner(
          self.0.0.saturating_sub(rhs.0.0),
          self.0.1.saturating_sub(rhs.0.1),
        ))
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
    pub fn widening_mul(self, rhs: Self) -> u16x32 {
      // x86 has no `_mm256_mul_epu8` intrinsic so there is no `avx2`
      // optimization.

      let [self_a, self_b] = cast::<u8x32, [u8x16; 2]>(self);
      let [rhs_a, rhs_b] = cast::<u8x32, [u8x16; 2]>(rhs);

      cast([self_a.widening_mul(rhs_a), self_b.widening_mul(rhs_b)])
    }
  }

  #[inline]
  pub fn mul_keep_low_high(self, rhs: Self) -> (Self, Self) {
    // x86 has no `_mm256_mul_epu8` intrinsic so there is no `avx2`
    // optimization.

    let [self_a, self_b] = cast::<u8x32, [u8x16; 2]>(self);
    let [rhs_a, rhs_b] = cast::<u8x32, [u8x16; 2]>(rhs);

    let result_a = self_a.mul_keep_low_high(rhs_a);
    let result_b = self_b.mul_keep_low_high(rhs_b);
    (cast([result_a.0, result_b.0]), cast([result_a.1, result_b.1]))
  }

  #[inline]
  pub fn mul_keep_high(self, rhs: Self) -> Self {
    // x86 has no `_mm256_mul_epu8` intrinsic so there is no `avx2`
    // optimization.

    let [self_a, self_b] = cast::<u8x32, [u8x16; 2]>(self);
    let [rhs_a, rhs_b] = cast::<u8x32, [u8x16; 2]>(rhs);

    cast([self_a.mul_keep_high(rhs_a), self_b.mul_keep_high(rhs_b)])
  }
}

/// The following functionality exists only for [`u8x32`], or only for
/// particular types inconsistently.
impl u8x32 {
  /// Returns a new vector with lanes selected from the lanes of the first input
  /// vector a specified in the second input vector `rhs`.
  /// The indices i in range `[0, 15]` select the i-th element of `self`. For
  /// indices outside of the range the resulting lane is `0`.
  ///
  /// This note that is the equivalent of two parallel swizzle operations on the
  /// two halves of the vector, and the indexes each refer to the
  /// corresponding half.
  #[inline]
  pub fn swizzle_half(self, rhs: i8x32) -> i8x32 {
    cast(i8x32::swizzle_half(cast(self), cast(rhs)))
  }

  /// Indices in the range `[0, 15]` will select the i-th element of `self`. If
  /// the high bit of any element of `rhs` is set (negative) then the
  /// corresponding output lane is guaranteed to be zero. Otherwise if the
  /// element of `rhs` is within the range `[32, 127]` then the output lane is
  /// either `0` or `self[rhs[i] % 16]` depending on the implementation.
  ///
  /// This is the equivalent to two parallel swizzle operations on the two
  /// halves of the vector, and the indexes each refer to their corresponding
  /// half.
  #[inline]
  pub fn swizzle_half_relaxed(self, rhs: u8x32) -> u8x32 {
    cast(i8x32::swizzle_half_relaxed(cast(self), cast(rhs)))
  }

  /// Full 32-entry byte table lookup. An index in `[0, 31]` selects
  /// `self[index]`; any index `>= 32` yields `0`.
  #[inline]
  pub fn swizzle(self, rhs: u8x32) -> u8x32 {
    cast(i8x32::swizzle(cast(self), cast(rhs)))
  }

  /// Like [`swizzle`](Self::swizzle), but out-of-range indices yield an
  /// implementation-defined result (`0` or `self[index % 32]`).
  #[inline]
  pub fn swizzle_relaxed(self, rhs: u8x32) -> u8x32 {
    cast(i8x32::swizzle_relaxed(cast(self), cast(rhs)))
  }
}
