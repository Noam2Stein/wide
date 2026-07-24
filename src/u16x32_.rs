use super::*;

use crate::{i16x32, simd::SimdBackend, u16x16};

#[cfg(not(target_feature = "avx512bw"))]
#[repr(C, align(64))]
#[derive(Default, Clone, Copy, PartialEq, Eq)]
pub(crate) struct Inner(pub u16x16, pub u16x16);

unsafe impl SimdBackend for u16x32 {
  pick! {
    if #[cfg(target_feature="avx512bw")] {
      type Inner = m512i;
    } else {
      type Inner = Inner;
    }
  }
}

impl_simd! {
  unsafe {
    T = u16,
    N = 32,
    Simd = u16x32,
    optional_type_x86_inner { X86Inner = __m512i },
    optional_type_arm_inner {},
    optional_type_wasm_inner {},
  }

  #[inline]
  fn simd_eq(self, rhs: Self) -> Self::Output {
    pick! {
      if #[cfg(target_feature="avx512bw")] {
        Self(cmp_op_mask_u16_m512i::<{cmp_int_op!(Eq)}>(self.0, rhs.0))
      } else {
        Self(Inner(self.0.0.simd_eq(rhs.0.0), self.0.1.simd_eq(rhs.0.1)))
      }
    }
  }

  #[inline]
  fn simd_ne(self, rhs: Self) -> Self::Output {
    pick! {
      if #[cfg(target_feature="avx512bw")] {
        Self(cmp_op_mask_u16_m512i::<{cmp_int_op!(Ne)}>(self.0, rhs.0))
      } else {
        Self(Inner(self.0.0.simd_ne(rhs.0.0), self.0.1.simd_ne(rhs.0.1)))
      }
    }
  }

  #[inline]
  fn simd_lt(self, rhs: Self) -> Self::Output {
    pick! {
      if #[cfg(target_feature="avx512bw")] {
        Self(cmp_op_mask_u16_m512i::<{cmp_int_op!(Lt)}>(self.0, rhs.0))
      } else {
        Self(Inner(rhs.0.0.simd_gt(self.0.0), rhs.0.1.simd_gt(self.0.1)))
      }
    }
  }

  #[inline]
  fn simd_gt(self, rhs: Self) -> Self::Output {
    pick! {
      if #[cfg(target_feature="avx512bw")] {
        Self(cmp_op_mask_u16_m512i::<{cmp_int_op!(Nle)}>(self.0, rhs.0))
      } else {
        Self(Inner(self.0.0.simd_gt(rhs.0.0), self.0.1.simd_gt(rhs.0.1)))
      }
    }
  }

  #[inline]
  fn simd_le(self, rhs: Self) -> Self::Output {
    pick! {
      if #[cfg(target_feature="avx512bw")] {
        Self(cmp_op_mask_u16_m512i::<{cmp_int_op!(Le)}>(self.0, rhs.0))
      } else {
        Self(Inner(self.0.0.simd_le(rhs.0.0), self.0.1.simd_le(rhs.0.1)))
      }
    }
  }

  #[inline]
  fn simd_ge(self, rhs: Self) -> Self::Output {
    pick! {
      if #[cfg(target_feature="avx512bw")] {
        Self(cmp_op_mask_u16_m512i::<{cmp_int_op!(Nlt)}>(self.0, rhs.0))
      } else {
        Self(Inner(self.0.0.simd_ge(rhs.0.0), self.0.1.simd_ge(rhs.0.1)))
      }
    }
  }

  #[inline]
  pub fn bitselect(self, if_one: Self, if_zero: Self) -> Self {
    pick! {
      if #[cfg(target_feature="avx512bw")] {
        Self(bitor_m512i(
          bitand_m512i(if_one.0, self.0),
          bitandnot_m512i(self.0, if_zero.0),
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
      if #[cfg(target_feature="avx512bw")] {
        Self(blend_varying_i8_m512i(if_false.0,if_true.0,movepi8_mask_m512i(self.0)))
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
    i16x32::to_bitmask(cast(self))
  }

  #[inline]
  pub fn any(self) -> bool {
    i16x32::any(cast(self))
  }

  #[inline]
  pub fn all(self) -> bool {
    i16x32::all(cast(self))
  }

  ///
  /// Currently this function is never accelerated.
  #[inline]
  pub fn transpose(data: [u16x32; 32]) -> [u16x32; 32] {
    cast(i16x32::transpose(cast(data)))
  }
}

impl_simd_uint! {
  unsafe {
    T = u16,
    N = 32,
    Simd = u16x32,
    SignedSimd = i16x32,
    T_BITS = 16,
    T_BITS_MUL_2 = 32,
    [
      0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20,
      21, 22, 23, 24, 25, 26, 27, 28, 29, 30, 31
    ],
  }

  #[inline]
  fn not(self) -> Self::Output {
    pick! {
      if #[cfg(target_feature="avx512bw")] {
        Self(bitxor_m512i(self.0, set_splat_i16_m512i(-1)))
      } else {
        Self(Inner(self.0.0.not(), self.0.1.not()))
      }
    }
  }

  #[inline]
  fn add(self, rhs: Self) -> Self::Output {
    pick! {
      if #[cfg(target_feature="avx512bw")] {
        Self(add_i16_m512i(self.0, rhs.0))
      } else {
        Self(Inner(self.0.0.add(rhs.0.0), self.0.1.add(rhs.0.1)))
      }
    }
  }

  #[inline]
  fn sub(self, rhs: Self) -> Self::Output {
    pick! {
      if #[cfg(target_feature="avx512bw")] {
        Self(sub_i16_m512i(self.0, rhs.0))
      } else {
        Self(Inner(self.0.0.sub(rhs.0.0), self.0.1.sub(rhs.0.1)))
      }
    }
  }

  #[inline]
  fn mul(self, rhs: Self) -> Self::Output {
    pick! {
      if #[cfg(target_feature="avx512bw")] {
        Self(mul_i16_keep_low_m512i(self.0, rhs.0))
      } else {
        Self(Inner(self.0.0.mul(rhs.0.0), self.0.1.mul(rhs.0.1)))
      }
    }
  }

  #[inline]
  fn shl(self, rhs: Self) -> Self::Output {
    pick! {
      if #[cfg(target_feature="avx512bw")] {
        // Mask `rhs` to 15 to match `wrapping_shl`.
        let rhs = bitand_m512i(rhs.0, set_splat_i16_m512i(15));
        Self(shl_each_u16_m512i(self.0, rhs))
      } else {
        let [self_a, self_b]: [u16x16; 2] = cast(self);
        let [rhs_a, rhs_b]: [u16x16; 2] = cast(rhs);

        cast([self_a << rhs_a, self_b << rhs_b])
      }
    }
  }

  #[inline]
  fn shl(self, rhs: u32) -> Self::Output {
    pick! {
      if #[cfg(target_feature="avx512bw")] {
        // Use `rhs % 16` to perform wrapping shift and not unbounded shift.
        #[expect(clippy::suspicious_arithmetic_impl)]
        let shift = rhs as u16 & 15;
        Self(shl_all_u16_m512i(self.0, shift))
      } else {
        Self(Inner(self.0.0.shl(rhs), self.0.1.shl(rhs)))
      }
    }
  }

  #[inline]
  fn shr(self, rhs: Self) -> Self::Output {
    pick! {
      if #[cfg(target_feature="avx512bw")] {
        // Mask `rhs` to 15 to match `wrapping_shr`.
        let rhs = bitand_m512i(rhs.0, set_splat_i16_m512i(15));
        Self(shr_each_u16_m512i(self.0, rhs))
      } else {
        let [self_a, self_b]: [u16x16; 2] = cast(self);
        let [rhs_a, rhs_b]: [u16x16; 2] = cast(rhs);

        cast([self_a >> rhs_a, self_b >> rhs_b])
      }
    }
  }

  #[inline]
  fn shr(self, rhs: u32) -> Self::Output {
    pick! {
      if #[cfg(target_feature="avx512bw")] {
        // Use `rhs % 16` to perform wrapping shift and not unbounded shift.
        #[expect(clippy::suspicious_arithmetic_impl)]
        let shift = rhs as u16 & 15;
        Self(shr_all_u16_m512i(self.0, shift))
      } else {
        Self(Inner(self.0.0.shr(rhs), self.0.1.shr(rhs)))
      }
    }
  }

  #[inline]
  fn bitand(self, rhs: Self) -> Self::Output {
    pick! {
      if #[cfg(target_feature="avx512bw")] {
        Self(bitand_m512i(self.0, rhs.0))
      } else {
        Self(Inner(self.0.0.bitand(rhs.0.0), self.0.1.bitand(rhs.0.1)))
      }
    }
  }

  #[inline]
  fn bitor(self, rhs: Self) -> Self::Output {
    pick! {
    if #[cfg(target_feature="avx512bw")] {
        Self(bitor_m512i(self.0, rhs.0))
      } else {
        Self(Inner(self.0.0.bitor(rhs.0.0), self.0.1.bitor(rhs.0.1)))
      }
    }
  }

  #[inline]
  fn bitxor(self, rhs: Self) -> Self::Output {
    pick! {
      if #[cfg(target_feature="avx512bw")] {
        Self(bitxor_m512i(self.0, rhs.0))
      } else {
        Self(Inner(self.0.0.bitxor(rhs.0.0), self.0.1.bitxor(rhs.0.1)))
      }
    }
  }

  #[inline]
  pub fn max(self, rhs: Self) -> Self {
    pick! {
      if #[cfg(target_feature="avx512bw")] {
        Self(max_u16_m512i(self.0, rhs.0))
      } else {
        Self(Inner(self.0.0.max(rhs.0.0), self.0.1.max(rhs.0.1)))
      }
    }
  }

  #[inline]
  pub fn min(self, rhs: Self) -> Self {
    pick! {
      if #[cfg(target_feature="avx512bw")] {
        Self(min_u16_m512i(self.0, rhs.0))
      } else {
        Self(Inner(self.0.0.min(rhs.0.0), self.0.1.min(rhs.0.1)))
      }
    }
  }

  #[inline]
  pub fn reduce_add(self) -> u16 {
    let array: [u16x16; 2] = cast(self);
    (array[0] + array[1]).reduce_add()
  }

  #[inline]
  pub fn reduce_mul(self) -> u16 {
    let array: [u16x16; 2] = cast(self);
    (array[0] * array[1]).reduce_mul()
  }

  #[inline]
  pub fn reduce_max(self) -> u16 {
    let array: [u16x16; 2] = cast(self);
    array[0].max(array[1]).reduce_max()
  }

  #[inline]
  pub fn reduce_min(self) -> u16 {
    let array: [u16x16; 2] = cast(self);
    array[0].min(array[1]).reduce_min()
  }

  #[inline]
  pub fn unbounded_shl(self, rhs: Self) -> Self {
    pick! {
      if #[cfg(target_feature="avx512bw")] {
        Self(shl_each_u16_m512i(self.0, rhs.0))
      } else {
        let [self_a, self_b] = cast::<u16x32, [u16x16; 2]>(self);
        let [rhs_a, rhs_b] = cast::<u16x32, [u16x16; 2]>(rhs);

        cast([self_a.unbounded_shl(rhs_a), self_b.unbounded_shl(rhs_b)])
      }
    }
  }

  #[inline]
  pub fn unbounded_shl_scalar(self, rhs: u32) -> Self {
    pick! {
      if #[cfg(target_feature="avx512bw")] {
        // `u32 as u16` truncates the higher half so we need to manually
        // saturate.
        Self(shl_all_u16_m512i(self.0, rhs.min(u16::MAX as u32) as u16))
      } else {
        Self(Inner(
          self.0.0.unbounded_shl_scalar(rhs),
          self.0.1.unbounded_shl_scalar(rhs),
        ))
      }
    }
  }

  #[inline]
  pub fn unbounded_shr(self, rhs: Self) -> Self {
    pick! {
      if #[cfg(target_feature="avx512bw")] {
        Self(shr_each_u16_m512i(self.0, rhs.0))
      } else {
        let [self_a, self_b] = cast::<u16x32, [u16x16; 2]>(self);
        let [rhs_a, rhs_b] = cast::<u16x32, [u16x16; 2]>(rhs);

        cast([self_a.unbounded_shr(rhs_a), self_b.unbounded_shr(rhs_b)])
      }
    }
  }

  #[inline]
  pub fn unbounded_shr_scalar(self, rhs: u32) -> Self {
    pick! {
      if #[cfg(target_feature="avx512bw")] {
        // `u32 as u16` truncates the higher half so we need to manually
        // saturate.
        Self(shr_all_u16_m512i(self.0, rhs.min(u16::MAX as u32) as u16))
      } else {
        Self(Inner(
          self.0.0.unbounded_shr_scalar(rhs),
          self.0.1.unbounded_shr_scalar(rhs),
        ))
      }
    }
  }

  #[inline]
  pub fn saturating_add(self, rhs: Self) -> Self {
    pick! {
      if #[cfg(target_feature="avx512bw")] {
        Self(add_saturating_u16_m512i(self.0, rhs.0))
      } else {
        Self(Inner(
          self.0.0.saturating_add(rhs.0.0),
          self.0.1.saturating_add(rhs.0.1),
        ))
      }
    }
  }

  #[inline]
  pub fn saturating_sub(self, rhs: Self) -> Self {
    pick! {
      if #[cfg(target_feature="avx512bw")] {
        Self(sub_saturating_u16_m512i(self.0, rhs.0))
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
    // Cannot have `widening_mul` because there is no `u32x32` type.
  }

  #[inline]
  pub fn mul_keep_low_high(self, rhs: Self) -> (Self, Self) {
    // x86 has no `_mm512_mul_epu16` intrinsic so there is no `avx512`
    // optimization.

    let [self_a, self_b] = cast::<u16x32, [u16x16; 2]>(self);
    let [rhs_a, rhs_b] = cast::<u16x32, [u16x16; 2]>(rhs);

    let result_a = self_a.mul_keep_low_high(rhs_a);
    let result_b = self_b.mul_keep_low_high(rhs_b);
    (cast([result_a.0, result_b.0]), cast([result_a.1, result_b.1]))
  }

  #[inline]
  pub fn mul_keep_high(self, rhs: Self) -> Self {
    // x86 has no `_mm512_mul_epu16` intrinsic so there is no `avx512`
    // optimization.

    let [self_a, self_b] = cast::<u16x32, [u16x16; 2]>(self);
    let [rhs_a, rhs_b] = cast::<u16x32, [u16x16; 2]>(rhs);

    cast([self_a.mul_keep_high(rhs_a), self_b.mul_keep_high(rhs_b)])
  }
}
