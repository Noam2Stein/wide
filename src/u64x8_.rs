use super::*;

use crate::{i64x8, simd::SimdBackend, u64x4};

#[cfg(not(target_feature = "avx512f"))]
#[repr(C, align(64))]
#[derive(Default, Clone, Copy, PartialEq, Eq)]
pub(crate) struct Inner(pub u64x4, pub u64x4);

unsafe impl SimdBackend for u64x8 {
  pick! {
    if #[cfg(target_feature="avx512f")] {
      type Inner = m512i;
    } else {
      type Inner = Inner;
    }
  }

  #[inline]
  fn simd_eq(self, rhs: Self) -> Self {
    pick! {
      if #[cfg(target_feature="avx512f")] {
        Self(cmp_op_mask_u64_m512i::<{cmp_int_op!(Eq)}>(self.0, rhs.0))
      } else {
        Self(Inner(self.0.0.simd_eq(rhs.0.0), self.0.1.simd_eq(rhs.0.1)))
      }
    }
  }

  #[inline]
  fn simd_ne(self, rhs: Self) -> Self {
    pick! {
      if #[cfg(target_feature="avx512f")] {
        Self(cmp_op_mask_u64_m512i::<{cmp_int_op!(Ne)}>(self.0, rhs.0))
      } else {
        Self(Inner(self.0.0.simd_ne(rhs.0.0), self.0.1.simd_ne(rhs.0.1)))
      }
    }
  }

  #[inline]
  fn simd_lt(self, rhs: Self) -> Self {
    pick! {
      if #[cfg(target_feature="avx512f")] {
        Self(cmp_op_mask_u64_m512i::<{cmp_int_op!(Lt)}>(self.0, rhs.0))
      } else {
        Self(Inner(self.0.0.simd_lt(rhs.0.0), self.0.1.simd_lt(rhs.0.1)))
      }
    }
  }

  #[inline]
  fn simd_gt(self, rhs: Self) -> Self {
    pick! {
      if #[cfg(target_feature="avx512f")] {
        Self(cmp_op_mask_u64_m512i::<{cmp_int_op!(Nle)}>(self.0, rhs.0))
      } else {
        Self(Inner(self.0.0.simd_gt(rhs.0.0), self.0.1.simd_gt(rhs.0.1)))
      }
    }
  }

  #[inline]
  fn simd_le(self, rhs: Self) -> Self {
    pick! {
      if #[cfg(target_feature="avx512f")] {
        Self(cmp_op_mask_u64_m512i::<{cmp_int_op!(Le)}>(self.0, rhs.0))
      } else {
        Self(Inner(self.0.0.simd_le(rhs.0.0), self.0.1.simd_le(rhs.0.1)))
      }
    }
  }

  #[inline]
  fn simd_ge(self, rhs: Self) -> Self {
    pick! {
      if #[cfg(target_feature="avx512f")] {
        Self(cmp_op_mask_u64_m512i::<{cmp_int_op!(Nlt)}>(self.0, rhs.0))
      } else {
        Self(Inner(self.0.0.simd_ge(rhs.0.0), self.0.1.simd_ge(rhs.0.1)))
      }
    }
  }

  #[inline]
  fn bitselect(self, if_one: Self, if_zero: Self) -> Self {
    pick! {
      if #[cfg(target_feature="avx512f")] {
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
  fn select(self, if_true: Self, if_false: Self) -> Self {
    pick! {
      if #[cfg(target_feature="avx512f")] {
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
  fn to_bitmask(self) -> u32 {
    i64x8::to_bitmask(cast(self))
  }

  #[inline]
  fn any(self) -> bool {
    i64x8::any(cast(self))
  }

  #[inline]
  fn all(self) -> bool {
    i64x8::all(cast(self))
  }

  #[inline]
  fn transpose(data: [u64x8; 8]) -> [u64x8; 8] {
    cast(i64x8::transpose(cast(data)))
  }
}

impl_simd_uint! {
  unsafe {
    T = u64,
    N = 8,
    Simd = u64x8,
    SignedSimd = i64x8,
    T_BITS = 64,
    T_BITS_MUL_2 = 128,
    [0, 1, 2, 3, 4, 5, 6, 7],
  }

  #[inline]
  fn not(self) -> Self::Output {
    pick! {
      if #[cfg(target_feature="avx512f")] {
        Self(bitxor_m512i(self.0, set_splat_i64_m512i(-1)))
      } else {
        Self(Inner(self.0.0.not(), self.0.1.not()))
      }
    }
  }

  #[inline]
  fn add(self, rhs: Self) -> Self::Output {
    pick! {
      if #[cfg(target_feature="avx512f")] {
        Self(add_i64_m512i(self.0, rhs.0))
      } else {
        Self(Inner(self.0.0.add(rhs.0.0), self.0.1.add(rhs.0.1)))
      }
    }
  }

  #[inline]
  fn sub(self, rhs: Self) -> Self::Output {
    pick! {
      if #[cfg(target_feature="avx512f")] {
        Self(sub_i64_m512i(self.0, rhs.0))
      } else {
        Self(Inner(self.0.0.sub(rhs.0.0), self.0.1.sub(rhs.0.1)))
      }
    }
  }

  #[inline]
  fn mul(self, rhs: Self) -> Self::Output {
    pick! {
      if #[cfg(target_feature="avx512f")] {
        let arr1: [u64; 8] = cast(self);
        let arr2: [u64; 8] = cast(rhs);
        cast([
          arr1[0].wrapping_mul(arr2[0]),
          arr1[1].wrapping_mul(arr2[1]),
          arr1[2].wrapping_mul(arr2[2]),
          arr1[3].wrapping_mul(arr2[3]),
          arr1[4].wrapping_mul(arr2[4]),
          arr1[5].wrapping_mul(arr2[5]),
          arr1[6].wrapping_mul(arr2[6]),
          arr1[7].wrapping_mul(arr2[7]),
        ])
      } else {
        Self(Inner(self.0.0.mul(rhs.0.0), self.0.1.mul(rhs.0.1)))
      }
    }
  }

  #[inline]
  fn shl(self, rhs: Self) -> Self::Output {
    pick! {
      if #[cfg(target_feature="avx512f")] {
        // Use `rhs % 64` to perform wrapping shift and not unbounded shift.
        let rhs = bitand_m512i(rhs.0, set_splat_i64_m512i(63));
        Self(shl_each_u64_m512i(self.0, rhs))
      } else {
        Self(Inner(self.0.0.shl(rhs.0.0), self.0.1.shl(rhs.0.1)))
      }
    }
  }

  #[inline]
  fn shl(self, rhs: u32) -> Self::Output {
    pick! {
      if #[cfg(target_feature="avx512f")] {
        // Use `rhs % 64` to perform wrapping shift and not unbounded shift.
        #[expect(clippy::suspicious_arithmetic_impl)]
        let shift = rhs as u64 & 63;
        Self(shl_all_u64_m512i(self.0, shift))
      } else {
        Self(Inner(self.0.0.shl(rhs), self.0.1.shl(rhs)))
      }
    }
  }

  #[inline]
  fn shr(self, rhs: Self) -> Self::Output {
    pick! {
      if #[cfg(target_feature="avx512f")] {
        // Use `rhs % 64` to perform wrapping shift and not unbounded shift.
        let rhs = bitand_m512i(rhs.0, set_splat_i64_m512i(63));
        Self(shr_each_u64_m512i(self.0, rhs))
      } else {
        Self(Inner(self.0.0.shr(rhs.0.0), self.0.1.shr(rhs.0.1)))
      }
    }
  }

  #[inline]
  fn shr(self, rhs: u32) -> Self::Output {
    pick! {
      if #[cfg(target_feature="avx512f")] {
        // Use `rhs % 64` to perform wrapping shift and not unbounded shift.
        #[expect(clippy::suspicious_arithmetic_impl)]
        let shift = rhs as u64 & 63;
        Self(shr_all_u64_m512i(self.0, shift))
      } else {
        Self(Inner(self.0.0.shr(rhs), self.0.1.shr(rhs)))
      }
    }
  }

  #[inline]
  fn bitand(self, rhs: Self) -> Self::Output {
    pick! {
      if #[cfg(target_feature="avx512f")] {
        Self(bitand_m512i(self.0, rhs.0))
      } else {
        Self(Inner(self.0.0.bitand(rhs.0.0), self.0.1.bitand(rhs.0.1)))
      }
    }
  }

  #[inline]
  fn bitor(self, rhs: Self) -> Self::Output {
    pick! {
    if #[cfg(target_feature="avx512f")] {
        Self(bitor_m512i(self.0, rhs.0))
      } else {
        Self(Inner(self.0.0.bitor(rhs.0.0), self.0.1.bitor(rhs.0.1)))
      }
    }
  }

  #[inline]
  fn bitxor(self, rhs: Self) -> Self::Output {
    pick! {
      if #[cfg(target_feature="avx512f")] {
        Self(bitxor_m512i(self.0, rhs.0))
      } else {
        Self(Inner(self.0.0.bitxor(rhs.0.0), self.0.1.bitxor(rhs.0.1)))
      }
    }
  }

  #[inline]
  pub fn max(self, rhs: Self) -> Self {
    pick! {
      if #[cfg(target_feature="avx512f")] {
        Self(max_u64_m512i(self.0, rhs.0))
      } else {
        Self(Inner(self.0.0.max(rhs.0.0), self.0.1.max(rhs.0.1)))
      }
    }
  }

  #[inline]
  pub fn min(self, rhs: Self) -> Self {
    pick! {
      if #[cfg(target_feature="avx512f")] {
        Self(min_u64_m512i(self.0, rhs.0))
      } else {
        Self(Inner(self.0.0.min(rhs.0.0), self.0.1.min(rhs.0.1)))
      }
    }
  }

  #[inline]
  pub fn reduce_add(self) -> u64 {
    let array: [u64x4; 2] = cast(self);
    (array[0] + array[1]).reduce_add()
  }


  #[inline]
  pub fn reduce_mul(self) -> u64 {
    let array: [u64x4; 2] = cast(self);
    (array[0] * array[1]).reduce_mul()
  }

  #[inline]
  pub fn reduce_max(self) -> u64 {
    let array: [u64x4; 2] = cast(self);
    array[0].max(array[1]).reduce_max()
  }

  #[inline]
  pub fn reduce_min(self) -> u64 {
    let array: [u64x4; 2] = cast(self);
    array[0].min(array[1]).reduce_min()
  }

  #[inline]
  pub fn unbounded_shl(self, rhs: Self) -> Self {
    pick! {
      if #[cfg(target_feature="avx512f")] {
        Self(shl_each_u64_m512i(self.0, rhs.0))
      } else {
        Self(Inner(
          self.0.0.unbounded_shl(rhs.0.0),
          self.0.1.unbounded_shl(rhs.0.1),
        ))
      }
    }
  }

  #[inline]
  pub fn unbounded_shl_scalar(self, rhs: u32) -> Self {
    pick! {
      if #[cfg(target_feature="avx512f")] {
        Self(shl_all_u64_m512i(self.0, rhs as u64))
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
      if #[cfg(target_feature="avx512f")] {
        Self(shr_each_u64_m512i(self.0, rhs.0))
      } else {
        Self(Inner(
          self.0.0.unbounded_shr(rhs.0.0),
          self.0.1.unbounded_shr(rhs.0.1),
        ))
      }
    }
  }

  #[inline]
  pub fn unbounded_shr_scalar(self, rhs: u32) -> Self {
    pick! {
      if #[cfg(target_feature="avx512f")] {
        Self(shr_all_u64_m512i(self.0, rhs as u64))
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
      if #[cfg(target_feature="avx512f")] {
        let result = self + rhs;
        let overflow = result.simd_lt(self);
        // Return `MAX` (all bits set) if overflow occurs.
        result | overflow
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
      if #[cfg(target_feature="avx512f")] {
        let result = self - rhs;
        let no_overflow = result.simd_le(self);
        // Return `0` (no bits set) if overflow occurs.
        result & no_overflow
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
    // TODO(perf): This implementation looks quite bad. Is there a better
    // one? This intentionally avoids `mul_keep_low_high` because getting the
    // high bits of 64-bit multiplication could be slow.

    let self_array = self.to_array();
    let rhs_array = rhs.to_array();

    let result = [
      self_array[0].overflowing_mul(rhs_array[0]),
      self_array[1].overflowing_mul(rhs_array[1]),
      self_array[2].overflowing_mul(rhs_array[2]),
      self_array[3].overflowing_mul(rhs_array[3]),
      self_array[4].overflowing_mul(rhs_array[4]),
      self_array[5].overflowing_mul(rhs_array[5]),
      self_array[6].overflowing_mul(rhs_array[6]),
      self_array[7].overflowing_mul(rhs_array[7]),
    ];
    (
      Self::new([
        result[0].0,
        result[1].0,
        result[2].0,
        result[3].0,
        result[4].0,
        result[5].0,
        result[6].0,
        result[7].0,
      ]),
      Self::new([
        -(result[0].1 as i64) as u64,
        -(result[1].1 as i64) as u64,
        -(result[2].1 as i64) as u64,
        -(result[3].1 as i64) as u64,
        -(result[4].1 as i64) as u64,
        -(result[5].1 as i64) as u64,
        -(result[6].1 as i64) as u64,
        -(result[7].1 as i64) as u64,
      ]),
    )
  }

  optional_fn_widening_mul {
    // Cannot have `widening_mul` because there is no `u128x8` type.
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
      (self_array[2] as u128).wrapping_mul(rhs_array[2] as u128),
      (self_array[3] as u128).wrapping_mul(rhs_array[3] as u128),
      (self_array[4] as u128).wrapping_mul(rhs_array[4] as u128),
      (self_array[5] as u128).wrapping_mul(rhs_array[5] as u128),
      (self_array[6] as u128).wrapping_mul(rhs_array[6] as u128),
      (self_array[7] as u128).wrapping_mul(rhs_array[7] as u128),
    ];

    (
      Self::new([
        widening_mul[0] as u64,
        widening_mul[1] as u64,
        widening_mul[2] as u64,
        widening_mul[3] as u64,
        widening_mul[4] as u64,
        widening_mul[5] as u64,
        widening_mul[6] as u64,
        widening_mul[7] as u64,
      ]),
      Self::new([
        (widening_mul[0] >> 64) as u64,
        (widening_mul[1] >> 64) as u64,
        (widening_mul[2] >> 64) as u64,
        (widening_mul[3] >> 64) as u64,
        (widening_mul[4] >> 64) as u64,
        (widening_mul[5] >> 64) as u64,
        (widening_mul[6] >> 64) as u64,
        (widening_mul[7] >> 64) as u64,
      ]),
    )
  }

  #[inline]
  pub fn mul_keep_high(self, rhs: Self) -> Self {
    pick! {
      if #[cfg(target_feature="avx512f")] {
        let arr1: [u64; 8] = cast(self);
        let arr2: [u64; 8] = cast(rhs);
        cast([
          (arr1[0] as u128 * arr2[0] as u128 >> 64) as u64,
          (arr1[1] as u128 * arr2[1] as u128 >> 64) as u64,
          (arr1[2] as u128 * arr2[2] as u128 >> 64) as u64,
          (arr1[3] as u128 * arr2[3] as u128 >> 64) as u64,
          (arr1[4] as u128 * arr2[4] as u128 >> 64) as u64,
          (arr1[5] as u128 * arr2[5] as u128 >> 64) as u64,
          (arr1[6] as u128 * arr2[6] as u128 >> 64) as u64,
          (arr1[7] as u128 * arr2[7] as u128 >> 64) as u64,
        ])
      } else {
        Self(Inner(
          self.0.0.mul_keep_high(rhs.0.0),
          self.0.1.mul_keep_high(rhs.0.1),
        ))
      }
    }
  }
}
