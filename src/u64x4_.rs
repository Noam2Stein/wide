use super::*;

#[cfg(not(target_feature = "avx2"))]
use crate::u64x2;
use crate::{i64x4, simd::SimdBackend};

#[cfg(not(target_feature = "avx2"))]
#[repr(C, align(32))]
#[derive(Default, Clone, Copy, PartialEq, Eq)]
pub(crate) struct Inner(pub u64x2, pub u64x2);

unsafe impl SimdBackend for u64x4 {
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
    T = u64,
    N = 4,
    Simd = u64x4,
    optional_type_x86_inner { X86Inner = __m256i },
    optional_type_arm_inner {},
    optional_type_wasm_inner {},
  }

  #[inline]
  fn simd_eq(self, rhs: Self) -> Self::Output {
    pick! {
      if #[cfg(target_feature="avx2")] {
        Self(cmp_eq_mask_i64_m256i(self.0, rhs.0))
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
    // lt is just gt the other way around
    rhs.simd_gt(self)
  }

  #[inline]
  fn simd_gt(self, rhs: Self) -> Self::Output {
    pick! {
      if #[cfg(target_feature="avx2")] {
        // no unsigned gt than so inverting the high bit will get the correct result
        let highbit = u64x4::splat(1 << 63);
        Self(cmp_gt_mask_i64_m256i((self ^ highbit).0, (rhs ^ highbit).0))
      } else {
        Self(Inner(self.0.0.simd_gt(rhs.0.0), self.0.1.simd_gt(rhs.0.1)))
      }
    }
  }

  #[inline]
  fn simd_le(self, rhs: Self) -> Self::Output {
    pick! {
      if #[cfg(target_feature="avx2")] {
        !self.simd_gt(rhs)
      } else {
        Self(Inner(self.0.0.simd_le(rhs.0.0), self.0.1.simd_le(rhs.0.1)))
      }
    }
  }

  #[inline]
  fn simd_ge(self, rhs: Self) -> Self::Output {
    pick! {
      if #[cfg(target_feature="avx2")] {
        !self.simd_lt(rhs)
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
        Self(blend_varying_i8_m256i(if_false.0,if_true.0,self.0))
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
    i64x4::to_bitmask(cast(self))
  }

  #[inline]
  pub fn any(self) -> bool {
    i64x4::any(cast(self))
  }

  #[inline]
  pub fn all(self) -> bool {
    i64x4::all(cast(self))
  }

  ///
  /// Currently this function is only accelerated on `avx2`.
  #[inline]
  pub fn transpose(data: [u64x4; 4]) -> [u64x4; 4] {
    cast(i64x4::transpose(cast(data)))
  }
}

impl_simd_uint! {
  unsafe {
    T = u64,
    N = 4,
    Simd = u64x4,
    SignedSimd = i64x4,
    T_BITS = 64,
    T_BITS_MUL_2 = 128,
    [0, 1, 2, 3],
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
        Self(add_i64_m256i(self.0, rhs.0))
      } else {
        Self(Inner(self.0.0.add(rhs.0.0), self.0.1.add(rhs.0.1)))
      }
    }
  }

  #[inline]
  fn sub(self, rhs: Self) -> Self::Output {
    pick! {
      if #[cfg(target_feature="avx2")] {
        Self(sub_i64_m256i(self.0, rhs.0))
      } else {
        Self(Inner(self.0.0.sub(rhs.0.0), self.0.1.sub(rhs.0.1)))
      }
    }
  }

  #[inline]
  fn mul(self, rhs: Self) -> Self::Output {
    pick! {
      if #[cfg(target_feature="avx2")] {
        let arr1: [i64; 4] = cast(self);
        let arr2: [i64; 4] = cast(rhs);
        cast([
          arr1[0].wrapping_mul(arr2[0]),
          arr1[1].wrapping_mul(arr2[1]),
          arr1[2].wrapping_mul(arr2[2]),
          arr1[3].wrapping_mul(arr2[3]),
        ])
      } else {
        Self(Inner(self.0.0.mul(rhs.0.0), self.0.1.mul(rhs.0.1)))
      }
    }
  }

  #[inline]
  fn shl(self, rhs: Self) -> Self::Output {
    pick! {
      if #[cfg(target_feature="avx2")] {
        // mask the shift count to 63 to have same behavior on all platforms
        let shift_by = rhs & Self::splat(63);
        Self(shl_each_u64_m256i(self.0, shift_by.0))
      } else {
        Self(Inner(self.0.0.shl(rhs.0.0), self.0.1.shl(rhs.0.1)))
      }
    }
  }

  #[inline]
  fn shl(self, rhs: u32) -> Self::Output {
    pick! {
      if #[cfg(target_feature="avx2")] {
        // Use `rhs % 64` to perform wrapping shift and not unbounded shift.
        #[expect(clippy::suspicious_arithmetic_impl)]
        let shift = cast([rhs as u64 & 63, 0]);
        Self(shl_all_u64_m256i(self.0, shift))
      } else {
        Self(Inner(self.0.0.shl(rhs), self.0.1.shl(rhs)))
      }
    }
  }

  #[inline]
  fn shr(self, rhs: Self) -> Self::Output {
    pick! {
      if #[cfg(target_feature="avx2")] {
        // mask the shift count to 63 to have same behavior on all platforms
        let shift_by = rhs & Self::splat(63);
        Self(shr_each_u64_m256i(self.0, shift_by.0))
      } else {
        Self(Inner(self.0.0.shr(rhs.0.0), self.0.1.shr(rhs.0.1)))
      }
    }
  }

  #[inline]
  fn shr(self, rhs: u32) -> Self::Output {
    pick! {
      if #[cfg(target_feature="avx2")] {
        // Use `rhs % 64` to perform wrapping shift and not unbounded shift.
        #[expect(clippy::suspicious_arithmetic_impl)]
        let shift = cast([rhs as u64 & 63, 0]);
        Self(shr_all_u64_m256i(self.0, shift))
      } else {
        Self(Inner(self.0.0.shr(rhs), self.0.1.shr(rhs)))
      }
    }
  }

  #[inline]
  fn bitand(self, rhs: Self) -> Self::Output {
    pick! {
      if #[cfg(target_feature="avx2")] {
        Self(bitand_m256i(self.0, rhs.0))
      } else {
        Self(Inner(self.0.0.bitand(rhs.0.0), self.0.1.bitand(rhs.0.1)))
      }
    }
  }

  #[inline]
  fn bitor(self, rhs: Self) -> Self::Output {
    pick! {
    if #[cfg(target_feature="avx2")] {
        Self(bitor_m256i(self.0, rhs.0))
      } else {
        Self(Inner(self.0.0.bitor(rhs.0.0), self.0.1.bitor(rhs.0.1)))
      }
    }
  }

  #[inline]
  fn bitxor(self, rhs: Self) -> Self::Output {
    pick! {
      if #[cfg(target_feature="avx2")] {
        Self(bitxor_m256i(self.0, rhs.0))
      } else {
        Self(Inner(self.0.0.bitxor(rhs.0.0), self.0.1.bitxor(rhs.0.1)))
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
      if #[cfg(all(target_arch="x86_64", target_feature="avx2"))] {
        let zwxx  = shuffle_ai_i64_all_m256i::<0b00_00_11_10>(self.0);
        let xz_yw = add_i64_m256i(zwxx, self.0);
        let yw_xz  = shuffle_ai_i64_all_m256i::<0b00_00_00_01>(xz_yw);
        let sum = add_i64_m256i(xz_yw, yw_xz);
        extract_i64_from_m256i::<0>(sum).cast_unsigned()
      } else {
        let array: [u64; 4] = cast(self);
        array[0]
          .wrapping_add(array[1])
          .wrapping_add(array[2])
          .wrapping_add(array[3])
      }
    }
  }

  #[inline]
  pub fn reduce_mul(self) -> u64 {
    let array: [u64; 4] = cast(self);
    array[0]
      .wrapping_mul(array[1])
      .wrapping_mul(array[2])
      .wrapping_mul(array[3])
  }

  #[inline]
  pub fn reduce_max(self) -> u64 {
    let array: [u64; 4] = cast(self);
    array[0].max(array[1]).max(array[2]).max(array[3])
  }

  #[inline]
  pub fn reduce_min(self) -> u64 {
    let array: [u64; 4] = cast(self);
    array[0].min(array[1]).min(array[2]).min(array[3])
  }

  #[inline]
  pub fn unbounded_shl(self, rhs: Self) -> Self {
    pick! {
      if #[cfg(target_feature="avx2")] {
        Self(shl_each_u64_m256i(self.0, rhs.0))
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
      if #[cfg(target_feature="avx2")] {
        Self(shl_all_u64_m256i(self.0, cast([rhs as u64, 0])))
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
      if #[cfg(target_feature="avx2")] {
        Self(shr_each_u64_m256i(self.0, rhs.0))
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
      if #[cfg(target_feature="avx2")] {
        Self(shr_all_u64_m256i(self.0, cast([rhs as u64, 0])))
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
      if #[cfg(target_feature="avx2")] {
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
      if #[cfg(target_feature="avx2")] {
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
    ];
    (
      Self::new([result[0].0, result[1].0, result[2].0, result[3].0]),
      Self::new([
        -(result[0].1 as i64) as u64,
        -(result[1].1 as i64) as u64,
        -(result[2].1 as i64) as u64,
        -(result[3].1 as i64) as u64,
      ]),
    )
  }

  optional_fn_widening_mul {
    // Cannot have `widening_mul` because there is no `u128x4` type.
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
    ];

    (
      Self::new([
        widening_mul[0] as u64,
        widening_mul[1] as u64,
        widening_mul[2] as u64,
        widening_mul[3] as u64,
      ]),
      Self::new([
        (widening_mul[0] >> 64) as u64,
        (widening_mul[1] >> 64) as u64,
        (widening_mul[2] >> 64) as u64,
        (widening_mul[3] >> 64) as u64,
      ]),
    )
  }

  #[inline]
  pub fn mul_keep_high(self, rhs: Self) -> Self {
    pick! {
      if #[cfg(target_feature="avx2")] {
        let arr1: [u64; 4] = cast(self);
        let arr2: [u64; 4] = cast(rhs);
        cast([
          (arr1[0] as u128 * arr2[0] as u128 >> 64) as u64,
          (arr1[1] as u128 * arr2[1] as u128 >> 64) as u64,
          (arr1[2] as u128 * arr2[2] as u128 >> 64) as u64,
          (arr1[3] as u128 * arr2[3] as u128 >> 64) as u64,
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
