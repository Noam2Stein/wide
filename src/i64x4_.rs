use super::*;

use crate::{f64x4, i64x2, simd::SimdBackend, u64x2, u64x4};

#[cfg(not(target_feature = "avx2"))]
#[repr(C, align(32))]
#[derive(Default, Clone, Copy, PartialEq, Eq)]
pub(crate) struct Inner(pub i64x2, pub i64x2);

unsafe impl SimdBackend for i64x4 {
  pick! {
    if #[cfg(target_feature="avx2")] {
      type Inner = m256i;
    } else {
      type Inner = Inner;
    }
  }

  #[inline]
  fn simd_eq(self, rhs: Self) -> Self {
    pick! {
      if #[cfg(target_feature="avx2")] {
        Self(cmp_eq_mask_i64_m256i(self.0, rhs.0))
      } else {
        Self(Inner(self.0.0.simd_eq(rhs.0.0), self.0.1.simd_eq(rhs.0.1)))
      }
    }
  }

  #[inline]
  fn simd_ne(self, rhs: Self) -> Self {
    pick! {
      if #[cfg(target_feature="avx2")] {
        !self.simd_eq(rhs)
      } else {
        Self(Inner(self.0.0.simd_ne(rhs.0.0), self.0.1.simd_ne(rhs.0.1)))
      }
    }
  }

  #[inline]
  fn simd_lt(self, rhs: Self) -> Self {
    pick! {
      if #[cfg(target_feature="avx2")] {
        Self(!(cmp_gt_mask_i64_m256i(self.0, rhs.0) ^ cmp_eq_mask_i64_m256i(self.0, rhs.0)))
      } else {
        Self(Inner(self.0.0.simd_lt(rhs.0.0), self.0.1.simd_lt(rhs.0.1)))
      }
    }
  }

  #[inline]
  fn simd_gt(self, rhs: Self) -> Self {
    pick! {
      if #[cfg(target_feature="avx2")] {
        Self(cmp_gt_mask_i64_m256i(self.0, rhs.0))
      } else {
        Self(Inner(self.0.0.simd_gt(rhs.0.0), self.0.1.simd_gt(rhs.0.1)))
      }
    }
  }

  #[inline]
  fn simd_le(self, rhs: Self) -> Self {
    pick! {
      if #[cfg(target_feature="avx2")] {
        !self.simd_gt(rhs)
      } else {
        Self(Inner(self.0.0.simd_le(rhs.0.0), self.0.1.simd_le(rhs.0.1)))
      }
    }
  }

  #[inline]
  fn simd_ge(self, rhs: Self) -> Self {
    pick! {
      if #[cfg(target_feature="avx2")] {
        !self.simd_lt(rhs)
      } else {
        Self(Inner(self.0.0.simd_ge(rhs.0.0), self.0.1.simd_ge(rhs.0.1)))
      }
    }
  }

  #[inline]
  fn bitselect(self, if_one: Self, if_zero: Self) -> Self {
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
  fn select(self, if_true: Self, if_false: Self) -> Self {
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

  /// returns the bit mask for each high bit set in the vector with the lowest
  /// lane being the lowest bit
  #[inline]
  fn to_bitmask(self) -> u32 {
    pick! {
      if #[cfg(target_feature="avx2")] {
        // use f64 move_mask since it is the same size as i64
        move_mask_m256d(cast(self.0)) as u32
      } else {
        self.0.0.to_bitmask() | (self.0.1.to_bitmask() << 2)
      }
    }
  }

  /// true if any high bits are set for any value in the vector
  #[inline]
  fn any(self) -> bool {
    pick! {
      if #[cfg(target_feature="avx2")] {
        move_mask_m256d(cast(self.0)) != 0
      } else {
        (self.0.0 | self.0.1).any()
      }
    }
  }

  /// true if all high bits are set for every value in the vector
  #[inline]
  fn all(self) -> bool {
    pick! {
      if #[cfg(target_feature="avx2")] {
        move_mask_m256d(cast(self.0)) == 0b1111
      } else {
        (self.0.0 & self.0.1).all()
      }
    }
  }

  #[inline]
  fn transpose(data: [i64x4; 4]) -> [i64x4; 4] {
    pick! {
      if #[cfg(target_feature="avx2")] {
        // Can this be optimized?
        let a = data[0].unpack_lo(data[2]);
        let b = data[1].unpack_lo(data[3]);
        let c = data[0].unpack_hi(data[2]);
        let d = data[1].unpack_hi(data[3]);
        [
          a.unpack_lo(b),
          a.unpack_hi(b),
          c.unpack_lo(d),
          c.unpack_hi(d),
        ]
      } else {
        #[inline(always)]
        fn transpose_column(data: &[i64x4; 4], index: usize) -> i64x4 {
          i64x4::new([
            data[0].as_array()[index],
            data[1].as_array()[index],
            data[2].as_array()[index],
            data[3].as_array()[index],
          ])
        }

        [
          transpose_column(&data, 0),
          transpose_column(&data, 1),
          transpose_column(&data, 2),
          transpose_column(&data, 3),
        ]
      }
    }
  }
}

impl_simd_int! {
  unsafe {
    T = i64,
    N = 4,
    Simd = i64x4,
    UnsignedSimd = u64x4,
    T_BITS = 64,
    T_BITS_MUL_2 = 128,
    [0, 1, 2, 3],
  }

  #[inline]
  fn shr(self, rhs: u64x4) -> Self::Output {
    pick! {
      if #[cfg(target_feature="avx2")] {
        let arr: [i64; 4] = cast(self);
        let rhs: [u64; 4] = cast(rhs);
        cast([
          arr[0].wrapping_shr(rhs[0] as u32),
          arr[1].wrapping_shr(rhs[1] as u32),
          arr[2].wrapping_shr(rhs[2] as u32),
          arr[3].wrapping_shr(rhs[3] as u32),
        ])
      } else {
        Self(Inner(self.0.0.shr(rhs.0.0), self.0.1.shr(rhs.0.1)))
      }
    }
  }

  #[inline]
  fn shr(self, rhs: u32) -> Self::Output {
    // there is no signed right shift in AVX2
    let [a,b] : [i64x2; 2] = cast(self);
    cast([a.shr(rhs), b.shr(rhs)])
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
  pub fn reduce_max(self) -> i64 {
    let array: [i64; 4] = cast(self);
    array[0].max(array[1]).max(array[2]).max(array[3])
  }

  #[inline]
  pub fn reduce_min(self) -> i64 {
    let array: [i64; 4] = cast(self);
    array[0].min(array[1]).min(array[2]).min(array[3])
  }

  #[inline]
  pub fn unbounded_shr(self, rhs: u64x4) -> Self {
    let [self_a, self_b] = cast::<i64x4, [i64x2; 2]>(self);
    let [rhs_a, rhs_b] = cast::<u64x4, [u64x2; 2]>(rhs);

    cast([self_a.unbounded_shr(rhs_a), self_b.unbounded_shr(rhs_b)])
  }

  #[inline]
  pub fn unbounded_shr_scalar(self, rhs: u32) -> Self {
    // there is no signed right shift in AVX2
    let [self_a, self_b] = cast::<i64x4, [i64x2; 2]>(self);
    cast([self_a.unbounded_shr_scalar(rhs), self_b.unbounded_shr_scalar(rhs)])
  }

  #[inline]
  pub fn saturating_add(self, rhs: Self) -> Self {
    pick! {
      if #[cfg(target_feature="avx2")] {
        let result = self + rhs;
        let overflow = (!(self ^ rhs) & (self ^ result)).is_negative();
        let negative = self.is_negative();

        // If overflow occurs return `MAX` if positive or `MIN` if negative.
        overflow.select(Self::MAX ^ negative, result)
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
        let overflow = ((self ^ rhs) & (self ^ result)).is_negative();
        let negative = self.is_negative();

        // If overflow occurs return `MAX` if positive or `MIN` if negative.
        overflow.select(Self::MAX ^ negative, result)
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
        -(result[0].1 as i64),
        -(result[1].1 as i64),
        -(result[2].1 as i64),
        -(result[3].1 as i64),
      ]),
    )
  }

  optional_fn_widening_mul {
    // Cannot have `widening_mul` because there is no `i128x4` type.
  }

  #[inline]
  pub fn mul_keep_low_high(self, rhs: Self) -> (u64x4, i64x4) {
    // TODO(perf): This implementation looks quite bad. Is there a better
    // one?

    let self_array = self.to_array();
    let rhs_array = rhs.to_array();

    let widening_mul = [
      (self_array[0] as i128).wrapping_mul(rhs_array[0] as i128),
      (self_array[1] as i128).wrapping_mul(rhs_array[1] as i128),
      (self_array[2] as i128).wrapping_mul(rhs_array[2] as i128),
      (self_array[3] as i128).wrapping_mul(rhs_array[3] as i128),
    ];

    (
      u64x4::new([
        widening_mul[0] as u64,
        widening_mul[1] as u64,
        widening_mul[2] as u64,
        widening_mul[3] as u64,
      ]),
      i64x4::new([
        (widening_mul[0] >> 64) as i64,
        (widening_mul[1] >> 64) as i64,
        (widening_mul[2] >> 64) as i64,
        (widening_mul[3] >> 64) as i64,
      ]),
    )
  }

  #[inline]
  pub fn mul_keep_high(self, rhs: Self) -> Self {
    let self_array = self.to_array();
    let rhs_array = rhs.to_array();

    Self::new([
      ((self_array[0] as i128).wrapping_mul(rhs_array[0] as i128) >> 64) as i64,
      ((self_array[1] as i128).wrapping_mul(rhs_array[1] as i128) >> 64) as i64,
      ((self_array[2] as i128).wrapping_mul(rhs_array[2] as i128) >> 64) as i64,
      ((self_array[3] as i128).wrapping_mul(rhs_array[3] as i128) >> 64) as i64,
    ])
  }

  #[inline]
  pub fn abs(self) -> Self {
    pick! {
      if #[cfg(target_feature="avx2")] {
        // avx x86 doesn't have this builtin
        let arr: [i64; 4] = cast(self);
        cast(
          [
            arr[0].wrapping_abs(),
            arr[1].wrapping_abs(),
            arr[2].wrapping_abs(),
            arr[3].wrapping_abs(),
          ])
      } else {
        Self(Inner(self.0.0.abs(), self.0.1.abs()))
      }
    }
  }

  #[inline]
  pub fn is_positive(self) -> Self {
    pick! {
      if #[cfg(all(target_feature="neon", target_arch="aarch64"))] {
        // `neon` has dedicated greater-than-zero intrinsics.
        Self(Inner(self.0.0.is_positive(), self.0.1.is_positive()))
      } else {
        self.simd_gt(Self::ZERO)
      }
    }
  }

  #[inline]
  pub fn is_negative(self) -> Self {
    pick! {
      if #[cfg(all(target_feature="neon", target_arch="aarch64"))] {
        // `neon` has dedicated less-than-zero intrinsics.
        Self(Inner(self.0.0.is_negative(), self.0.1.is_negative()))
      } else {
        self.simd_lt(Self::ZERO)
      }
    }
  }
}

/// The following functionality exists only for [`i64x4`], or only for
/// particular types inconsistently.
impl i64x4 {
  /// Converts each element from [`i64`] to [`f64`].
  #[inline]
  #[must_use]
  pub fn round_float(self) -> f64x4 {
    let arr: [i64; 4] = cast(self);
    cast([arr[0] as f64, arr[1] as f64, arr[2] as f64, arr[3] as f64])
  }

  // Sometimes used for `transpose`.
  #[must_use]
  #[inline]
  #[allow(dead_code)]
  pub(crate) fn unpack_lo(self, b: Self) -> Self {
    pick! {
      if #[cfg(target_feature="avx2")] {
        let [aa, _]: [i64x2; 2] = cast(self);
        let [ba, _]: [i64x2; 2] = cast(b);
        cast([aa.unpack_lo(ba), aa.unpack_hi(ba)])
      } else {
        Self(Inner(self.0.0.unpack_lo(b.0.0), self.0.0.unpack_hi(b.0.0)))
      }
    }
  }

  // Sometimes used for `transpose`.
  #[must_use]
  #[inline]
  #[allow(dead_code)]
  pub(crate) fn unpack_hi(self, b: Self) -> Self {
    pick! {
      if #[cfg(target_feature="avx2")] {
        let [_, ab]: [i64x2; 2] = cast(self);
        let [_, bb]: [i64x2; 2] = cast(b);
        cast([ab.unpack_lo(bb), ab.unpack_hi(bb)])
      } else {
        Self(Inner(self.0.1.unpack_lo(b.0.1), self.0.1.unpack_hi(b.0.1)))
      }
    }
  }
}
