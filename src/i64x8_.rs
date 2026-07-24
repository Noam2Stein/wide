use super::*;

use crate::{f64x8, i64x4, simd::SimdBackend, u64x4, u64x8};

#[cfg(not(target_feature = "avx512f"))]
#[repr(C, align(64))]
#[derive(Default, Clone, Copy, PartialEq, Eq)]
pub(crate) struct Inner(pub i64x4, pub i64x4);

unsafe impl SimdBackend for i64x8 {
  pick! {
    if #[cfg(target_feature="avx512f")] {
      type Inner = m512i;
    } else {
      type Inner = Inner;
    }
  }
}

impl_simd! {
  unsafe {
    T = i64,
    N = 8,
    Simd = i64x8,
    optional_type_x86_inner { X86Inner = __m512i },
    optional_type_arm_inner {},
    optional_type_wasm_inner {},
  }

  #[inline]
  fn simd_eq(self, rhs: Self) -> Self::Output {
    pick! {
      if #[cfg(target_feature="avx512f")] {
        Self(cmp_op_mask_i64_m512i::<{cmp_int_op!(Eq)}>(self.0, rhs.0))
      } else {
        Self(Inner(self.0.0.simd_eq(rhs.0.0), self.0.1.simd_eq(rhs.0.1)))
      }
    }
  }

  #[inline]
  fn simd_ne(self, rhs: Self) -> Self::Output {
    pick! {
      if #[cfg(target_feature="avx512f")] {
        Self(cmp_op_mask_i64_m512i::<{cmp_int_op!(Ne)}>(self.0, rhs.0))
      } else {
        Self(Inner(self.0.0.simd_ne(rhs.0.0), self.0.1.simd_ne(rhs.0.1)))
      }
    }
  }

  #[inline]
  fn simd_lt(self, rhs: Self) -> Self::Output {
    pick! {
      if #[cfg(target_feature="avx512f")] {
        Self(cmp_op_mask_i64_m512i::<{cmp_int_op!(Lt)}>(self.0, rhs.0))
      } else {
        Self(Inner(rhs.0.0.simd_gt(self.0.0), rhs.0.1.simd_gt(self.0.1)))
      }
    }
  }

  #[inline]
  fn simd_gt(self, rhs: Self) -> Self::Output {
    pick! {
      if #[cfg(target_feature="avx512f")] {
        Self(cmp_op_mask_i64_m512i::<{cmp_int_op!(Nle)}>(self.0, rhs.0))
      } else {
        Self(Inner(self.0.0.simd_gt(rhs.0.0), self.0.1.simd_gt(rhs.0.1)))
      }
    }
  }

  #[inline]
  fn simd_le(self, rhs: Self) -> Self::Output {
    pick! {
      if #[cfg(target_feature="avx512f")] {
        Self(cmp_op_mask_i64_m512i::<{cmp_int_op!(Le)}>(self.0, rhs.0))
      } else {
        Self(Inner(self.0.0.simd_le(rhs.0.0), self.0.1.simd_le(rhs.0.1)))
      }
    }
  }

  #[inline]
  fn simd_ge(self, rhs: Self) -> Self::Output {
    pick! {
      if #[cfg(target_feature="avx512f")] {
        Self(cmp_op_mask_i64_m512i::<{cmp_int_op!(Nlt)}>(self.0, rhs.0))
      } else {
        Self(Inner(self.0.0.simd_ge(rhs.0.0), self.0.1.simd_ge(rhs.0.1)))
      }
    }
  }

  #[inline]
  pub fn bitselect(self, if_one: Self, if_zero: Self) -> Self {
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
  pub fn select(self, if_true: Self, if_false: Self) -> Self {
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

  /// returns the bit mask for each high bit set in the vector with the lowest
  /// lane being the lowest bit
  #[inline]
  pub fn to_bitmask(self) -> u32 {
    pick! {
      if #[cfg(target_feature="avx512dq")] {
        // use f64 move_mask since it is the same size as i64
        movepi64_mask_m512d(cast(self.0)) as u32
      } else {
        self.0.0.to_bitmask() | (self.0.1.to_bitmask() << 4)
      }
    }
  }

  /// true if any high bits are set for any value in the vector
  #[inline]
  pub fn any(self) -> bool {
    pick! {
      if #[cfg(target_feature="avx512f")] {
        movepi64_mask_m512d(cast(self.0)) != 0
      } else {
        let [a, b]: [i64x4; 2] = cast(self);
        (a | b).any()
      }
    }
  }

  /// true if all high bits are set for every value in the vector
  #[inline]
  pub fn all(self) -> bool {
    pick! {
      if #[cfg(target_feature="avx512bw")] {
        movepi64_mask_m512d(cast(self.0)) == 0b11111111
      } else {
        let [a, b]: [i64x4; 2] = cast(self);
        (a & b).all()
      }
    }
  }

  ///
  /// Currently this function is never accelerated.
  #[inline]
  pub fn transpose(data: [i64x8; 8]) -> [i64x8; 8] {
    // Can this be optimized?

    #[inline(always)]
    fn transpose_column(data: &[i64x8; 8], index: usize) -> i64x8 {
      i64x8::new([
        data[0].as_array()[index],
        data[1].as_array()[index],
        data[2].as_array()[index],
        data[3].as_array()[index],
        data[4].as_array()[index],
        data[5].as_array()[index],
        data[6].as_array()[index],
        data[7].as_array()[index],
      ])
    }

    [
      transpose_column(&data, 0),
      transpose_column(&data, 1),
      transpose_column(&data, 2),
      transpose_column(&data, 3),
      transpose_column(&data, 4),
      transpose_column(&data, 5),
      transpose_column(&data, 6),
      transpose_column(&data, 7),
    ]
  }
}

impl_simd_int! {
  unsafe {
    T = i64,
    N = 8,
    Simd = i64x8,
    UnsignedSimd = u64x8,
    T_BITS = 64,
    T_BITS_MUL_2 = 128,
    [0, 1, 2, 3, 4, 5, 6, 7],
  }

  #[inline]
  fn shr(self, rhs: u64x8) -> Self::Output {
    pick! {
      if #[cfg(target_feature="avx512f")] {
        // TODO(safe_arch): add shr_each_i64_m512i (arithmetic right shift)
        // Self(shr_each_i64_m512i(self.0, rhs.0))
        // Fallback for now:
        let a: [i64; 8] = cast(self);
        let r: [u64; 8] = cast(rhs);
        cast([
          a[0].wrapping_shr(r[0] as u32),
          a[1].wrapping_shr(r[1] as u32),
          a[2].wrapping_shr(r[2] as u32),
          a[3].wrapping_shr(r[3] as u32),
          a[4].wrapping_shr(r[4] as u32),
          a[5].wrapping_shr(r[5] as u32),
          a[6].wrapping_shr(r[6] as u32),
          a[7].wrapping_shr(r[7] as u32),
        ])
      } else {
        // widen via two halves
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
        Self(shr_all_i64_m512i(self.0, shift))
      } else {
        Self(Inner(self.0.0.shr(rhs), self.0.1.shr(rhs)))
      }
    }
  }

  #[inline]
  pub fn max(self, rhs: Self) -> Self {
    pick! {
      if #[cfg(target_feature="avx512f")] {
        Self(max_i64_m512i(self.0, rhs.0))
      } else {
        Self(Inner(self.0.0.max(rhs.0.0), self.0.1.max(rhs.0.1)))
      }
    }
  }

  #[inline]
  pub fn min(self, rhs: Self) -> Self {
    pick! {
      if #[cfg(target_feature="avx512f")] {
        Self(min_i64_m512i(self.0, rhs.0))
      } else {
        Self(Inner(self.0.0.min(rhs.0.0), self.0.1.min(rhs.0.1)))
      }
    }
  }

  #[inline]
  pub fn reduce_max(self) -> i64 {
    let array: [i64x4; 2] = cast(self);
    array[0].max(array[1]).reduce_max()
  }

  #[inline]
  pub fn reduce_min(self) -> i64 {
    let array: [i64x4; 2] = cast(self);
    array[0].min(array[1]).reduce_min()
  }

  #[inline]
  pub fn unbounded_shr(self, rhs: u64x8) -> Self {
    // TODO(safe_arch): add shr_each_i64_m512i (arithmetic right shift)
    // Self(shr_each_i64_m512i(self.0, rhs.0))
    // Fallback for now:

    let [self_a, self_b] = cast::<i64x8, [i64x4; 2]>(self);
    let [rhs_a, rhs_b] = cast::<u64x8, [u64x4; 2]>(rhs);

    cast([self_a.unbounded_shr(rhs_a), self_b.unbounded_shr(rhs_b)])
  }

  #[inline]
  pub fn unbounded_shr_scalar(self, rhs: u32) -> Self {
    pick! {
      if #[cfg(target_feature="avx512f")] {
        Self(shr_all_i64_m512i(self.0, rhs as u64))
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
      if #[cfg(target_feature="avx512f")] {
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
        -(result[0].1 as i64),
        -(result[1].1 as i64),
        -(result[2].1 as i64),
        -(result[3].1 as i64),
        -(result[4].1 as i64),
        -(result[5].1 as i64),
        -(result[6].1 as i64),
        -(result[7].1 as i64),
      ]),
    )
  }

  optional_fn_widening_mul {
    // Cannot have `widening_mul` because there is no `i128x8` type.
  }

  #[inline]
  pub fn mul_keep_low_high(self, rhs: Self) -> (u64x8, i64x8) {
    // TODO(perf): This implementation looks quite bad. Is there a better
    // one?

    let self_array = self.to_array();
    let rhs_array = rhs.to_array();

    let widening_mul = [
      (self_array[0] as i128).wrapping_mul(rhs_array[0] as i128),
      (self_array[1] as i128).wrapping_mul(rhs_array[1] as i128),
      (self_array[2] as i128).wrapping_mul(rhs_array[2] as i128),
      (self_array[3] as i128).wrapping_mul(rhs_array[3] as i128),
      (self_array[4] as i128).wrapping_mul(rhs_array[4] as i128),
      (self_array[5] as i128).wrapping_mul(rhs_array[5] as i128),
      (self_array[6] as i128).wrapping_mul(rhs_array[6] as i128),
      (self_array[7] as i128).wrapping_mul(rhs_array[7] as i128),
    ];

    (
      u64x8::new([
        widening_mul[0] as u64,
        widening_mul[1] as u64,
        widening_mul[2] as u64,
        widening_mul[3] as u64,
        widening_mul[4] as u64,
        widening_mul[5] as u64,
        widening_mul[6] as u64,
        widening_mul[7] as u64,
      ]),
      i64x8::new([
        (widening_mul[0] >> 64) as i64,
        (widening_mul[1] >> 64) as i64,
        (widening_mul[2] >> 64) as i64,
        (widening_mul[3] >> 64) as i64,
        (widening_mul[4] >> 64) as i64,
        (widening_mul[5] >> 64) as i64,
        (widening_mul[6] >> 64) as i64,
        (widening_mul[7] >> 64) as i64,
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
      ((self_array[4] as i128).wrapping_mul(rhs_array[4] as i128) >> 64) as i64,
      ((self_array[5] as i128).wrapping_mul(rhs_array[5] as i128) >> 64) as i64,
      ((self_array[6] as i128).wrapping_mul(rhs_array[6] as i128) >> 64) as i64,
      ((self_array[7] as i128).wrapping_mul(rhs_array[7] as i128) >> 64) as i64,
    ])
  }

  #[inline]
  pub fn abs(self) -> Self {
    pick! {
      if #[cfg(target_feature="avx512f")] {
        // AVX512 might have this, unsure for now
        let arr: [i64; 8] = cast(self);
        cast(
          [
            arr[0].wrapping_abs(),
            arr[1].wrapping_abs(),
            arr[2].wrapping_abs(),
            arr[3].wrapping_abs(),
            arr[4].wrapping_abs(),
            arr[5].wrapping_abs(),
            arr[6].wrapping_abs(),
            arr[7].wrapping_abs(),
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

/// The following functionality exists only for [`i64x8`], or only for
/// particular types inconsistently.
impl i64x8 {
  /// Converts each element from [`i64`] to [`f64`].
  #[inline]
  #[must_use]
  pub fn round_float(self) -> f64x8 {
    let arr: [i64; 8] = cast(self);
    cast([
      arr[0] as f64,
      arr[1] as f64,
      arr[2] as f64,
      arr[3] as f64,
      arr[4] as f64,
      arr[5] as f64,
      arr[6] as f64,
      arr[7] as f64,
    ])
  }
}
