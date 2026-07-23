use super::*;

use crate::{i16x16, i32x16, u16x16, u16x32};

pick! {
  if #[cfg(target_feature="avx512bw")] {
    /// A SIMD vector with 32 elements of type [`i16`].
    ///
    /// See the [crate level documentation] for more information about SIMD
    /// vectors.
    ///
    /// [crate level documentation]: crate
    #[repr(transparent)]
    #[derive(Default, Clone, Copy, PartialEq, Eq)]
    pub struct i16x32(pub(crate) m512i);
  } else {
    /// A SIMD vector with 32 elements of type [`i16`].
    ///
    /// See the [crate level documentation] for more information about SIMD
    /// vectors.
    ///
    /// [crate level documentation]: crate
    #[repr(transparent)]
    #[derive(Default, Clone, Copy, PartialEq, Eq)]
    pub struct i16x32(pub(crate) Inner);

    #[repr(C, align(64))]
    #[derive(Default, Clone, Copy, PartialEq, Eq)]
    pub(crate) struct Inner(pub i16x16, pub i16x16);
  }
}

impl_simd! {
  unsafe {
    T = i16,
    N = 32,
    Simd = i16x32,
    optional_type_x86_inner { X86Inner = __m512i },
    optional_type_arm_inner {},
    optional_type_wasm_inner {},
  }

  #[inline]
  fn simd_eq(self, rhs: Self) -> Self::Output {
    pick! {
      if #[cfg(target_feature="avx512bw")] {
        Self(cmp_op_mask_i16_m512i::<{cmp_int_op!(Eq)}>(self.0, rhs.0))
      } else {
        Self(Inner(self.0.0.simd_eq(rhs.0.0), self.0.1.simd_eq(rhs.0.1)))
      }
    }
  }

  #[inline]
  fn simd_ne(self, rhs: Self) -> Self::Output {
    pick! {
      if #[cfg(target_feature="avx512bw")] {
        Self(cmp_op_mask_i16_m512i::<{cmp_int_op!(Ne)}>(self.0, rhs.0))
      } else {
        Self(Inner(self.0.0.simd_ne(rhs.0.0), self.0.1.simd_ne(rhs.0.1)))
      }
    }
  }

  #[inline]
  fn simd_lt(self, rhs: Self) -> Self::Output {
    pick! {
      if #[cfg(target_feature="avx512bw")] {
        Self(cmp_op_mask_i16_m512i::<{cmp_int_op!(Lt)}>(self.0, rhs.0))
      } else {
        Self(Inner(rhs.0.0.simd_gt(self.0.0), rhs.0.1.simd_gt(self.0.1)))
      }
    }
  }

  #[inline]
  fn simd_gt(self, rhs: Self) -> Self::Output {
    pick! {
      if #[cfg(target_feature="avx512bw")] {
        Self(cmp_op_mask_i16_m512i::<{cmp_int_op!(Nle)}>(self.0, rhs.0))
      } else {
        Self(Inner(self.0.0.simd_gt(rhs.0.0), self.0.1.simd_gt(rhs.0.1)))
      }
    }
  }

  #[inline]
  fn simd_le(self, rhs: Self) -> Self::Output {
    pick! {
      if #[cfg(target_feature="avx512bw")] {
        Self(cmp_op_mask_i16_m512i::<{cmp_int_op!(Le)}>(self.0, rhs.0))
      } else {
        Self(Inner(self.0.0.simd_le(rhs.0.0), self.0.1.simd_le(rhs.0.1)))
      }
    }
  }

  #[inline]
  fn simd_ge(self, rhs: Self) -> Self::Output {
    pick! {
      if #[cfg(target_feature="avx512bw")] {
        Self(cmp_op_mask_i16_m512i::<{cmp_int_op!(Nlt)}>(self.0, rhs.0))
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
    pick! {
      if #[cfg(target_feature="avx512bw")] {
        // use f16 move_mask since it is the same size as i16
        movepi16_mask_m512i(self.0) as u32
      } else {
        self.0.0.to_bitmask() | (self.0.1.to_bitmask() << 16)
      }
    }
  }

  #[inline]
  pub fn any(self) -> bool {
    pick! {
      if #[cfg(target_feature="avx512bw")] {
        movepi16_mask_m512i(self.0) != 0
      } else {
        (self.0.0 | self.0.1).any()
      }
    }
  }

  #[inline]
  pub fn all(self) -> bool {
    pick! {
      if #[cfg(target_feature="avx512bw")] {
        movepi16_mask_m512i(self.0) == 0xFFFFFFFF
      } else {
        (self.0.0 & self.0.1).all()
      }
    }
  }

  ///
  /// Currently this function is never accelerated.
  #[inline]
  pub fn transpose(data: [i16x32; 32]) -> [i16x32; 32] {
    // Can this be optimized?

    #[inline(always)]
    fn transpose_column(data: &[i16x32; 32], index: usize) -> i16x32 {
      i16x32::new([
        data[0].as_array()[index],
        data[1].as_array()[index],
        data[2].as_array()[index],
        data[3].as_array()[index],
        data[4].as_array()[index],
        data[5].as_array()[index],
        data[6].as_array()[index],
        data[7].as_array()[index],
        data[8].as_array()[index],
        data[9].as_array()[index],
        data[10].as_array()[index],
        data[11].as_array()[index],
        data[12].as_array()[index],
        data[13].as_array()[index],
        data[14].as_array()[index],
        data[15].as_array()[index],
        data[16].as_array()[index],
        data[17].as_array()[index],
        data[18].as_array()[index],
        data[19].as_array()[index],
        data[20].as_array()[index],
        data[21].as_array()[index],
        data[22].as_array()[index],
        data[23].as_array()[index],
        data[24].as_array()[index],
        data[25].as_array()[index],
        data[26].as_array()[index],
        data[27].as_array()[index],
        data[28].as_array()[index],
        data[29].as_array()[index],
        data[30].as_array()[index],
        data[31].as_array()[index],
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
      transpose_column(&data, 8),
      transpose_column(&data, 9),
      transpose_column(&data, 10),
      transpose_column(&data, 11),
      transpose_column(&data, 12),
      transpose_column(&data, 13),
      transpose_column(&data, 14),
      transpose_column(&data, 15),
      transpose_column(&data, 16),
      transpose_column(&data, 17),
      transpose_column(&data, 18),
      transpose_column(&data, 19),
      transpose_column(&data, 20),
      transpose_column(&data, 21),
      transpose_column(&data, 22),
      transpose_column(&data, 23),
      transpose_column(&data, 24),
      transpose_column(&data, 25),
      transpose_column(&data, 26),
      transpose_column(&data, 27),
      transpose_column(&data, 28),
      transpose_column(&data, 29),
      transpose_column(&data, 30),
      transpose_column(&data, 31),
    ]
  }
}

impl_simd_int! {
  unsafe {
    T = i16,
    N = 32,
    Simd = i16x32,
    UnsignedSimd = u16x32,
    T_BITS = 16,
    T_BITS_MUL_2 = 32,
    [
      0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20,
      21, 22, 23, 24, 25, 26, 27, 28, 29, 30, 31
    ],
  }

  #[inline]
  fn shr(self, rhs: u16x32) -> Self::Output {
    pick! {
      if #[cfg(target_feature="avx512bw")] {
        #[cfg(target_arch = "x86")]
        use core::arch::x86::_mm512_srav_epi16;
        #[cfg(target_arch = "x86_64")]
        use core::arch::x86_64::_mm512_srav_epi16;

        // Mask `rhs` to 15 to match `wrapping_shr`.
        let rhs = bitand_m512i(rhs.0, set_splat_i16_m512i(15));
        // TODO(safe_arch): Add `_mm512_srav_epi16`.
        Self(m512i(unsafe { _mm512_srav_epi16(self.0.0, rhs.0) }))
      } else {
        let [self_a, self_b]: [i16x16; 2] = cast(self);
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
        Self(shr_all_i16_m512i(self.0, shift))
      } else {
        Self(Inner(self.0.0.shr(rhs), self.0.1.shr(rhs)))
      }
    }
  }

  #[inline]
  pub fn max(self, rhs: Self) -> Self {
    pick! {
      if #[cfg(target_feature="avx512bw")] {
        Self(max_i16_m512i(self.0, rhs.0))
      } else {
        Self(Inner(self.0.0.max(rhs.0.0), self.0.1.max(rhs.0.1)))
      }
    }
  }

  #[inline]
  pub fn min(self, rhs: Self) -> Self {
    pick! {
      if #[cfg(target_feature="avx512bw")] {
        Self(min_i16_m512i(self.0, rhs.0))
      } else {
        Self(Inner(self.0.0.min(rhs.0.0), self.0.1.min(rhs.0.1)))
      }
    }
  }

  #[inline]
  pub fn reduce_max(self) -> i16 {
    let arr: [i16x16; 2] = cast(self);
    arr[0].max(arr[1]).reduce_max()
  }

  #[inline]
  pub fn reduce_min(self) -> i16 {
    let arr: [i16x16; 2] = cast(self);
    arr[0].min(arr[1]).reduce_min()
  }

  #[inline]
  pub fn unbounded_shr(self, rhs: u16x32) -> Self {
    pick! {
      if #[cfg(target_feature="avx512bw")] {
        #[cfg(target_arch = "x86")]
        use core::arch::x86::_mm512_srav_epi16;
        #[cfg(target_arch = "x86_64")]
        use core::arch::x86_64::_mm512_srav_epi16;

        // TODO(safe_arch): Add `_mm512_srav_epi16`.
        Self(m512i(unsafe { _mm512_srav_epi16(self.0.0, rhs.0.0) }))
      } else {
        let [self_a, self_b] = cast::<i16x32, [i16x16; 2]>(self);
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
        Self(shr_all_i16_m512i(self.0, rhs.min(u16::MAX as u32) as u16))
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
        Self(add_saturating_i16_m512i(self.0, rhs.0))
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
        Self(sub_saturating_i16_m512i(self.0, rhs.0))
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
    let low = cast::<u16x32, i16x32>(low);

    let overflow = high.simd_ne(low.is_negative());
    (low, overflow)
  }

  optional_fn_widening_mul {
    // Cannot have `widening_mul` because there is no `i32x32` type.
  }

  #[inline]
  pub fn mul_keep_low_high(self, rhs: Self) -> (u16x32, i16x32) {
    // x86 has no `_mm512_mul_epi16` intrinsic so there is no `avx512`
    // optimization.

    let [self_a, self_b] = cast::<i16x32, [i16x16; 2]>(self);
    let [rhs_a, rhs_b] = cast::<i16x32, [i16x16; 2]>(rhs);

    let result_a = self_a.mul_keep_low_high(rhs_a);
    let result_b = self_b.mul_keep_low_high(rhs_b);
    (cast([result_a.0, result_b.0]), cast([result_a.1, result_b.1]))
  }

  #[inline]
  pub fn mul_keep_high(self, rhs: Self) -> Self {
    // x86 has no `_mm512_mul_epi16` intrinsic so there is no `avx512`
    // optimization.

    let [self_a, self_b] = cast::<i16x32, [i16x16; 2]>(self);
    let [rhs_a, rhs_b] = cast::<i16x32, [i16x16; 2]>(rhs);

    cast([self_a.mul_keep_high(rhs_a), self_b.mul_keep_high(rhs_b)])
  }

  #[inline]
  pub fn abs(self) -> Self {
    pick! {
      if #[cfg(target_feature="avx512bw")] {
        Self(abs_i16_m512i(self.0))
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

/// The following functionality exists only for [`i16x32`], or only for
/// particular types inconsistently.
impl i16x32 {
  /// Partially computes the dot product.
  ///
  /// First this multiplies the input 16-bit integers, producing intermediate
  /// 32-bit integers. Then this horizontally adds adjacent pairs, resulting in
  /// sixteen 32-bit integers.
  #[inline]
  #[must_use]
  pub fn dot(self, rhs: Self) -> i32x16 {
    pick! {
      if #[cfg(target_feature="avx512bw")] {
        i32x16(mul_i16_horizontal_add_m512i(self.0, rhs.0))
      } else {
        i32x16(crate::i32x16_::Inner(self.0.0.dot(rhs.0.0), self.0.1.dot(rhs.0.1)))
      }
    }
  }
}
