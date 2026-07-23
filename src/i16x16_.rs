use super::*;

use crate::{i8x16, i16x8, i32x8, i32x16, u8x16, u16x8, u16x16};

pick! {
  if #[cfg(target_feature="avx2")] {
    /// A SIMD vector with 16 elements of type [`i16`].
    ///
    /// See the [crate level documentation] for more information about SIMD
    /// vectors.
    ///
    /// [crate level documentation]: crate
    #[repr(transparent)]
    #[derive(Default, Clone, Copy, PartialEq, Eq)]
    pub struct i16x16(pub(crate) m256i);
  } else {
    /// A SIMD vector with 16 elements of type [`i16`].
    ///
    /// See the [crate level documentation] for more information about SIMD
    /// vectors.
    ///
    /// [crate level documentation]: crate
    #[repr(transparent)]
    #[derive(Default, Clone, Copy, PartialEq, Eq)]
    pub struct i16x16(pub(crate) Inner);

    #[repr(C, align(32))]
    #[derive(Default, Clone, Copy, PartialEq, Eq)]
    pub(crate) struct Inner(pub i16x8, pub i16x8);
  }
}

impl_simd! {
  unsafe {
    T = i16,
    N = 16,
    Simd = i16x16,
    optional_type_x86_inner { X86Inner = __m256i },
    optional_type_arm_inner {},
    optional_type_wasm_inner {},
  }

  #[inline]
  fn simd_eq(self, rhs: Self) -> Self::Output {
    pick! {
      if #[cfg(target_feature="avx2")] {
        Self(cmp_eq_mask_i16_m256i(self.0, rhs.0))
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
        Self(!cmp_gt_mask_i16_m256i(self.0, rhs.0) ^ cmp_eq_mask_i16_m256i(self.0,rhs.0))
      } else {
        Self(Inner(self.0.0.simd_lt(rhs.0.0), self.0.1.simd_lt(rhs.0.1)))
      }
    }
  }

  #[inline]
  fn simd_gt(self, rhs: Self) -> Self::Output {
    pick! {
      if #[cfg(target_feature="avx2")] {
        Self(cmp_gt_mask_i16_m256i(self.0, rhs.0))
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
    pick! {
      if #[cfg(target_feature="sse2")] {
          let [a,b] = cast::<_,[m128i;2]>(self);
          move_mask_i8_m128i( pack_i16_to_i8_m128i(a,b)) as u32
        } else {
        self.0.0.to_bitmask() | (self.0.1.to_bitmask() << 8)
      }
    }
  }

  #[inline]
  pub fn any(self) -> bool {
    pick! {
      if #[cfg(target_feature="avx2")] {
        ((move_mask_i8_m256i(self.0) as u32) & 0b10101010101010101010101010101010) != 0
      } else {
        (self.0.0 | self.0.1).any()
      }
    }
  }

  #[inline]
  pub fn all(self) -> bool {
    pick! {
      if #[cfg(target_feature="avx2")] {
        ((move_mask_i8_m256i(self.0) as u32) & 0b10101010101010101010101010101010) == 0b10101010101010101010101010101010
      } else {
        (self.0.0 & self.0.1).all()
      }
    }
  }

  ///
  /// Currently this function is never accelerated.
  #[inline]
  pub fn transpose(data: [i16x16; 16]) -> [i16x16; 16] {
    // Can this be optimized?

    #[inline(always)]
    fn transpose_column(data: &[i16x16; 16], index: usize) -> i16x16 {
      i16x16::new([
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
    ]
  }
}

impl_simd_int! {
  unsafe {
    T = i16,
    N = 16,
    Simd = i16x16,
    UnsignedSimd = u16x16,
    T_BITS = 16,
    T_BITS_MUL_2 = 32,
    [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15],
  }

  #[inline]
  fn shr(self, rhs: u16x16) -> Self::Output {
    pick! {
      if #[cfg(all(target_feature="avx512bw", target_feature="avx512vl"))] {
        #[cfg(target_arch = "x86")]
        use core::arch::x86::_mm256_srav_epi16;
        #[cfg(target_arch = "x86_64")]
        use core::arch::x86_64::_mm256_srav_epi16;

        // Mask `rhs` to 15 to match `wrapping_shr`.
        let rhs = bitand_m256i(rhs.0, set_splat_i16_m256i(15));
        // TODO(safe_arch): Add `_mm256_srav_epi16`.
        cast(unsafe { _mm256_srav_epi16(self.0.0, rhs.0) })
      } else {
        let [self_a, self_b]: [i16x8; 2] = cast(self);
        let [rhs_a, rhs_b]: [u16x8; 2] = cast(rhs);

        cast([self_a >> rhs_a, self_b >> rhs_b])
      }
    }
  }

  #[inline]
  fn shr(self, rhs: u32) -> Self::Output {
    pick! {
      if #[cfg(target_feature="avx2")] {
        // Use `rhs % 16` to perform wrapping shift and not unbounded shift.
        #[expect(clippy::suspicious_arithmetic_impl)]
        let shift = cast([rhs as u64 & 15, 0]);
        Self(shr_all_i16_m256i(self.0, shift))
      } else {
        Self(Inner(self.0.0.shr(rhs), self.0.1.shr(rhs)))
      }
    }
  }

  #[inline]
  pub fn max(self, rhs: Self) -> Self {
    pick! {
      if #[cfg(target_feature="avx2")] {
        Self(max_i16_m256i(self.0, rhs.0))
      } else {
        Self(Inner(self.0.0.max(rhs.0.0), self.0.1.max(rhs.0.1)))
      }
    }
  }

  #[inline]
  pub fn min(self, rhs: Self) -> Self {
    pick! {
      if #[cfg(target_feature="avx2")] {
        Self(min_i16_m256i(self.0, rhs.0))
      } else {
        Self(Inner(self.0.0.min(rhs.0.0), self.0.1.min(rhs.0.1)))
      }
    }
  }

  #[inline]
  pub fn reduce_max(self) -> i16 {
    let arr: [i16x8; 2] = cast(self);

    arr[0].max(arr[1]).reduce_max()
  }

  #[inline]
  pub fn reduce_min(self) -> i16 {
    let arr: [i16x8; 2] = cast(self);

    arr[0].min(arr[1]).reduce_min()
  }

  #[inline]
  pub fn unbounded_shr(self, rhs: u16x16) -> Self {
    pick! {
      if #[cfg(all(target_feature="avx512bw", target_feature="avx512vl"))] {
        #[cfg(target_arch = "x86")]
        use core::arch::x86::_mm256_srav_epi16;
        #[cfg(target_arch = "x86_64")]
        use core::arch::x86_64::_mm256_srav_epi16;

        // TODO(safe_arch): Add `_mm256_srav_epi16`.
        cast(unsafe { _mm256_srav_epi16(self.0.0, rhs.0.0) })
      } else {
        let [self_a, self_b] = cast::<i16x16, [i16x8; 2]>(self);
        let [rhs_a, rhs_b] = cast::<u16x16, [u16x8; 2]>(rhs);

        cast([self_a.unbounded_shr(rhs_a), self_b.unbounded_shr(rhs_b)])
      }
    }
  }

  #[inline]
  pub fn unbounded_shr_scalar(self, rhs: u32) -> Self {
    pick! {
      if #[cfg(target_feature="avx2")] {
        Self(shr_all_i16_m256i(self.0, cast([rhs as u64, 0])))
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
        Self(add_saturating_i16_m256i(self.0, rhs.0))
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
        Self(sub_saturating_i16_m256i(self.0, rhs.0))
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
    let low = cast::<u16x16, i16x16>(low);

    let overflow = high.simd_ne(low.is_negative());
    (low, overflow)
  }

  optional_fn_widening_mul {
    #[inline]
    pub fn widening_mul(self, rhs: Self) -> i32x16 {
      // x86 has no `_mm256_mul_epi16` intrinsic so there is no `avx2`
      // optimization.

      let [self_a, self_b] = cast::<i16x16, [i16x8; 2]>(self);
      let [rhs_a, rhs_b] = cast::<i16x16, [i16x8; 2]>(rhs);

      cast([self_a.widening_mul(rhs_a), self_b.widening_mul(rhs_b)])
    }
  }

  #[inline]
  pub fn mul_keep_low_high(self, rhs: Self) -> (u16x16, i16x16) {
    // x86 has no `_mm256_mul_epi16` intrinsic so there is no `avx2`
    // optimization.

    let [self_a, self_b] = cast::<i16x16, [i16x8; 2]>(self);
    let [rhs_a, rhs_b] = cast::<i16x16, [i16x8; 2]>(rhs);

    let result_a = self_a.mul_keep_low_high(rhs_a);
    let result_b = self_b.mul_keep_low_high(rhs_b);
    (cast([result_a.0, result_b.0]), cast([result_a.1, result_b.1]))
  }

  #[inline]
  pub fn mul_keep_high(self, rhs: Self) -> Self {
    // x86 has no `_mm256_mul_epi16` intrinsic so there is no `avx2`
    // optimization.

    let [self_a, self_b] = cast::<i16x16, [i16x8; 2]>(self);
    let [rhs_a, rhs_b] = cast::<i16x16, [i16x8; 2]>(rhs);

    cast([self_a.mul_keep_high(rhs_a), self_b.mul_keep_high(rhs_b)])
  }

  #[inline]
  pub fn abs(self) -> Self {
    pick! {
      if #[cfg(target_feature="avx2")] {
        Self(abs_i16_m256i(self.0))
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

impl From<i8x16> for i16x16 {
  /// widen with sign extend from i8 to i16
  #[inline]
  fn from(i: i8x16) -> Self {
    i16x16::from_i8x16(i)
  }
}

impl From<u8x16> for i16x16 {
  /// widen with zero extend from u8 to i16
  #[inline]
  fn from(i: u8x16) -> Self {
    cast(u16x16::from(i))
  }
}

/// The following functionality exists only for [`i16x16`], or only for
/// particular types inconsistently.
impl i16x16 {
  /// Converts each element from [`i8`] to [`i16`].
  #[inline]
  #[must_use]
  pub fn from_i8x16(v: i8x16) -> Self {
    pick! {
      if #[cfg(target_feature="avx2")] {
        i16x16(convert_to_i16_m256i_from_i8_m128i(v.0))
      } else if #[cfg(target_feature="sse4.1")] {
        i16x16(Inner(
          i16x8(convert_to_i16_m128i_from_lower8_i8_m128i(v.0)),
          i16x8(convert_to_i16_m128i_from_lower8_i8_m128i(unpack_high_i64_m128i(v.0, v.0))),
        ))
      } else if #[cfg(target_feature="sse2")] {
        i16x16(Inner(
          i16x8(shr_imm_i16_m128i::<8>(unpack_low_i8_m128i(v.0, v.0))),
          i16x8(shr_imm_i16_m128i::<8>( unpack_high_i8_m128i(v.0, v.0))),
        ))
      } else {

        i16x16::new([
          v.as_array()[0] as i16,
          v.as_array()[1] as i16,
          v.as_array()[2] as i16,
          v.as_array()[3] as i16,
          v.as_array()[4] as i16,
          v.as_array()[5] as i16,
          v.as_array()[6] as i16,
          v.as_array()[7] as i16,
          v.as_array()[8] as i16,
          v.as_array()[9] as i16,
          v.as_array()[10] as i16,
          v.as_array()[11] as i16,
          v.as_array()[12] as i16,
          v.as_array()[13] as i16,
          v.as_array()[14] as i16,
          v.as_array()[15] as i16,
          ])
      }
    }
  }

  /// Partially computes the dot product.
  ///
  /// First this multiplies the input 16-bit integers, producing intermediate
  /// 32-bit integers. Then this horizontally adds adjacent pairs, resulting in
  /// eight 32-bit integers.
  #[inline]
  #[must_use]
  pub fn dot(self, rhs: Self) -> i32x8 {
    pick! {
      if #[cfg(target_feature="avx2")] {
        i32x8(mul_i16_horizontal_add_m256i(self.0, rhs.0))
      } else {
        i32x8(crate::i32x8_::Inner(self.0.0.dot(rhs.0.0), self.0.1.dot(rhs.0.1)))
      }
    }
  }

  /// Multiply and scale equivalent to `((self * rhs) + 0x4000) >> 15` on each
  /// lane, effectively multiplying by a 16 bit fixed point number between `-1`
  /// and `1`. This corresponds to the following instructions:
  /// - `vqrdmulhq_n_s16` instruction on neon
  /// - `i16x8_q15mulr_sat` on simd128
  /// - `_mm256_mulhrs_epi16` on avx2
  /// - emulated via `mul_i16_*` on sse2
  #[inline]
  #[must_use]
  pub fn mul_scale_round(self, rhs: Self) -> Self {
    pick! {
      if #[cfg(target_feature="avx2")] {
        Self(mul_i16_scale_round_m256i(self.0, rhs.0))
      } else {
        Self(Inner(
          self.0.0.mul_scale_round(rhs.0.0),
          self.0.1.mul_scale_round(rhs.0.1),
        ))
      }
    }
  }

  /// Multiply and scale equivalent to `((self * rhs) + 0x4000) >> 15` on each
  /// lane, effectively multiplying by a 16 bit fixed point number between `-1`
  /// and `1`. This corresponds to the following instructions:
  /// - `vqrdmulhq_n_s16` instruction on neon
  /// - `i16x8_q15mulr_sat` on simd128
  /// - `_mm256_mulhrs_epi16` on avx2
  /// - emulated via `mul_i16_*` on sse2
  #[inline]
  #[must_use]
  pub fn mul_scale_round_n(self, rhs: i16) -> Self {
    pick! {
      if #[cfg(target_feature="avx2")] {
        Self(mul_i16_scale_round_m256i(self.0, set_splat_i16_m256i(rhs)))
      } else {
        Self(Inner(
          self.0.0.mul_scale_round_n(rhs),
          self.0.1.mul_scale_round_n(rhs),
        ))
      }
    }
  }
}
