use super::*;

use crate::{f32x16, i32x8, simd::SimdBackend, u32x16};

#[cfg(not(target_feature = "avx512f"))]
#[repr(C, align(64))]
#[derive(Default, Clone, Copy, PartialEq, Eq)]
pub(crate) struct Inner(pub i32x8, pub i32x8);

unsafe impl SimdBackend for i32x16 {
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
        Self(cmp_op_mask_i32_m512i::<{cmp_int_op!(Eq)}>(self.0, rhs.0))
      } else {
        Self(Inner(self.0.0.simd_eq(rhs.0.0), self.0.1.simd_eq(rhs.0.1)))
      }
    }
  }

  #[inline]
  fn simd_ne(self, rhs: Self) -> Self {
    pick! {
      if #[cfg(target_feature="avx512f")] {
        Self(cmp_op_mask_i32_m512i::<{cmp_int_op!(Ne)}>(self.0, rhs.0))
      } else {
        Self(Inner(self.0.0.simd_ne(rhs.0.0), self.0.1.simd_ne(rhs.0.1)))
      }
    }
  }

  #[inline]
  fn simd_lt(self, rhs: Self) -> Self {
    pick! {
      if #[cfg(target_feature="avx512f")] {
        Self(cmp_op_mask_i32_m512i::<{cmp_int_op!(Lt)}>(self.0, rhs.0))
      } else {
        Self(Inner(self.0.0.simd_lt(rhs.0.0), self.0.1.simd_lt(rhs.0.1)))
      }
    }
  }

  #[inline]
  fn simd_gt(self, rhs: Self) -> Self {
    pick! {
      if #[cfg(target_feature="avx512f")] {
        Self(cmp_op_mask_i32_m512i::<{cmp_int_op!(Nle)}>(self.0, rhs.0))
      } else {
        Self(Inner(self.0.0.simd_gt(rhs.0.0), self.0.1.simd_gt(rhs.0.1)))
      }
    }
  }

  #[inline]
  fn simd_le(self, rhs: Self) -> Self {
    pick! {
      if #[cfg(target_feature="avx512f")] {
        Self(cmp_op_mask_i32_m512i::<{cmp_int_op!(Le)}>(self.0, rhs.0))
      } else {
        Self(Inner(self.0.0.simd_le(rhs.0.0), self.0.1.simd_le(rhs.0.1)))
      }
    }
  }

  #[inline]
  fn simd_ge(self, rhs: Self) -> Self {
    pick! {
      if #[cfg(target_feature="avx512f")] {
        Self(cmp_op_mask_i32_m512i::<{cmp_int_op!(Nlt)}>(self.0, rhs.0))
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
    pick! {
      if #[cfg(target_feature="avx512dq")] {
        movepi32_mask_m512i(self.0) as u32
      } else {
        self.0.0.to_bitmask() | (self.0.1.to_bitmask() << 8)
      }
    }
  }

  #[inline]
  fn any(self) -> bool {
    pick! {
      if #[cfg(target_feature="avx512bw")] {
        movepi32_mask_m512i(self.0) != 0
      } else {
        let [a, b]: [i32x8; 2] = cast(self);
        (a | b).any()
      }
    }
  }

  #[inline]
  fn all(self) -> bool {
    pick! {
      if #[cfg(target_feature="avx512bw")] {
        movepi32_mask_m512i(self.0) == 0xFFFF
      } else {
        let [a, b]: [i32x8; 2] = cast(self);
        (a & b).all()
      }
    }
  }

  #[inline]
  fn transpose(data: [i32x16; 16]) -> [i32x16; 16] {
    // Can this be optimized?

    #[inline(always)]
    fn transpose_column(data: &[i32x16; 16], index: usize) -> i32x16 {
      i32x16::new([
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
    T = i32,
    N = 16,
    Simd = i32x16,
    UnsignedSimd = u32x16,
    T_BITS = 32,
    T_BITS_MUL_2 = 64,
    [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15],
  }

  #[inline]
  fn shr(self, rhs: u32x16) -> Self::Output {
    pick! {
      if #[cfg(target_feature="avx512f")] {
        #[cfg(target_arch = "x86")]
        use core::arch::x86::_mm512_srav_epi32;
        #[cfg(target_arch = "x86_64")]
        use core::arch::x86_64::_mm512_srav_epi32;

        // Mask `rhs` to 31 to match `wrapping_shr`.
        let rhs = bitand_m512i(rhs.0, set_splat_i32_m512i(31));
        // TODO(safe_arch): Add `_mm512_srav_epi32`.
        Self(m512i(unsafe { _mm512_srav_epi32(self.0.0, rhs.0) }))
      } else {
        Self(Inner(self.0.0 >> rhs.0.0, self.0.1 >> rhs.0.1))
      }
    }
  }

  #[inline]
  fn shr(self, rhs: u32) -> Self::Output {
    pick! {
      if #[cfg(target_feature="avx512f")] {
        // Use `rhs % 32` to perform wrapping shift and not unbounded shift.
        #[expect(clippy::suspicious_arithmetic_impl)]
        let shift = rhs & 31;
        Self(shr_all_i32_m512i(self.0, shift))
      } else {
        Self(Inner(self.0.0.shr(rhs), self.0.1.shr(rhs)))
      }
    }
  }

  #[inline]
  pub fn max(self, rhs: Self) -> Self {
    pick! {
      if #[cfg(target_feature="avx512f")] {
        Self(max_i32_m512i(self.0, rhs.0))
      } else {
        Self(Inner(self.0.0.max(rhs.0.0), self.0.1.max(rhs.0.1)))
      }
    }
  }

  #[inline]
  pub fn min(self, rhs: Self) -> Self {
    pick! {
      if #[cfg(target_feature="avx512f")] {
        Self(min_i32_m512i(self.0, rhs.0))
      } else {
        Self(Inner(self.0.0.min(rhs.0.0), self.0.1.min(rhs.0.1)))
      }
    }
  }

  #[inline]
  pub fn reduce_max(self) -> i32 {
    let arr: [i32x8; 2] = cast(self);
    arr[0].max(arr[1]).reduce_max()
  }

  #[inline]
  pub fn reduce_min(self) -> i32 {
    let arr: [i32x8; 2] = cast(self);
    arr[0].min(arr[1]).reduce_min()
  }

  #[inline]
  pub fn unbounded_shr(self, rhs: u32x16) -> Self {
    pick! {
      if #[cfg(target_feature="avx512f")] {
        #[cfg(target_arch = "x86")]
        use core::arch::x86::_mm512_srav_epi32;
        #[cfg(target_arch = "x86_64")]
        use core::arch::x86_64::_mm512_srav_epi32;

        // TODO(safe_arch): Add `_mm512_srav_epi32`.
        Self(m512i(unsafe { _mm512_srav_epi32(self.0.0, rhs.0.0) }))
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
        Self(shr_all_i32_m512i(self.0, rhs))
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
    let (low, high) = self.mul_keep_low_high(rhs);
    let low = cast::<u32x16, i32x16>(low);

    let overflow = high.simd_ne(low.is_negative());
    (low, overflow)
  }

  optional_fn_widening_mul {
    // Cannot have `widening_mul` because there is no `i64x16` type.
  }

  #[inline]
  pub fn mul_keep_low_high(self, rhs: Self) -> (u32x16, i32x16) {
    pick! {
      if #[cfg(all(target_feature="avx512f", target_feature="avx512dq"))] {
        #[cfg(target_arch = "x86")]
        use core::arch::x86::{_mm512_unpackhi_epi64, _mm512_unpacklo_epi64};
        #[cfg(target_arch = "x86_64")]
        use core::arch::x86_64::{_mm512_unpackhi_epi64, _mm512_unpacklo_epi64};

        let even_wide_mul = mul_i32_wide_m512i(self.0, rhs.0);
        let odd_wide_mul = mul_i32_wide_m512i(
          shuffle_i32_m512i::<0b_00_11_00_01>(self.0),
          shuffle_i32_m512i::<0b_00_11_00_01>(rhs.0),
        );
        let ll_hh_1 = unpack_low_i32_m512i(even_wide_mul, odd_wide_mul);
        let ll_hh_2 = unpack_high_i32_m512i(even_wide_mul, odd_wide_mul);
        // TODO(safe_arch): Add `_mm512_unpacklo_epi64` and `_mm512_unpackhi_epi64`.
        (
          u32x16(m512i(unsafe { _mm512_unpacklo_epi64(ll_hh_1.0, ll_hh_2.0) })),
          i32x16(m512i(unsafe { _mm512_unpackhi_epi64(ll_hh_1.0, ll_hh_2.0) })),
        )
      } else {
        let [self_a, self_b] = cast::<i32x16, [i32x8; 2]>(self);
        let [rhs_a, rhs_b] = cast::<i32x16, [i32x8; 2]>(rhs);

        let result_a = self_a.mul_keep_low_high(rhs_a);
        let result_b = self_b.mul_keep_low_high(rhs_b);
        (
          cast([result_a.0, result_b.0]),
          cast([result_a.1, result_b.1]),
        )
      }
    }
  }

  #[inline]
  pub fn mul_keep_high(self, rhs: Self) -> Self {
    pick! {
      if #[cfg(all(target_feature="avx512f", target_feature="avx512dq"))] {
        #[cfg(target_arch = "x86")]
        use core::arch::x86::_mm512_unpackhi_epi64;
        #[cfg(target_arch = "x86_64")]
        use core::arch::x86_64::_mm512_unpackhi_epi64;

        let even_wide_mul = mul_i32_wide_m512i(self.0, rhs.0);
        let odd_wide_mul = mul_i32_wide_m512i(
          shuffle_i32_m512i::<0b_00_11_00_01>(self.0),
          shuffle_i32_m512i::<0b_00_11_00_01>(rhs.0),
        );
        let ll_hh_1 = unpack_low_i32_m512i(even_wide_mul, odd_wide_mul);
        let ll_hh_2 = unpack_high_i32_m512i(even_wide_mul, odd_wide_mul);
        // TODO(safe_arch): Add `_mm512_unpackhi_epi64`.
        Self(m512i(unsafe { _mm512_unpackhi_epi64(ll_hh_1.0, ll_hh_2.0) }))
      } else {
        let [self_a, self_b] = cast::<i32x16, [i32x8; 2]>(self);
        let [rhs_a, rhs_b] = cast::<i32x16, [i32x8; 2]>(rhs);

        cast([self_a.mul_keep_high(rhs_a), self_b.mul_keep_high(rhs_b)])
      }
    }
  }

  #[inline]
  pub fn abs(self) -> Self {
    pick! {
      if #[cfg(target_feature="avx512bw")] {
        Self(abs_i32_m512i(self.0))
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

/// The following functionality exists only for [`i32x16`], or only for
/// particular types inconsistently.
impl i32x16 {
  /// Converts each element from [`i32`] to [`f32`].
  #[inline]
  #[must_use]
  pub fn round_float(self) -> f32x16 {
    pick! {
      if #[cfg(target_feature="avx512f")] {
        cast(convert_to_m512_from_i32_m512i(self.0))
      } else {
        Simd(crate::f32x16_::Inner(self.0.0.round_float(), self.0.1.round_float()))
      }
    }
  }
}
