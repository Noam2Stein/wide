use super::*;

use crate::{i32x16, u16x16, u32x8};

pick! {
  if #[cfg(target_feature="avx512f")] {
    /// A SIMD vector with 16 elements of type [`u32`].
    ///
    /// See the [crate level documentation] for more information about SIMD
    /// vectors.
    ///
    /// [crate level documentation]: crate
    #[repr(transparent)]
    #[derive(Default, Clone, Copy, PartialEq, Eq)]
    pub struct u32x16(pub(crate) m512i);
  } else {
    /// A SIMD vector with 16 elements of type [`u32`].
    ///
    /// See the [crate level documentation] for more information about SIMD
    /// vectors.
    ///
    /// [crate level documentation]: crate
    #[repr(transparent)]
    #[derive(Default, Clone, Copy, PartialEq, Eq)]
    pub struct u32x16(pub(crate) Inner);

    #[repr(C, align(64))]
    #[derive(Default, Clone, Copy, PartialEq, Eq)]
    pub(crate) struct Inner(pub u32x8, pub u32x8);
  }
}

impl_simd! {
  unsafe {
    T = u32,
    N = 16,
    Simd = u32x16,
    optional_type_x86_inner { X86Inner = __m512i },
    optional_type_arm_inner {},
    optional_type_wasm_inner {},
  }

  #[inline]
  fn simd_eq(self, rhs: Self) -> Self::Output {
    pick! {
      if #[cfg(target_feature="avx512f")] {
        Self(cmp_op_mask_u32_m512i::<{cmp_int_op!(Eq)}>(self.0, rhs.0))
      } else {
        Self(Inner(self.0.0.simd_eq(rhs.0.0), self.0.1.simd_eq(rhs.0.1)))
      }
    }
  }

  #[inline]
  fn simd_ne(self, rhs: Self) -> Self::Output {
    pick! {
      if #[cfg(target_feature="avx512f")] {
        Self(cmp_op_mask_u32_m512i::<{cmp_int_op!(Ne)}>(self.0, rhs.0))
      } else {
        Self(Inner(self.0.0.simd_ne(rhs.0.0), self.0.1.simd_ne(rhs.0.1)))
      }
    }
  }

  #[inline]
  fn simd_lt(self, rhs: Self) -> Self::Output {
    pick! {
      if #[cfg(target_feature="avx512f")] {
        Self(cmp_op_mask_u32_m512i::<{cmp_int_op!(Lt)}>(self.0, rhs.0))
      } else {
        Self(Inner(rhs.0.0.simd_gt(self.0.0), rhs.0.1.simd_gt(self.0.1)))
      }
    }
  }

  #[inline]
  fn simd_gt(self, rhs: Self) -> Self::Output {
    pick! {
      if #[cfg(target_feature="avx512f")] {
        Self(cmp_op_mask_u32_m512i::<{cmp_int_op!(Nle)}>(self.0, rhs.0))
      } else {
        Self(Inner(self.0.0.simd_gt(rhs.0.0), self.0.1.simd_gt(rhs.0.1)))
      }
    }
  }

  #[inline]
  fn simd_le(self, rhs: Self) -> Self::Output {
    pick! {
      if #[cfg(target_feature="avx512f")] {
        Self(cmp_op_mask_u32_m512i::<{cmp_int_op!(Le)}>(self.0, rhs.0))
      } else {
        Self(Inner(self.0.0.simd_le(rhs.0.0), self.0.1.simd_le(rhs.0.1)))
      }
    }
  }

  #[inline]
  fn simd_ge(self, rhs: Self) -> Self::Output {
    pick! {
      if #[cfg(target_feature="avx512f")] {
        Self(cmp_op_mask_u32_m512i::<{cmp_int_op!(Nlt)}>(self.0, rhs.0))
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

  #[inline]
  pub fn to_bitmask(self) -> u32 {
    i32x16::to_bitmask(cast(self))
  }

  #[inline]
  pub fn any(self) -> bool {
    i32x16::any(cast(self))
  }

  #[inline]
  pub fn all(self) -> bool {
    i32x16::all(cast(self))
  }

  ///
  /// Currently this function is never accelerated.
  #[inline]
  pub fn transpose(data: [u32x16; 16]) -> [u32x16; 16] {
    cast(i32x16::transpose(cast(data)))
  }
}

impl_simd_uint! {
  unsafe {
    T = u32,
    N = 16,
    Simd = u32x16,
    SignedSimd = i32x16,
    T_BITS = 32,
    T_BITS_MUL_2 = 64,
    [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15],
  }

  #[inline]
  fn not(self) -> Self::Output {
    pick! {
      if #[cfg(target_feature="avx512f")] {
        Self(bitxor_m512i(self.0, set_splat_i32_m512i(-1)))
      } else {
        Self(Inner(self.0.0.not(), self.0.1.not()))
      }
    }
  }

  #[inline]
  fn add(self, rhs: Self) -> Self::Output {
    pick! {
      if #[cfg(target_feature="avx512f")] {
        Self(add_i32_m512i(self.0, rhs.0))
      } else {
        Self(Inner(self.0.0.add(rhs.0.0), self.0.1.add(rhs.0.1)))
      }
    }
  }

  #[inline]
  fn sub(self, rhs: Self) -> Self::Output {
    pick! {
      if #[cfg(target_feature="avx512f")] {
        Self(sub_i32_m512i(self.0, rhs.0))
      } else {
        Self(Inner(self.0.0.sub(rhs.0.0), self.0.1.sub(rhs.0.1)))
      }
    }
  }

  #[inline]
  fn mul(self, rhs: Self) -> Self::Output {
    pick! {
      if #[cfg(target_feature="avx512f")] {
        Self(mul_i32_keep_low_m512i(self.0, rhs.0))
      } else {
        Self(Inner(self.0.0.mul(rhs.0.0), self.0.1.mul(rhs.0.1)))
      }
    }
  }

  #[inline]
  fn shl(self, rhs: u32x16) -> Self::Output {
    pick! {
      if #[cfg(target_feature="avx512f")] {
        let shift_by = bitand_m512i(rhs.0, set_splat_i32_m512i(31));
        Self(shl_each_u32_m512i(self.0, shift_by))
      } else {
        Self(Inner(self.0.0.shl(rhs.0.0), self.0.1.shl(rhs.0.1)))
      }
    }
  }

  #[inline]
  fn shl(self, rhs: u32) -> Self::Output {
    pick! {
      if #[cfg(target_feature="avx512f")] {
        // Use `rhs % 32` to perform wrapping shift and not unbounded shift.
        #[expect(clippy::suspicious_arithmetic_impl)]
        let shift = rhs & 31;
        Self(shl_all_u32_m512i(self.0, shift))
      } else {
        Self(Inner(self.0.0.shl(rhs), self.0.1.shl(rhs)))
      }
    }
  }

  #[inline]
  fn shr(self, rhs: u32x16) -> Self::Output {
    pick! {
      if #[cfg(target_feature="avx512f")] {
        let shift_by = bitand_m512i(rhs.0, set_splat_i32_m512i(31));
        Self(shr_each_u32_m512i(self.0, shift_by))
      } else {
        Self(Inner(self.0.0.shr(rhs.0.0), self.0.1.shr(rhs.0.1)))
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
        Self(shr_all_u32_m512i(self.0, shift))
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
        Self(max_u32_m512i(self.0, rhs.0))
      } else {
        Self(Inner(self.0.0.max(rhs.0.0), self.0.1.max(rhs.0.1)))
      }
    }
  }

  #[inline]
  pub fn min(self, rhs: Self) -> Self {
    pick! {
      if #[cfg(target_feature="avx512f")] {
        Self(min_u32_m512i(self.0, rhs.0))
      } else {
        Self(Inner(self.0.0.min(rhs.0.0), self.0.1.min(rhs.0.1)))
      }
    }
  }

  #[inline]
  pub fn reduce_add(self) -> u32 {
    let array: [u32x8; 2] = cast(self);
    (array[0] + array[1]).reduce_add()
  }

  #[inline]
  pub fn reduce_mul(self) -> u32 {
    let array: [u32x8; 2] = cast(self);
    (array[0] * array[1]).reduce_mul()
  }

  #[inline]
  pub fn reduce_max(self) -> u32 {
    let array: [u32x8; 2] = cast(self);
    array[0].max(array[1]).reduce_max()
  }

  #[inline]
  pub fn reduce_min(self) -> u32 {
    let array: [u32x8; 2] = cast(self);
    array[0].min(array[1]).reduce_min()
  }

  #[inline]
  pub fn unbounded_shl(self, rhs: Self) -> Self {
    pick! {
      if #[cfg(target_feature="avx512f")] {
        Self(shl_each_u32_m512i(self.0, rhs.0))
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
        Self(shl_all_u32_m512i(self.0, rhs))
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
        Self(shr_each_u32_m512i(self.0, rhs.0))
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
        Self(shr_all_u32_m512i(self.0, rhs))
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
    let (low, high) = self.mul_keep_low_high(rhs);
    let overflow = high.simd_ne(Self::ZERO);
    (low, overflow)
  }

  optional_fn_widening_mul {
    // Cannot have `widening_mul` because there is no `u64x16` type.
  }

  #[inline]
  pub fn mul_keep_low_high(self, rhs: Self) -> (Self, Self) {
    pick! {
      if #[cfg(all(target_feature="avx512f", target_feature="avx512dq"))] {
        #[cfg(target_arch = "x86")]
        use core::arch::x86::{_mm512_unpackhi_epi64, _mm512_unpacklo_epi64};
        #[cfg(target_arch = "x86_64")]
        use core::arch::x86_64::{_mm512_unpackhi_epi64, _mm512_unpacklo_epi64};

        let even_wide_mul = mul_u32_wide_m512i(self.0, rhs.0);
        let odd_wide_mul = mul_u32_wide_m512i(
          shuffle_i32_m512i::<0b_00_11_00_01>(self.0),
          shuffle_i32_m512i::<0b_00_11_00_01>(rhs.0),
        );

        let ll_hh_1 = unpack_low_i32_m512i(even_wide_mul, odd_wide_mul);
        let ll_hh_2 = unpack_high_i32_m512i(even_wide_mul, odd_wide_mul);
        // TODO(safe_arch): Add `_mm512_unpacklo_epi64` and `_mm512_unpackhi_epi64`.
        (
          Self(m512i(unsafe { _mm512_unpacklo_epi64(ll_hh_1.0, ll_hh_2.0) })),
          Self(m512i(unsafe { _mm512_unpackhi_epi64(ll_hh_1.0, ll_hh_2.0) })),
        )
      } else {
        let [self_a, self_b] = cast::<u32x16, [u32x8; 2]>(self);
        let [rhs_a, rhs_b] = cast::<u32x16, [u32x8; 2]>(rhs);

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
      if #[cfg(target_feature="avx512f")] {
        let alo = extract_m256i32_from_m512i::<0>(self.0);
        let ahi = extract_m256i32_from_m512i::<1>(self.0);
        let blo = extract_m256i32_from_m512i::<0>(rhs.0);
        let bhi = extract_m256i32_from_m512i::<1>(rhs.0);

        let lo_res: m256i = {
          let a8 = u32x8(alo);
          let b8 = u32x8(blo);
          a8.mul_keep_high(b8).0
        };
        let hi_res: m256i = {
          let a8 = u32x8(ahi);
          let b8 = u32x8(bhi);
          a8.mul_keep_high(b8).0
        };

        let zero = zeroed_m512i();
        let with_lo = insert_m256i32_to_m512i::<0>(zero, lo_res);
        Self(insert_m256i32_to_m512i::<1>(with_lo, hi_res))
      } else {
        Self(Inner(
          self.0.0.mul_keep_high(rhs.0.0),
          self.0.1.mul_keep_high(rhs.0.1),
        ))
      }
    }
  }
}

impl From<u16x16> for u32x16 {
  /// Widens and zero-extends each u16 lane to u32
  #[inline]
  fn from(v: u16x16) -> Self {
    pick! {
      if #[cfg(target_feature = "avx512f")] {
        Self(convert_to_u32_m512i_from_u16_m256i(v.0))
      } else if #[cfg(target_feature = "avx2")] {
        let lo: m128i = extract_m128i_from_m256i::<0>(v.0);
        let hi: m128i = extract_m128i_from_m256i::<1>(v.0);
        Self(Inner(
          u32x8(convert_to_i32_m256i_from_u16_m128i(lo)),
          u32x8(convert_to_i32_m256i_from_u16_m128i(hi)),
        ))
      } else if #[cfg(target_feature = "sse2")] {
        Self(Inner(
          u32x8(crate::u32x8_::Inner(
            u32x4(shr_imm_u32_m128i::<16>(unpack_low_i16_m128i(v.0.0.0, v.0.0.0))),
            u32x4(shr_imm_u32_m128i::<16>(unpack_high_i16_m128i(v.0.0.0, v.0.0.0))),
          )),
          u32x8(crate::u32x8_::Inner(
            u32x4(shr_imm_u32_m128i::<16>(unpack_low_i16_m128i(v.0.1.0, v.0.1.0))),
            u32x4(shr_imm_u32_m128i::<16>(unpack_high_i16_m128i(v.0.1.0, v.0.1.0))),
          )),
        ))
      } else {
        // Portable fallback
        let arr = v.as_array();
        Self::new([
          arr[0] as u32,  arr[1] as u32,  arr[2] as u32,  arr[3] as u32,
          arr[4] as u32,  arr[5] as u32,  arr[6] as u32,  arr[7] as u32,
          arr[8] as u32,  arr[9] as u32,  arr[10] as u32, arr[11] as u32,
          arr[12] as u32, arr[13] as u32, arr[14] as u32, arr[15] as u32,
        ])
      }
    }
  }
}
