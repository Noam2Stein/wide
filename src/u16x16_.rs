use super::*;

use crate::{Simd, i16x16, simd::SimdBackend, u8x16, u16x8, u32x16};

#[cfg(not(target_feature = "avx2"))]
#[repr(C, align(32))]
#[derive(Default, Clone, Copy, PartialEq, Eq)]
pub(crate) struct Inner(pub u16x8, pub u16x8);

unsafe impl SimdBackend for u16x16 {
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
    T = u16,
    N = 16,
    Simd = u16x16,
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
    // no gt, so just reverse to get same answer
    Self::simd_gt(rhs, self)
  }

  #[inline]
  fn simd_gt(self, rhs: Self) -> Self::Output {
    pick! {
      if #[cfg(target_feature = "avx2")] {
        let bias = m256i::from([0x8000u16; 16]);
        let a_biased = sub_i16_m256i(self.0, bias);
        let b_biased = sub_i16_m256i(rhs.0, bias);
        Self(cmp_gt_mask_i16_m256i(a_biased, b_biased))
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
    i16x16::to_bitmask(cast(self))
  }

  #[inline]
  pub fn any(self) -> bool {
    i16x16::any(cast(self))
  }

  #[inline]
  pub fn all(self) -> bool {
    i16x16::all(cast(self))
  }

  ///
  /// Currently this function is never accelerated.
  #[inline]
  pub fn transpose(data: [u16x16; 16]) -> [u16x16; 16] {
    cast(i16x16::transpose(cast(data)))
  }
}

impl_simd_uint! {
  unsafe {
    T = u16,
    N = 16,
    Simd = u16x16,
    SignedSimd = i16x16,
    T_BITS = 16,
    T_BITS_MUL_2 = 32,
    [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15],
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
        Self(add_i16_m256i(self.0, rhs.0))
      } else {
        Self(Inner(self.0.0.add(rhs.0.0), self.0.1.add(rhs.0.1)))
      }
    }
  }

  #[inline]
  fn sub(self, rhs: Self) -> Self::Output {
    pick! {
      if #[cfg(target_feature="avx2")] {
        Self(sub_i16_m256i(self.0, rhs.0))
      } else {
        Self(Inner(self.0.0.sub(rhs.0.0), self.0.1.sub(rhs.0.1)))
      }
    }
  }

  #[inline]
  fn mul(self, rhs: Self) -> Self::Output {
    pick! {
      if #[cfg(target_feature="avx2")] {
        // non-widening multiplication is the same for unsigned and signed
        Self(mul_i16_keep_low_m256i(self.0, rhs.0))
      } else {
        Self(Inner(self.0.0.mul(rhs.0.0), self.0.1.mul(rhs.0.1)))
      }
    }
  }

  #[inline]
  fn shl(self, rhs: Self) -> Self::Output {
    pick! {
      if #[cfg(all(target_feature="avx512bw", target_feature="avx512vl"))] {
        #[cfg(target_arch = "x86")]
        use core::arch::x86::_mm256_sllv_epi16;
        #[cfg(target_arch = "x86_64")]
        use core::arch::x86_64::_mm256_sllv_epi16;

        // Mask `rhs` to 15 to match `wrapping_shl`.
        let rhs = bitand_m256i(rhs.0, set_splat_i16_m256i(15));
        // TODO(safe_arch): Add `_mm256_sllv_epi16`.
        cast(unsafe { _mm256_sllv_epi16(self.0.0, rhs.0) })
      } else {
        let [self_a, self_b]: [u16x8; 2] = cast(self);
        let [rhs_a, rhs_b]: [u16x8; 2] = cast(rhs);

        cast([self_a << rhs_a, self_b << rhs_b])
      }
    }
  }

  #[inline]
  fn shl(self, rhs: u32) -> Self::Output {
    pick! {
      if #[cfg(target_feature="avx2")] {
        // Use `rhs % 16` to perform wrapping shift and not unbounded shift.
        #[expect(clippy::suspicious_arithmetic_impl)]
        let shift = cast([rhs as u64 & 15, 0]);
        Self(shl_all_u16_m256i(self.0, shift))
      } else {
        Self(Inner(self.0.0.shl(rhs), self.0.1.shl(rhs)))
      }
    }
  }

  #[inline]
  fn shr(self, rhs: Self) -> Self::Output {
    pick! {
      if #[cfg(all(target_feature="avx512bw", target_feature="avx512vl"))] {
        #[cfg(target_arch = "x86")]
        use core::arch::x86::_mm256_srlv_epi16;
        #[cfg(target_arch = "x86_64")]
        use core::arch::x86_64::_mm256_srlv_epi16;

        // Mask `rhs` to 15 to match `wrapping_shr`.
        let rhs = bitand_m256i(rhs.0, set_splat_i16_m256i(15));
        // TODO(safe_arch): Add `_mm256_srlv_epi16`.
        cast(unsafe { _mm256_srlv_epi16(self.0.0, rhs.0) })
      } else {
        let [self_a, self_b]: [u16x8; 2] = cast(self);
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
        Self(shr_all_u16_m256i(self.0, shift))
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
    pick! {
      if #[cfg(target_feature="avx2")] {
        Self(max_u16_m256i(self.0, rhs.0))
      } else {
        Self(Inner(self.0.0.max(rhs.0.0), self.0.1.max(rhs.0.1)))
      }
    }
  }

  #[inline]
  pub fn min(self, rhs: Self) -> Self {
    pick! {
      if #[cfg(target_feature="avx2")] {
        Self(min_u16_m256i(self.0, rhs.0))
      } else {
        Self(Inner(self.0.0.min(rhs.0.0), self.0.1.min(rhs.0.1)))
      }
    }
  }

  #[inline]
  pub fn reduce_add(self) -> u16 {
    let array: [u16x8; 2] = cast(self);
    (array[0] + array[1]).reduce_add()
  }

  #[inline]
  pub fn reduce_mul(self) -> u16 {
    let array: [u16x8; 2] = cast(self);
    (array[0] * array[1]).reduce_mul()
  }

  #[inline]
  pub fn reduce_max(self) -> u16 {
    let array: [u16x8; 2] = cast(self);
    array[0].max(array[1]).reduce_max()
  }

  #[inline]
  pub fn reduce_min(self) -> u16 {
    let array: [u16x8; 2] = cast(self);
    array[0].min(array[1]).reduce_min()
  }

  #[inline]
  pub fn unbounded_shl(self, rhs: Self) -> Self {
    pick! {
      if #[cfg(all(target_feature="avx512bw", target_feature="avx512vl"))] {
        #[cfg(target_arch = "x86")]
        use core::arch::x86::_mm256_sllv_epi16;
        #[cfg(target_arch = "x86_64")]
        use core::arch::x86_64::_mm256_sllv_epi16;

        // TODO(safe_arch): Add `_mm256_sllv_epi16`.
        cast(unsafe { _mm256_sllv_epi16(self.0.0, rhs.0.0) })
      } else {
        let [self_a, self_b] = cast::<u16x16, [u16x8; 2]>(self);
        let [rhs_a, rhs_b] = cast::<u16x16, [u16x8; 2]>(rhs);

        cast([self_a.unbounded_shl(rhs_a), self_b.unbounded_shl(rhs_b)])
      }
    }
  }

  #[inline]
  pub fn unbounded_shl_scalar(self, rhs: u32) -> Self {
    pick! {
      if #[cfg(target_feature="avx2")] {
        Self(shl_all_u16_m256i(self.0, cast([rhs as u64, 0])))
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
      if #[cfg(all(target_feature="avx512bw", target_feature="avx512vl"))] {
        #[cfg(target_arch = "x86")]
        use core::arch::x86::_mm256_srlv_epi16;
        #[cfg(target_arch = "x86_64")]
        use core::arch::x86_64::_mm256_srlv_epi16;

        // TODO(safe_arch): Add `_mm256_srlv_epi16`.
        cast(unsafe { _mm256_srlv_epi16(self.0.0, rhs.0.0) })
      } else {
        let [self_a, self_b] = cast::<u16x16, [u16x8; 2]>(self);
        let [rhs_a, rhs_b] = cast::<u16x16, [u16x8; 2]>(rhs);

        cast([self_a.unbounded_shr(rhs_a), self_b.unbounded_shr(rhs_b)])
      }
    }
  }

  #[inline]
  pub fn unbounded_shr_scalar(self, rhs: u32) -> Self {
    pick! {
      if #[cfg(target_feature="avx2")] {
        Self(shr_all_u16_m256i(self.0, cast([rhs as u64, 0])))
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
        Self(add_saturating_u16_m256i(self.0, rhs.0))
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
        Self(sub_saturating_u16_m256i(self.0, rhs.0))
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
    pub fn widening_mul(self, rhs: Self) -> u32x16 {
      // x86 has no `_mm256_mul_epu16` intrinsic so there is no `avx2`
      // optimization.

      let [self_a, self_b] = cast::<u16x16, [u16x8; 2]>(self);
      let [rhs_a, rhs_b] = cast::<u16x16, [u16x8; 2]>(rhs);

      cast([self_a.widening_mul(rhs_a), self_b.widening_mul(rhs_b)])
    }
  }

  #[inline]
  pub fn mul_keep_low_high(self, rhs: Self) -> (Self, Self) {
    // x86 has no `_mm256_mul_epu16` intrinsic so there is no `avx2`
    // optimization.

    let [self_a, self_b] = cast::<u16x16, [u16x8; 2]>(self);
    let [rhs_a, rhs_b] = cast::<u16x16, [u16x8; 2]>(rhs);

    let result_a = self_a.mul_keep_low_high(rhs_a);
    let result_b = self_b.mul_keep_low_high(rhs_b);
    (cast([result_a.0, result_b.0]), cast([result_a.1, result_b.1]))
  }

  #[inline]
  pub fn mul_keep_high(self, rhs: Self) -> Self {
    // x86 has no `_mm256_mul_epu16` intrinsic so there is no `avx2`
    // optimization.

    let [self_a, self_b] = cast::<u16x16, [u16x8; 2]>(self);
    let [rhs_a, rhs_b] = cast::<u16x16, [u16x8; 2]>(rhs);

    cast([self_a.mul_keep_high(rhs_a), self_b.mul_keep_high(rhs_b)])
  }
}

impl From<u8x16> for u16x16 {
  /// widens and sign extends to u16x16
  #[inline]
  fn from(v: u8x16) -> Self {
    pick! {
      if #[cfg(target_feature="avx2")] {
        Self(convert_to_i16_m256i_from_u8_m128i(v.0))
      } else if #[cfg(target_feature="sse2")] {
        Self(Inner(
          Simd(shr_imm_u16_m128i::<8>( unpack_low_i8_m128i(v.0, v.0))),
          Simd(shr_imm_u16_m128i::<8>( unpack_high_i8_m128i(v.0, v.0))),
        ))
      } else {

        u16x16::new([
          v.as_array()[0] as u16,
          v.as_array()[1] as u16,
          v.as_array()[2] as u16,
          v.as_array()[3] as u16,
          v.as_array()[4] as u16,
          v.as_array()[5] as u16,
          v.as_array()[6] as u16,
          v.as_array()[7] as u16,
          v.as_array()[8] as u16,
          v.as_array()[9] as u16,
          v.as_array()[10] as u16,
          v.as_array()[11] as u16,
          v.as_array()[12] as u16,
          v.as_array()[13] as u16,
          v.as_array()[14] as u16,
          v.as_array()[15] as u16,
          ])
      }
    }
  }
}
