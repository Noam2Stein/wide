use super::*;

use crate::{i32x8, simd::SimdBackend, u16x8, u32x4, u64x8};

#[cfg(not(target_feature = "avx2"))]
#[repr(C, align(32))]
#[derive(Default, Clone, Copy, PartialEq, Eq)]
pub(crate) struct Inner(pub u32x4, pub u32x4);

unsafe impl SimdBackend for u32x8 {
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
        Self(cmp_eq_mask_i32_m256i(self.0, rhs.0))
      } else {
        Self(Inner(self.0.0.simd_eq(rhs.0.0), self.0.1.simd_eq(rhs.0.1)))
      }
    }
  }

  #[inline]
  fn simd_ne(self, rhs: Self) -> Self {
    !self.simd_eq(rhs)
  }

  #[inline]
  fn simd_lt(self, rhs: Self) -> Self {
    // lt is just gt the other way around
    rhs.simd_gt(self)
  }

  #[inline]
  fn simd_gt(self, rhs: Self) -> Self {
    pick! {
      if #[cfg(target_feature="avx2")] {
        // no unsigned gt than so inverting the high bit will get the correct result
        let highbit = u32x8::splat(1 << 31);
        Self(cmp_gt_mask_i32_m256i((self ^ highbit).0, (rhs ^ highbit).0))
      } else {
        Self(Inner(self.0.0.simd_gt(rhs.0.0), self.0.1.simd_gt(rhs.0.1)))
      }
    }
  }

  #[inline]
  fn simd_le(self, rhs: Self) -> Self {
    self.simd_eq(rhs) | self.simd_lt(rhs)
  }

  #[inline]
  fn simd_ge(self, rhs: Self) -> Self {
    self.simd_eq(rhs) | self.simd_gt(rhs)
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
  fn to_bitmask(self) -> u32 {
    i32x8::to_bitmask(cast(self))
  }

  #[inline]
  fn any(self) -> bool {
    pick! {
      if #[cfg(target_feature="avx2")] {
        ((move_mask_i8_m256i(self.0) as u32) & 0b10001000100010001000100010001000) != 0
      } else {
        (self.0.0 | self.0.1).any()
      }
    }
  }

  #[inline]
  fn all(self) -> bool {
    pick! {
      if #[cfg(target_feature="avx2")] {
        ((move_mask_i8_m256i(self.0) as u32) & 0b10001000100010001000100010001000) == 0b10001000100010001000100010001000
      } else {
        (self.0.0 & self.0.1).all()
      }
    }
  }

  #[inline]
  fn transpose(data: [u32x8; 8]) -> [u32x8; 8] {
    cast(i32x8::transpose(cast(data)))
  }
}

impl_simd_uint! {
  unsafe {
    T = u32,
    N = 8,
    Simd = u32x8,
    SignedSimd = i32x8,
    T_BITS = 32,
    T_BITS_MUL_2 = 64,
    [0, 1, 2, 3, 4, 5, 6, 7],
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
        Self(add_i32_m256i(self.0, rhs.0))
      } else {
        Self(Inner(self.0.0.add(rhs.0.0), self.0.1.add(rhs.0.1)))
      }
    }
  }

  #[inline]
  fn sub(self, rhs: Self) -> Self::Output {
    pick! {
      if #[cfg(target_feature="avx2")] {
        Self(sub_i32_m256i(self.0, rhs.0))
      } else {
        Self(Inner(self.0.0.sub(rhs.0.0), self.0.1.sub(rhs.0.1)))
      }
    }
  }

  #[inline]
  fn mul(self, rhs: Self) -> Self::Output {
    pick! {
      if #[cfg(target_feature="avx2")] {
        Self(mul_i32_keep_low_m256i(self.0, rhs.0))
      } else {
        Self(Inner(self.0.0.mul(rhs.0.0), self.0.1.mul(rhs.0.1)))
      }
    }
  }

  #[inline]
  fn shl(self, rhs: u32x8) -> Self::Output {
    pick! {
      if #[cfg(target_feature="avx2")] {
        // ensure same behavior as scalar wrapping_shl
        let shift_by = bitand_m256i(rhs.0, set_splat_i32_m256i(31));
        Self(shl_each_u32_m256i(self.0, shift_by))
      } else {
        Self(Inner(self.0.0.shl(rhs.0.0), self.0.1.shl(rhs.0.1)))
      }
    }
  }

  #[inline]
  fn shl(self, rhs: u32) -> Self::Output {
    pick! {
      if #[cfg(target_feature="avx2")] {
        // Use `rhs % 32` to perform wrapping shift and not unbounded shift.
        #[expect(clippy::suspicious_arithmetic_impl)]
        let shift = cast([rhs as u64 & 31, 0]);
        Self(shl_all_u32_m256i(self.0, shift))
      } else {
        Self(Inner(self.0.0.shl(rhs), self.0.1.shl(rhs)))
      }
    }
  }

  #[inline]
  fn shr(self, rhs: u32x8) -> Self::Output {
    pick! {
      if #[cfg(target_feature="avx2")] {
        // ensure same behavior as scalar wrapping_shr
        let shift_by = bitand_m256i(rhs.0, set_splat_i32_m256i(31));
        Self(shr_each_u32_m256i(self.0, shift_by ))
      } else {
        Self(Inner(self.0.0.shr(rhs.0.0), self.0.1.shr(rhs.0.1)))
      }
    }
  }

  #[inline]
  fn shr(self, rhs: u32) -> Self::Output {
    pick! {
      if #[cfg(target_feature="avx2")] {
        // Use `rhs % 32` to perform wrapping shift and not unbounded shift.
        #[expect(clippy::suspicious_arithmetic_impl)]
        let shift = cast([rhs as u64 & 31, 0]);
        Self(shr_all_u32_m256i(self.0, shift))
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
        Self(max_u32_m256i(self.0, rhs.0))
      } else {
        Self(Inner(self.0.0.max(rhs.0.0), self.0.1.max(rhs.0.1)))
      }
    }
  }

  #[inline]
  pub fn min(self, rhs: Self) -> Self {
    pick! {
      if #[cfg(target_feature="avx2")] {
        Self(min_u32_m256i(self.0, rhs.0))
      } else {
        Self(Inner(self.0.0.min(rhs.0.0), self.0.1.min(rhs.0.1)))
      }
    }
  }

  #[inline]
  pub fn reduce_add(self) -> u32 {
    let array: [u32x4; 2] = cast(self);
    (array[0] + array[1]).reduce_add()
  }

  #[inline]
  pub fn reduce_mul(self) -> u32 {
    let array: [u32x4; 2] = cast(self);
    (array[0] * array[1]).reduce_mul()
  }

  #[inline]
  pub fn reduce_max(self) -> u32 {
    let array: [u32x4; 2] = cast(self);
    array[0].max(array[1]).reduce_max()
  }

  #[inline]
  pub fn reduce_min(self) -> u32 {
    let array: [u32x4; 2] = cast(self);
    array[0].min(array[1]).reduce_min()
  }

  #[inline]
  pub fn unbounded_shl(self, rhs: Self) -> Self {
    pick! {
      if #[cfg(target_feature="avx2")] {
        Self(shl_each_u32_m256i(self.0, rhs.0))
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
        Self(shl_all_u32_m256i(self.0, cast([rhs as u64, 0])))
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
        Self(shr_each_u32_m256i(self.0, rhs.0))
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
        Self(shr_all_u32_m256i(self.0, cast([rhs as u64, 0])))
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
    let (low, high) = self.mul_keep_low_high(rhs);
    let overflow = high.simd_ne(Self::ZERO);
    (low, overflow)
  }

  optional_fn_widening_mul {
    #[inline]
    pub fn widening_mul(self, rhs: Self) -> u64x8 {
      pick! {
        if #[cfg(all(target_feature="avx512f", target_feature="avx2"))] {
          const SHUFFLE_INDICES: i64x8 = i64x8::new([0, 4, 1, 5, 2, 6, 3, 7]);

          let even_wide_mul = mul_u64_low_bits_m256i(self.0, rhs.0);
          let odd_wide_mul = mul_u64_low_bits_m256i(
            shuffle_ai_i32_half_m256i::<0b_00_11_00_01>(self.0),
            shuffle_ai_i32_half_m256i::<0b_00_11_00_01>(rhs.0),
          );
          let even_then_odd = cast::<[m256i; 2], m512i>([even_wide_mul, odd_wide_mul]);
          u64x8(permute_i64_m512i(SHUFFLE_INDICES.0, even_then_odd))
        } else {
          let [self_a, self_b] = cast::<u32x8, [u32x4; 2]>(self);
          let [rhs_a, rhs_b] = cast::<u32x8, [u32x4; 2]>(rhs);

          cast([self_a.widening_mul(rhs_a), self_b.widening_mul(rhs_b)])
        }
      }
    }
  }

  #[inline]
  pub fn mul_keep_low_high(self, rhs: Self) -> (Self, Self) {
    pick! {
      if #[cfg(target_feature="avx2")] {
        let even_wide_mul = mul_u64_low_bits_m256i(self.0, rhs.0);
        let odd_wide_mul = mul_u64_low_bits_m256i(
          shuffle_ai_i32_half_m256i::<0b_00_11_00_01>(self.0),
          shuffle_ai_i32_half_m256i::<0b_00_11_00_01>(rhs.0),
        );
        let ll_hh_1 = unpack_low_i32_m256i(even_wide_mul, odd_wide_mul);
        let ll_hh_2 = unpack_high_i32_m256i(even_wide_mul, odd_wide_mul);
        (
          Self(unpack_low_i64_m256i(ll_hh_1, ll_hh_2)),
          Self(unpack_high_i64_m256i(ll_hh_1, ll_hh_2)),
        )
      } else {
        let [self_a, self_b] = cast::<u32x8, [u32x4; 2]>(self);
        let [rhs_a, rhs_b] = cast::<u32x8, [u32x4; 2]>(rhs);

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
  pub fn mul_keep_high(self, rhs: u32x8) -> u32x8 {
    pick! {
      if #[cfg(target_feature="avx2")] {
        let a : [u32;8]= cast(self);
        let b : [u32;8]= cast(rhs);

        // let the compiler shuffle the values around, it does the right thing
        let r1 : [u32;8] = cast(mul_u64_low_bits_m256i(cast([a[0], 0, a[1], 0, a[2], 0, a[3], 0]), cast([b[0], 0, b[1], 0, b[2], 0, b[3], 0])));
        let r2 : [u32;8] = cast(mul_u64_low_bits_m256i(cast([a[4], 0, a[5], 0, a[6], 0, a[7], 0]), cast([b[4], 0, b[5], 0, b[6], 0, b[7], 0])));

        cast([r1[1], r1[3], r1[5], r1[7], r2[1], r2[3], r2[5], r2[7]])
      } else {
        Self(Inner(
          self.0.0.mul_keep_high(rhs.0.0),
          self.0.1.mul_keep_high(rhs.0.1),
        ))
      }
    }
  }
}

impl From<u16x8> for u32x8 {
  /// widens and zero extends to u32x8
  #[inline]
  fn from(v: u16x8) -> Self {
    pick! {
      if #[cfg(target_feature="avx2")] {
        Self(convert_to_i32_m256i_from_u16_m128i(v.0))
      } else if #[cfg(target_feature="sse2")] {
        Self(Inner(
          Simd(shr_imm_u32_m128i::<16>( unpack_low_i16_m128i(v.0, v.0))),
          Simd(shr_imm_u32_m128i::<16>( unpack_high_i16_m128i(v.0, v.0))),
        ))
      } else {
        u32x8::new([
          u32::from(v.as_array()[0]),
          u32::from(v.as_array()[1]),
          u32::from(v.as_array()[2]),
          u32::from(v.as_array()[3]),
          u32::from(v.as_array()[4]),
          u32::from(v.as_array()[5]),
          u32::from(v.as_array()[6]),
          u32::from(v.as_array()[7]),
        ])
      }
    }
  }
}
