use super::*;

use crate::{f32x8, i16x8, i32x4, i64x8, simd::SimdBackend, u16x8, u32x8};

#[cfg(not(target_feature = "avx2"))]
#[repr(C, align(32))]
#[derive(Default, Clone, Copy, PartialEq, Eq)]
pub(crate) struct Inner(pub i32x4, pub i32x4);

unsafe impl SimdBackend for i32x8 {
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
    T = i32,
    N = 8,
    Simd = i32x8,
    optional_type_x86_inner { X86Inner = __m256i },
    optional_type_arm_inner {},
    optional_type_wasm_inner {},
  }

  #[inline]
  fn simd_eq(self, rhs: Self) -> Self::Output {
    pick! {
      if #[cfg(target_feature="avx2")] {
        Self(cmp_eq_mask_i32_m256i(self.0, rhs.0))
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
        Self(cmp_gt_mask_i32_m256i(rhs.0, self.0))
      } else {
        Self(Inner(self.0.0.simd_lt(rhs.0.0), self.0.1.simd_lt(rhs.0.1)))
      }
    }
  }

  #[inline]
  fn simd_gt(self, rhs: Self) -> Self::Output {
    pick! {
      if #[cfg(target_feature="avx2")] {
        Self(cmp_gt_mask_i32_m256i(self.0, rhs.0))
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
      if #[cfg(target_feature="avx2")] {
        // use f32 move_mask since it is the same size as i32
        move_mask_m256(cast(self.0)) as u32
      } else {
        self.0.0.to_bitmask() | (self.0.1.to_bitmask() << 4)
      }
    }
  }

  #[inline]
  pub fn any(self) -> bool {
    pick! {
      if #[cfg(target_feature="avx2")] {
        move_mask_m256(cast(self.0)) != 0
      } else {
        (self.0.0 | self.0.1).any()
      }
    }
  }

  #[inline]
  pub fn all(self) -> bool {
    pick! {
      if #[cfg(target_feature="avx2")] {
        move_mask_m256(cast(self.0)) == 0b11111111
      } else {
        (self.0.0 & self.0.1).all()
      }
    }
  }

  ///
  /// Currently this function is only accelerated on `avx2`.
  #[inline]
  pub fn transpose(data: [i32x8; 8]) -> [i32x8; 8] {
    pick! {
      if #[cfg(target_feature="avx2")] {
        let a0 = unpack_low_i32_m256i(data[0].0, data[1].0);
        let a1 = unpack_high_i32_m256i(data[0].0, data[1].0);
        let a2 = unpack_low_i32_m256i(data[2].0, data[3].0);
        let a3 = unpack_high_i32_m256i(data[2].0, data[3].0);
        let a4 = unpack_low_i32_m256i(data[4].0, data[5].0);
        let a5 = unpack_high_i32_m256i(data[4].0, data[5].0);
        let a6 = unpack_low_i32_m256i(data[6].0, data[7].0);
        let a7 = unpack_high_i32_m256i(data[6].0, data[7].0);

        pub const fn mm_shuffle(z: i32, y: i32, x: i32, w: i32) -> i32 {
          (z << 6) | (y << 4) | (x << 2) | w
        }

        const SHUFF_LO : i32 = mm_shuffle(1,0,1,0);
        const SHUFF_HI : i32 = mm_shuffle(3,2,3,2);

        // possible todo: intel performance manual suggests alternative with blend to avoid port 5 pressure
        // (since blend runs on a different port than shuffle)
        let b0 = cast::<m256,m256i>(shuffle_m256::<SHUFF_LO>(cast(a0),cast(a2)));
        let b1 = cast::<m256,m256i>(shuffle_m256::<SHUFF_HI>(cast(a0),cast(a2)));
        let b2 = cast::<m256,m256i>(shuffle_m256::<SHUFF_LO>(cast(a1),cast(a3)));
        let b3 = cast::<m256,m256i>(shuffle_m256::<SHUFF_HI>(cast(a1),cast(a3)));
        let b4 = cast::<m256,m256i>(shuffle_m256::<SHUFF_LO>(cast(a4),cast(a6)));
        let b5 = cast::<m256,m256i>(shuffle_m256::<SHUFF_HI>(cast(a4),cast(a6)));
        let b6 = cast::<m256,m256i>(shuffle_m256::<SHUFF_LO>(cast(a5),cast(a7)));
        let b7 = cast::<m256,m256i>(shuffle_m256::<SHUFF_HI>(cast(a5),cast(a7)));

        [
          i32x8(permute2z_m256i::<0x20>(b0, b4)),
          i32x8(permute2z_m256i::<0x20>(b1, b5)),
          i32x8(permute2z_m256i::<0x20>(b2, b6)),
          i32x8(permute2z_m256i::<0x20>(b3, b7)),
          i32x8(permute2z_m256i::<0x31>(b0, b4)),
          i32x8(permute2z_m256i::<0x31>(b1, b5)),
          i32x8(permute2z_m256i::<0x31>(b2, b6)),
          i32x8(permute2z_m256i::<0x31>(b3, b7)),
        ]
      } else {
        // possible todo: not sure that 128bit SIMD gives us a a lot of speedup here

        #[inline(always)]
        fn transpose_column(data: &[i32x8; 8], index: usize) -> i32x8 {
          i32x8::new([
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
  }
}

impl_simd_int! {
  unsafe {
    T = i32,
    N = 8,
    Simd = i32x8,
    UnsignedSimd = u32x8,
    T_BITS = 32,
    T_BITS_MUL_2 = 64,
    [0, 1, 2, 3, 4, 5, 6, 7],
  }

  #[inline]
  fn shr(self, rhs: u32x8) -> Self::Output {
    pick! {
      if #[cfg(target_feature="avx2")] {
        // ensure same behavior as scalar
        let shift_by = bitand_m256i(rhs.0, set_splat_i32_m256i(31));
        Self(shr_each_i32_m256i(self.0, shift_by))
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
        Self(shr_all_i32_m256i(self.0, shift))
      } else {
        Self(Inner(self.0.0.shr(rhs), self.0.1.shr(rhs)))
      }
    }
  }

  #[inline]
  pub fn max(self, rhs: Self) -> Self {
    pick! {
      if #[cfg(target_feature="avx2")] {
        Self(max_i32_m256i(self.0, rhs.0))
      } else {
        Self(Inner(self.0.0.max(rhs.0.0), self.0.1.max(rhs.0.1)))
      }
    }
  }

  #[inline]
  pub fn min(self, rhs: Self) -> Self {
    pick! {
      if #[cfg(target_feature="avx2")] {
        Self(min_i32_m256i(self.0, rhs.0))
      } else {
        Self(Inner(self.0.0.min(rhs.0.0), self.0.1.min(rhs.0.1)))
      }
    }
  }

  #[inline]
  pub fn reduce_max(self) -> i32 {
    let arr: [i32x4; 2] = cast(self);
    arr[0].max(arr[1]).reduce_max()
  }

  #[inline]
  pub fn reduce_min(self) -> i32 {
    let arr: [i32x4; 2] = cast(self);
    arr[0].min(arr[1]).reduce_min()
  }

  #[inline]
  pub fn unbounded_shr(self, rhs: u32x8) -> Self {
    pick! {
      if #[cfg(target_feature="avx2")] {
        Self(shr_each_i32_m256i(self.0, rhs.0))
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
        Self(shr_all_i32_m256i(self.0, cast([rhs as u64, 0])))
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
    let (low, high) = self.mul_keep_low_high(rhs);
    let low = cast::<u32x8, i32x8>(low);

    let overflow = high.simd_ne(low.is_negative());
    (low, overflow)
  }

  optional_fn_widening_mul {
    #[inline]
    pub fn widening_mul(self, rhs: Self) -> i64x8 {
      pick! {
        if #[cfg(all(target_feature="avx512f", target_feature="avx2"))] {
          const SHUFFLE_INDICES: i64x8 = i64x8::new([0, 4, 1, 5, 2, 6, 3, 7]);

          let even_wide_mul = mul_i64_low_bits_m256i(self.0, rhs.0);
          let odd_wide_mul = mul_i64_low_bits_m256i(
            shuffle_ai_i32_half_m256i::<0b_00_11_00_01>(self.0),
            shuffle_ai_i32_half_m256i::<0b_00_11_00_01>(rhs.0),
          );
          let even_then_odd = cast::<[m256i; 2], m512i>([even_wide_mul, odd_wide_mul]);
          i64x8(permute_i64_m512i(SHUFFLE_INDICES.0, even_then_odd))
        } else if #[cfg(target_feature="avx2")] {
          let even_wide_mul = mul_i64_low_bits_m256i(self.0, rhs.0);
          let odd_wide_mul = mul_i64_low_bits_m256i(
            shuffle_ai_i32_half_m256i::<0b_00_11_00_01>(self.0),
            shuffle_ai_i32_half_m256i::<0b_00_11_00_01>(rhs.0),
          );
          let m0145 = unpack_low_i64_m256i(even_wide_mul, odd_wide_mul);
          let m2367 = unpack_high_i64_m256i(even_wide_mul, odd_wide_mul);

          cast([
            shuffle_abi_i128z_all_m256i::<0b_0010_0000>(m0145, m2367),
            shuffle_abi_i128z_all_m256i::<0b_0011_0001>(m0145, m2367),
          ])
        } else {
          let [self_a, self_b] = cast::<i32x8, [i32x4; 2]>(self);
          let [rhs_a, rhs_b] = cast::<i32x8, [i32x4; 2]>(rhs);

          cast([self_a.widening_mul(rhs_a), self_b.widening_mul(rhs_b)])
        }
      }
    }
  }

  #[inline]
  pub fn mul_keep_low_high(self, rhs: Self) -> (u32x8, i32x8) {
    pick! {
      if #[cfg(target_feature="avx2")] {
        let even_wide_mul = mul_i64_low_bits_m256i(self.0, rhs.0);
        let odd_wide_mul = mul_i64_low_bits_m256i(
          shuffle_ai_i32_half_m256i::<0b_00_11_00_01>(self.0),
          shuffle_ai_i32_half_m256i::<0b_00_11_00_01>(rhs.0),
        );
        let ll_hh_1 = unpack_low_i32_m256i(even_wide_mul, odd_wide_mul);
        let ll_hh_2 = unpack_high_i32_m256i(even_wide_mul, odd_wide_mul);

        (
          u32x8(unpack_low_i64_m256i(ll_hh_1, ll_hh_2)),
          i32x8(unpack_high_i64_m256i(ll_hh_1, ll_hh_2)),
        )
      } else {
        let [self_a, self_b] = cast::<i32x8, [i32x4; 2]>(self);
        let [rhs_a, rhs_b] = cast::<i32x8, [i32x4; 2]>(rhs);

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
      if #[cfg(target_feature="avx2")] {
        let even_wide_mul = mul_i64_low_bits_m256i(self.0, rhs.0);
        let odd_wide_mul = mul_i64_low_bits_m256i(
          shuffle_ai_i32_half_m256i::<0b_00_11_00_01>(self.0),
          shuffle_ai_i32_half_m256i::<0b_00_11_00_01>(rhs.0),
        );
        let ll_hh_1 = unpack_low_i32_m256i(even_wide_mul, odd_wide_mul);
        let ll_hh_2 = unpack_high_i32_m256i(even_wide_mul, odd_wide_mul);

        Self(unpack_high_i64_m256i(ll_hh_1, ll_hh_2))
      } else {
        let [self_a, self_b] = cast::<i32x8, [i32x4; 2]>(self);
        let [rhs_a, rhs_b] = cast::<i32x8, [i32x4; 2]>(rhs);

        cast([self_a.mul_keep_high(rhs_a), self_b.mul_keep_high(rhs_b)])
      }
    }
  }

  #[inline]
  pub fn abs(self) -> Self {
    pick! {
      if #[cfg(target_feature="avx2")] {
        Self(abs_i32_m256i(self.0))
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

impl From<i16x8> for i32x8 {
  #[inline]
  fn from(value: i16x8) -> Self {
    i32x8::from_i16x8(value)
  }
}

/// The following functionality exists only for [`i32x8`], or only for
/// particular types inconsistently.
impl i32x8 {
  /// Converts each element from [`i16`] to [`i32`].
  #[inline]
  #[must_use]
  pub fn from_i16x8(v: i16x8) -> Self {
    pick! {
      if #[cfg(target_feature="avx2")] {
        Self(convert_to_i32_m256i_from_i16_m128i(v.0))
      } else if #[cfg(target_feature="sse2")] {
        Self(Inner(
          Simd(shr_imm_i32_m128i::<16>( unpack_low_i16_m128i(v.0, v.0))),
          Simd(shr_imm_i32_m128i::<16>( unpack_high_i16_m128i(v.0, v.0))),
        ))
      } else {
        i32x8::new([
          i32::from(v.as_array()[0]),
          i32::from(v.as_array()[1]),
          i32::from(v.as_array()[2]),
          i32::from(v.as_array()[3]),
          i32::from(v.as_array()[4]),
          i32::from(v.as_array()[5]),
          i32::from(v.as_array()[6]),
          i32::from(v.as_array()[7]),
        ])
      }
    }
  }

  /// Converts each element from [`u16`] to [`i32`].
  #[inline]
  #[must_use]
  pub fn from_u16x8(v: u16x8) -> Self {
    pick! {
      if #[cfg(target_feature="avx2")] {
        Self(convert_to_i32_m256i_from_u16_m128i(v.0))
      } else if #[cfg(target_feature="sse2")] {
        Self(Inner(
          Simd(shr_imm_u32_m128i::<16>( unpack_low_i16_m128i(v.0, v.0))),
          Simd(shr_imm_u32_m128i::<16>( unpack_high_i16_m128i(v.0, v.0))),
        ))
      } else {
        i32x8::new([
          i32::from(v.as_array()[0]),
          i32::from(v.as_array()[1]),
          i32::from(v.as_array()[2]),
          i32::from(v.as_array()[3]),
          i32::from(v.as_array()[4]),
          i32::from(v.as_array()[5]),
          i32::from(v.as_array()[6]),
          i32::from(v.as_array()[7]),
        ])
      }
    }
  }

  /// Converts each element from [`i32`] to [`f32`].
  #[inline]
  #[must_use]
  pub fn round_float(self) -> f32x8 {
    pick! {
      if #[cfg(target_feature="avx2")] {
        cast(convert_to_m256_from_i32_m256i(self.0))
      } else {
        cast([
          self.0.0.round_float(),
          self.0.1.round_float(),
        ])
      }
    }
  }
}
