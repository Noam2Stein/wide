use super::*;

use crate::{i8x16, i16x32, simd::SimdBackend, u8x16, u8x32};

#[cfg(not(target_feature = "avx2"))]
#[repr(C, align(32))]
#[derive(Default, Clone, Copy, PartialEq, Eq)]
pub(crate) struct Inner(pub i8x16, pub i8x16);

unsafe impl SimdBackend for i8x32 {
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
        Self(cmp_eq_mask_i8_m256i(self.0,rhs.0))
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
    rhs.simd_gt(self)
  }

  #[inline]
  fn simd_gt(self, rhs: Self) -> Self {
    pick! {
      if #[cfg(target_feature="avx2")] {
        Self(cmp_gt_mask_i8_m256i(self.0,rhs.0))
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
    pick! {
      if #[cfg(target_feature="avx2")] {
        move_mask_i8_m256i(self.0) as u32
      } else {
        self.0.0.to_bitmask() | (self.0.1.to_bitmask() << 16)
      }
    }
  }

  #[inline]
  fn any(self) -> bool {
    pick! {
      if #[cfg(target_feature="avx2")] {
        move_mask_i8_m256i(self.0) != 0
      } else {
        (self.0.0 | self.0.1).any()
      }
    }
  }

  #[inline]
  fn all(self) -> bool {
    pick! {
      if #[cfg(target_feature="avx2")] {
        move_mask_i8_m256i(self.0) == -1
      } else {
        (self.0.0 & self.0.1).all()
      }
    }
  }

  #[inline]
  fn transpose(data: [i8x32; 32]) -> [i8x32; 32] {
    // Can this be optimized?

    #[inline(always)]
    fn transpose_column(data: &[i8x32; 32], index: usize) -> i8x32 {
      i8x32::new([
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
    T = i8,
    N = 32,
    Simd = i8x32,
    UnsignedSimd = u8x32,
    T_BITS = 8,
    T_BITS_MUL_2 = 16,
    [
      0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20,
      21, 22, 23, 24, 25, 26, 27, 28, 29, 30, 31
    ],
  }

  #[inline]
  fn shr(self, rhs: u8x32) -> Self::Output {
    // For x86, this technically can be done explicitly by converting to `i16`
    // or `i32` then converting back after multiplication, but that may not
    // actually be faster than auto-vectorization.
    let [self_a, self_b]: [i8x16; 2] = cast(self);
    let [rhs_a, rhs_b]: [u8x16; 2] = cast(rhs);
    cast([self_a >> rhs_a, self_b >> rhs_b])
  }

  #[inline]
  fn shr(self, rhs: u32) -> Self::Output {
    // For x86, this technically can be done explicitly by converting
    // to `i16` or `i32` then converting back after multiplication, but that
    // may not actually be faster than auto-vectorization.
    let [self_a, self_b]: [i8x16; 2] = cast(self);
    cast([self_a >> rhs, self_b >> rhs])
  }

  #[inline]
  pub fn max(self, rhs: Self) -> Self {
    pick! {
      if #[cfg(target_feature="avx2")] {
        Self(max_i8_m256i(self.0,rhs.0))
      } else {
        Self(Inner(self.0.0.max(rhs.0.0), self.0.1.max(rhs.0.1)))
      }
    }
  }

  #[inline]
  pub fn min(self, rhs: Self) -> Self {
    pick! {
      if #[cfg(target_feature="avx2")] {
        Self(min_i8_m256i(self.0,rhs.0))
      } else {
        Self(Inner(self.0.0.min(rhs.0.0), self.0.1.min(rhs.0.1)))
      }
    }
  }

  #[inline]
  pub fn reduce_max(self) -> i8 {
    let array: [i8x16; 2] = cast(self);
    array[0].max(array[1]).reduce_max()
  }

  #[inline]
  pub fn reduce_min(self) -> i8 {
    let array: [i8x16; 2] = cast(self);
    array[0].min(array[1]).reduce_min()
  }

  #[inline]
  pub fn unbounded_shr(self, rhs: u8x32) -> Self {
    // For x86, this technically can be done explicitly by converting to `i16`
    // or `i32` then converting back after multiplication, but that may not
    // actually be faster than auto-vectorization.
    let [self_a, self_b] = cast::<i8x32, [i8x16; 2]>(self);
    let [rhs_a, rhs_b] = cast::<u8x32, [u8x16; 2]>(rhs);
    cast([self_a.unbounded_shr(rhs_a), self_b.unbounded_shr(rhs_b)])
  }

  #[inline]
  pub fn unbounded_shr_scalar(self, rhs: u32) -> Self {
    // For x86, this technically can be done explicitly by converting
    // to `i16` or `i32` then converting back after multiplication, but that
    // may not actually be faster than auto-vectorization.
    let [self_a, self_b] = cast::<i8x32, [i8x16; 2]>(self);
    cast([self_a.unbounded_shr_scalar(rhs), self_b.unbounded_shr_scalar(rhs)])
  }

  #[inline]
  pub fn saturating_add(self, rhs: Self) -> Self {
    pick! {
      if #[cfg(target_feature="avx2")] {
        Self(add_saturating_i8_m256i(self.0, rhs.0))
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
        Self(sub_saturating_i8_m256i(self.0, rhs.0))
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
    let low = cast::<u8x32, i8x32>(low);

    let overflow = high.simd_ne(low.is_negative());
    (low, overflow)
  }

  optional_fn_widening_mul {
    #[inline]
    pub fn widening_mul(self, rhs: Self) -> i16x32 {
      // x86 has no `_mm256_mul_epi8` intrinsic so there is no `avx2`
      // optimization.

      let [self_a, self_b] = cast::<i8x32, [i8x16; 2]>(self);
      let [rhs_a, rhs_b] = cast::<i8x32, [i8x16; 2]>(rhs);

      cast([self_a.widening_mul(rhs_a), self_b.widening_mul(rhs_b)])
    }
  }

  #[inline]
  pub fn mul_keep_low_high(self, rhs: Self) -> (u8x32, i8x32) {
    // x86 has no `_mm256_mul_epi8` intrinsic so there is no `avx2`
    // optimization.

    let [self_a, self_b] = cast::<i8x32, [i8x16; 2]>(self);
    let [rhs_a, rhs_b] = cast::<i8x32, [i8x16; 2]>(rhs);

    let result_a = self_a.mul_keep_low_high(rhs_a);
    let result_b = self_b.mul_keep_low_high(rhs_b);
    (cast([result_a.0, result_b.0]), cast([result_a.1, result_b.1]))
  }

  #[inline]
  pub fn mul_keep_high(self, rhs: Self) -> Self {
    // x86 has no `_mm256_mul_epi8` intrinsic so there is no `avx2`
    // optimization.

    let [self_a, self_b] = cast::<i8x32, [i8x16; 2]>(self);
    let [rhs_a, rhs_b] = cast::<i8x32, [i8x16; 2]>(rhs);

    cast([self_a.mul_keep_high(rhs_a), self_b.mul_keep_high(rhs_b)])
  }

  #[inline]
  pub fn abs(self) -> Self {
    pick! {
      if #[cfg(target_feature="avx2")] {
        Self(abs_i8_m256i(self.0))
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

/// The following functionality exists only for [`i8x32`], or only for
/// particular types inconsistently.
impl i8x32 {
  /// Returns a new vector with lanes selected from the lanes of the first input
  /// vector a specified in the second input vector `rhs`.
  /// The indices i in range `[0, 15]` select the i-th element of `self`. For
  /// indices outside of the range the resulting lane is `0`.
  ///
  /// This note that is the equivalent of two parallel swizzle operations on the
  /// two halves of the vector, and the indexes each refer to the
  /// corresponding half.
  #[inline]
  pub fn swizzle_half(self, rhs: i8x32) -> i8x32 {
    pick! {
      if #[cfg(target_feature="avx2")] {
        Self(shuffle_av_i8z_half_m256i(self.0, add_saturating_u8_m256i(rhs.0, set_splat_i8_m256i(0x70))))
      } else {
        Self(Inner(self.0.0.swizzle(rhs.0.0), self.0.1.swizzle(rhs.0.1)))
      }
    }
  }

  /// Indices in the range `[0, 15]` will select the i-th element of `self`. If
  /// the high bit of any element of `rhs` is set (negative) then the
  /// corresponding output lane is guaranteed to be zero. Otherwise if the
  /// element of `rhs` is within the range `[32, 127]` then the output lane is
  /// either `0` or `self[rhs[i] % 16]` depending on the implementation.
  ///
  /// This is the equivalent to two parallel swizzle operations on the two
  /// halves of the vector, and the indexes each refer to their corresponding
  /// half.
  #[inline]
  pub fn swizzle_half_relaxed(self, rhs: i8x32) -> i8x32 {
    pick! {
      if #[cfg(target_feature="avx2")] {
        Self(shuffle_av_i8z_half_m256i(self.0, rhs.0))
      } else {
        Self(Inner(
          self.0.0.swizzle_relaxed(rhs.0.0),
          self.0.1.swizzle_relaxed(rhs.0.1),
        ))
      }
    }
  }

  /// Full 32-entry byte table lookup.
  ///
  /// * An index (interpreted as unsigned) in `[0, 31]` selects `self[index]`.
  /// * Any index `>= 32` (including negative `i8` values) yields `0`.
  ///
  /// Unlike [`swizzle_half`](Self::swizzle_half), indices address the entire
  /// 32-byte vector, not just their own 16-byte half.
  #[inline]
  pub fn swizzle(self, rhs: i8x32) -> i8x32 {
    pick! {
      if #[cfg(all(target_feature="avx512vbmi", target_feature="avx512vl"))] {
        #[cfg(target_arch = "x86")]
        use core::arch::x86::_mm256_permutexvar_epi8;
        #[cfg(target_arch = "x86_64")]
        use core::arch::x86_64::_mm256_permutexvar_epi8;
        // vpermb takes the index mod 32 and never zeroes, so zero the
        // out-of-range lanes ourselves: (rhs & 0xE0) == 0  <=>  rhs < 32.
        // TODO(safe_arch): Add `_mm256_permutexvar_epi8`.
        let permuted = m256i(unsafe { _mm256_permutexvar_epi8(rhs.0.0, self.0.0) });
        let hi_bits = bitand_m256i(rhs.0, set_splat_i8_m256i(0xE0_u8 as i8));
        let in_range = cmp_eq_mask_i8_m256i(hi_bits, zeroed_m256i());
        Self(bitand_m256i(permuted, in_range))
      } else if #[cfg(target_feature="avx2")] {
        // Broadcast each 16-byte table half into both 128-bit lanes, pshufb
        // each by the index, blend by index bit 4. Fold the >=32 zeroing into
        // pshufb with an unsigned saturating add of 0x60 (0x60 + 32 = 0x80).
        let idx = add_saturating_u8_m256i(rhs.0, set_splat_i8_m256i(0x60));
        let tbl_lo = shuffle_abi_i128z_all_m256i::<0x00>(self.0, self.0);
        let tbl_hi = shuffle_abi_i128z_all_m256i::<0x11>(self.0, self.0);
        let res_lo = shuffle_av_i8z_half_m256i(tbl_lo, idx);
        let res_hi = shuffle_av_i8z_half_m256i(tbl_hi, idx);
        // move index bit 4 into the sign bit (bit 7) for blendv.
        let sel = shl_imm_u16_m256i::<3>(rhs.0);
        Self(blend_varying_i8_m256i(res_lo, res_hi, sel))
      } else if #[cfg(all(target_feature="neon", target_arch="aarch64"))] {
        use core::arch::aarch64::{int8x16x2_t, vqtbl2q_s8, vreinterpretq_u8_s8};
        unsafe {
          let table = int8x16x2_t(self.0.0.0, self.0.1.0);
          Self(Inner(
            i8x16(vqtbl2q_s8(table, vreinterpretq_u8_s8(rhs.0.0.0))),
            i8x16(vqtbl2q_s8(table, vreinterpretq_u8_s8(rhs.0.1.0))),
          ))
        }
      } else {
        // Generic {a,b}: each output half pulls from either table half.
        // a.swizzle / b.swizzle are STRICT (zero index >= 16), and their
        // nonzero domains are disjoint, so a bitwise OR selects correctly and
        // out-of-range (>=32) falls out as 0 with no extra mask.
        let sixteen = i8x16::splat(16);
        Self(Inner(
          self.0.0.swizzle(rhs.0.0) | self.0.1.swizzle(rhs.0.0 - sixteen),
          self.0.0.swizzle(rhs.0.1) | self.0.1.swizzle(rhs.0.1 - sixteen),
        ))
      }
    }
  }

  /// Like [`swizzle`](Self::swizzle), but out-of-range indices (unsigned
  /// `>= 32`) yield an implementation-defined result (`0` or `self[index % 32]`).
  /// Prefer this when you know all indices are in range; it can be cheaper.
  #[inline]
  pub fn swizzle_relaxed(self, rhs: i8x32) -> i8x32 {
    pick! {
      if #[cfg(all(target_feature="avx512vbmi", target_feature="avx512vl"))] {
        #[cfg(target_arch = "x86")]
        use core::arch::x86::_mm256_permutexvar_epi8;
        #[cfg(target_arch = "x86_64")]
        use core::arch::x86_64::_mm256_permutexvar_epi8;
        // TODO(safe_arch): Add `_mm256_permutexvar_epi8`.
        Self(m256i(unsafe { _mm256_permutexvar_epi8(rhs.0.0, self.0.0) }))
      } else if #[cfg(target_feature="avx2")] {
        // Same broadcast+blend as strict, but skip the 0x60 zeroing fold.
        let tbl_lo = shuffle_abi_i128z_all_m256i::<0x00>(self.0, self.0);
        let tbl_hi = shuffle_abi_i128z_all_m256i::<0x11>(self.0, self.0);
        let res_lo = shuffle_av_i8z_half_m256i(tbl_lo, rhs.0);
        let res_hi = shuffle_av_i8z_half_m256i(tbl_hi, rhs.0);
        let sel = shl_imm_u16_m256i::<3>(rhs.0);
        Self(blend_varying_i8_m256i(res_lo, res_hi, sel))
      } else if #[cfg(all(target_feature="neon", target_arch="aarch64"))] {
        // vqtbl2 zeroes out-of-range anyway; identical to strict.
        use core::arch::aarch64::{int8x16x2_t, vqtbl2q_s8, vreinterpretq_u8_s8};
        unsafe {
          let table = int8x16x2_t(self.0.0.0, self.0.1.0);
          Self(Inner(
            i8x16(vqtbl2q_s8(table, vreinterpretq_u8_s8(rhs.0.0.0))),
            i8x16(vqtbl2q_s8(table, vreinterpretq_u8_s8(rhs.0.1.0))),
          ))
        }
      } else {
        // Strict fallback is a valid relaxed implementation (it zeroes OOR).
        let sixteen = i8x16::splat(16);
        Self(Inner(
          self.0.0.swizzle(rhs.0.0) | self.0.1.swizzle(rhs.0.0 - sixteen),
          self.0.0.swizzle(rhs.0.1) | self.0.1.swizzle(rhs.0.1 - sixteen),
        ))
      }
    }
  }
}
