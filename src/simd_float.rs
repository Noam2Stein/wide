macro_rules! impl_simd_float {
  (
    // SAFETY: The contents of this macro assume that:
    //
    // - `T` implements `Pod`
    // - `Pod` can be implemented for `Simd`
    // - `size_of::<Simd>()` is `size_of::<T>() * N`
    // - `align_of::<Simd>()` is `size_of::<Simd>()`
    unsafe {
      T = $T:ident,
      N = $N:literal,
      Simd = $Simd:ident,
      UnsignedT = $UnsignedT:ident,
      UnsignedSimd = $UnsignedSimd:ident,
    }
    old_powf_simd_fn_name = $old_powf_simd_fn_name:ident,

    $fn_neg:item
    $fn_not:item
    $fn_add:item
    $fn_sub:item
    $fn_mul:item
    $fn_div:item
    $fn_rem:item
    $fn_bitand:item
    $fn_bitor:item
    $fn_bitxor:item
    $fn_reduce_add:item
    $fn_reduce_mul:item
    $fn_is_nan:item
    $fn_is_inf:item
    $fn_is_finite:item
    $fn_is_sign_positive:item
    $fn_is_sign_negative:item
    $fn_recip:item
    $fn_recip_sqrt:item
    $fn_max:item
    $fn_fast_max:item
    $fn_min:item
    $fn_fast_min:item
    $fn_clamp:item
    $fn_fast_clamp:item
    $fn_abs:item
    $fn_floor:item
    $fn_ceil:item
    $fn_round:item
    $fn_round_int:item
    $fn_fast_round_int:item
    $fn_round_ties_even:item
    $fn_trunc:item
    $fn_trunc_int:item
    $fn_fast_trunc_int:item
    $fn_mul_add:item
    $fn_fast_mul_add:item
    $fn_mul_sub:item
    $fn_fast_mul_sub:item
    $fn_mul_neg_add:item
    $fn_fast_mul_neg_add:item
    $fn_mul_neg_sub:item
    $fn_fast_mul_neg_sub:item
    $fn_powf_simd:item
    $fn_sqrt:item
    $fn_exp:item
    $fn_exp2:item
    $fn_ln:item
    $fn_cbrt:item
    $fn_asin:item
    $fn_acos:item
    $fn_atan:item
    $fn_atan2:item
    $fn_sin_cos:item
    $fn_asin_acos:item
    $fn_exp_m1:item
    $fn_ln_1p:item
    $fn_sinh:item
    $fn_cosh:item
    $fn_tanh:item
  ) => {
    impl_unary_operator!($Simd, Neg, neg, $fn_neg);
    impl_unary_operator!($Simd, Not, not, $fn_not);

    impl_binary_operator!($T, $Simd, Add, add, AddAssign, add_assign, $fn_add);
    impl_binary_operator!($T, $Simd, Sub, sub, SubAssign, sub_assign, $fn_sub);
    impl_binary_operator!($T, $Simd, Mul, mul, MulAssign, mul_assign, $fn_mul);
    impl_binary_operator!($T, $Simd, Div, div, DivAssign, div_assign, $fn_div);
    impl_binary_operator!($T, $Simd, Rem, rem, RemAssign, rem_assign, $fn_rem);
    impl_binary_operator!(
      $T,
      $Simd,
      BitAnd,
      bitand,
      BitAndAssign,
      bitand_assign,
      $fn_bitand
    );
    impl_binary_operator!(
      $T,
      $Simd,
      BitOr,
      bitor,
      BitOrAssign,
      bitor_assign,
      $fn_bitor
    );
    impl_binary_operator!(
      $T,
      $Simd,
      BitXor,
      bitxor,
      BitXorAssign,
      bitxor_assign,
      $fn_bitxor
    );

    impl<Rhs> core::iter::Sum<Rhs> for $Simd
    where
      $Simd: AddAssign<Rhs>,
    {
      #[inline]
      fn sum<I: Iterator<Item = Rhs>>(iter: I) -> Self {
        let mut total = Self::zeroed();
        for val in iter {
          total += val;
        }
        total
      }
    }

    impl<Rhs> core::iter::Product<Rhs> for $Simd
    where
      $Simd: MulAssign<Rhs>,
    {
      #[inline]
      fn product<I: Iterator<Item = Rhs>>(iter: I) -> Self {
        let mut total = Self::from(1.0);
        for val in iter {
          total *= val;
        }
        total
      }
    }

    macro_rules! impl_formatting_trait {
      ($Trait:path) => {
        impl $Trait for $Simd {
          #[allow(clippy::missing_inline_in_public_items)]
          fn fmt(&self, f: &mut core::fmt::Formatter) -> core::fmt::Result {
            write!(f, "(")?;
            for (i, x) in self.to_array().iter().enumerate() {
              if i > 0 {
                write!(f, ", ")?;
              }
              <$UnsignedT as $Trait>::fmt(&x.to_bits(), f)?;
            }
            write!(f, ")")
          }
        }
      }
    }
    impl_formatting_trait!(core::fmt::Binary);
    impl_formatting_trait!(core::fmt::LowerHex);
    impl_formatting_trait!(core::fmt::Octal);
    impl_formatting_trait!(core::fmt::UpperHex);

    impl $Simd {
      pub const ONE: Self = Self::splat(1.0);
      pub const HALF: Self = Self::splat(0.5);
      pub const ZERO: Self = Self::splat(0.0);
      pub const EPSILON: Self = Self::splat($T::EPSILON);
      pub const MIN: Self = Self::splat($T::MIN);
      pub const MIN_POSITIVE: Self = Self::splat($T::MIN_POSITIVE);
      pub const MAX: Self = Self::splat($T::MAX);
      pub const NAN: Self = Self::splat($T::NAN);
      pub const INFINITY: Self = Self::splat($T::INFINITY);
      pub const NEG_INFINITY: Self = Self::splat($T::NEG_INFINITY);
      pub const E: Self = Self::splat(core::$T::consts::E);
      pub const FRAC_1_PI: Self = Self::splat(core::$T::consts::FRAC_1_PI);
      pub const FRAC_2_PI: Self = Self::splat(core::$T::consts::FRAC_2_PI);
      pub const FRAC_2_SQRT_PI: Self =
        Self::splat(core::$T::consts::FRAC_2_SQRT_PI);
      pub const FRAC_1_SQRT_2: Self =
        Self::splat(core::$T::consts::FRAC_1_SQRT_2);
      pub const FRAC_PI_2: Self = Self::splat(core::$T::consts::FRAC_PI_2);
      pub const FRAC_PI_3: Self = Self::splat(core::$T::consts::FRAC_PI_3);
      pub const FRAC_PI_4: Self = Self::splat(core::$T::consts::FRAC_PI_4);
      pub const FRAC_PI_6: Self = Self::splat(core::$T::consts::FRAC_PI_6);
      pub const FRAC_PI_8: Self = Self::splat(core::$T::consts::FRAC_PI_8);
      pub const LN_2: Self = Self::splat(core::$T::consts::LN_2);
      pub const LN_10: Self = Self::splat(core::$T::consts::LN_10);
      pub const LOG2_E: Self = Self::splat(core::$T::consts::LOG2_E);
      pub const LOG10_E: Self = Self::splat(core::$T::consts::LOG10_E);
      pub const LOG10_2: Self = Self::splat(core::$T::consts::LOG10_2);
      pub const LOG2_10: Self = Self::splat(core::$T::consts::LOG2_10);
      pub const PI: Self = Self::splat(core::$T::consts::PI);
      pub const SQRT_2: Self = Self::splat(core::$T::consts::SQRT_2);
      pub const TAU: Self = Self::splat(core::$T::consts::TAU);

      /// horizontal add of all the elements of the vector
      #[must_use]
      $fn_reduce_add

      /// horizontal multiplication of all the elements of the vector
      #[must_use]
      $fn_reduce_mul

      #[must_use]
      $fn_is_nan

      #[must_use]
      $fn_is_inf

      #[must_use]
      $fn_is_finite

      /// Returns true for each element if it has a positive sign, including `+0.0`,
      /// `NaN`s with positive sign bit and positive infinity.
      #[must_use]
      $fn_is_sign_positive

      /// Returns true for each element if it has a negative sign, including `-0.0`,
      /// `NaN`s with negative sign bit and negative infinity.
      #[must_use]
      $fn_is_sign_negative

      #[must_use]
      $fn_recip

      #[must_use]
      $fn_recip_sqrt

      #[inline]
      #[must_use]
      pub fn to_degrees(self) -> Self {
        const RAD_TO_DEG_RATIO: $Simd = $Simd::splat(180.0 / core::$T::consts::PI);
        self * RAD_TO_DEG_RATIO
      }

      #[inline]
      #[must_use]
      pub fn to_radians(self) -> Self {
        const DEG_TO_RAD_RATIO: $Simd = $Simd::splat(core::$T::consts::PI / 180.0);
        self * DEG_TO_RAD_RATIO
      }

      /// Calculates the lanewise maximum of both vectors. If either lane is
      /// NaN, the other lane gets chosen. Use `fast_max` for a faster
      /// implementation that doesn't handle NaNs.
      #[must_use]
      $fn_max

      /// Calculates the lanewise maximum of both vectors. This is a faster
      /// implementation than `max`, but it doesn't specify any behavior if NaNs
      /// are involved.
      #[must_use]
      $fn_fast_max

      /// Calculates the lanewise minimum of both vectors. If either lane is
      /// NaN, the other lane gets chosen. Use `fast_min` for a faster
      /// implementation that doesn't handle NaNs.
      #[must_use]
      $fn_min

      /// Calculates the lanewise minimum of both vectors. This is a faster
      /// implementation than `min`, but it doesn't specify any behavior if NaNs
      /// are involved.
      #[must_use]
      $fn_fast_min

      #[inline]
      #[must_use]
      pub fn midpoint(self, other: Self) -> Self {
        (self + other) * 0.5
      }

      /// Raw transmutation to unsigned integer vector.
      ///
      /// Note that this function preserves the *bitwise* value, and not the
      /// numeric value.
      #[inline]
      #[must_use]
      pub const fn to_bits(self) -> $UnsignedSimd {
        // SAFETY: Both types accept all bit-patterns and only contain
        // initialized memory.
        unsafe { core::mem::transmute::<$Simd, $UnsignedSimd>(self) }
      }

      /// Raw transmutation from unsigned integer vector.
      ///
      /// Note that this function preserves the *bitwise* value, and not the
      /// numeric value.
      #[inline]
      #[must_use]
      pub const fn from_bits(bits: $UnsignedSimd) -> Self {
        // SAFETY: Both types accept all bit-patterns and only contain
        // initialized memory.
        unsafe { core::mem::transmute::<$UnsignedSimd, $Simd>(bits) }
      }

      /// Restrict a value to a certain interval unless it is NaN.
      ///
      /// If `self`, `min` or `max` are NaN, the result is NaN.  If `min > max`,
      /// the result is `min` since `max(min)` dominates.
      #[must_use]
      $fn_clamp

      /// Restrict a value to a certain interval unless it is NaN.
      ///
      /// If `self` is NaN, the result is NaN.  If `min > max`, the result is
      /// `min` since `max(min)` dominates. If `min` or `max` are NaN, the
      /// result is unspecified.
      #[must_use]
      $fn_fast_clamp

      #[must_use]
      $fn_abs

      #[inline]
      #[must_use]
      pub fn signum(self) -> Self {
        let result = Self::ONE | self & -Self::ZERO;

        self.is_nan().select(self, result)
      }

      #[inline]
      #[must_use]
      pub fn copysign(self, sign: Self) -> Self {
        let magnitude_mask = Self::from($T::from_bits($UnsignedT::MAX >> 1));
        (self & magnitude_mask) | (sign & Self::from(-0.0))
      }

      #[inline]
      #[must_use]
      pub fn flip_signs(self, signs: Self) -> Self {
        self ^ (signs & Self::from(-0.0))
      }

      #[must_use]
      $fn_floor

      #[must_use]
      $fn_ceil

      /// Returns the nearest integers to `self`. If a value is half-way between
      /// two integers, round away from `0.0`.
      ///
      /// This function always returns the precise result.
      ///
      /// For most targets [`round`] is slower than [`round_ties_even`]. If you
      /// do not care about the difference, consider using that instead.
      ///
      /// [`round`]: Self::round
      /// [`round_ties_even`]: Self::round_ties_even
      #[must_use]
      $fn_round

      /// Rounds each lane into an integer. This saturates out of range values
      /// and turns NaNs into 0. Use `fast_round_int` for a faster
      /// implementation that doesn't handle out of range values or NaNs.
      #[must_use]
      $fn_round_int

      /// Rounds each lane into an integer. This is a faster implementation than
      /// `round_int`, but it doesn't handle out of range values or NaNs. For
      /// those values you get implementation defined behavior.
      #[must_use]
      $fn_fast_round_int

      /// Returns the nearest integers to `self`. Rounds half-way cases to the
      /// number with an even least significant digit.
      ///
      /// This function always returns the precise result.
      #[must_use]
      $fn_round_ties_even

      #[must_use]
      $fn_trunc

      /// Truncates each lane into an integer. This saturates out of range
      /// values and turns NaNs into 0. Use `fast_trunc_int` for a faster
      /// implementation that doesn't handle out of range values or NaNs.
      #[must_use]
      $fn_trunc_int

      /// Truncates each lane into an integer. This is a faster implementation
      /// than `trunc_int`, but it doesn't handle out of range values or NaNs.
      /// For those values you get implementation defined behavior.
      #[must_use]
      $fn_fast_trunc_int

      #[inline]
      #[must_use]
      pub fn fract(self) -> Self {
        self - self.trunc()
      }

      /// Fused multiply-add. Computes `(self * a) + b` with only one rounding
      /// error, yielding a more accurate result than an unfused multiply-add.
      ///
      /// This always returns the precise result, even if there is no hardware
      /// `fma` support. In past versions, this function used to fallback to an
      /// unfused multiply-add, instead of a precise but slow software
      /// implementation. The previous behavior has been moved to
      /// [`fast_mul_add`].
      ///
      /// # Unspecified precision
      ///
      /// The precision of this function is non-deterministic. This means it
      /// varies by platform, version, and can even differ within the same
      /// execution from one invocation to the next.
      ///
      /// [`fast_mul_add`]: Self::fast_mul_add
      #[must_use]
      $fn_mul_add

      /// Fused multiply-add. Computes `(self * a) + b` with only one rounding
      /// error if possible.
      ///
      /// If there is hardware FMA support, this computes the result with only
      /// one rounding error. If not, this falls back to separate multiply and
      /// add operations, resulting in two rounding errors.
      #[must_use]
      $fn_fast_mul_add

      /// Fused multiply-subtract. Computes `(self * a) - b` with only one
      /// rounding error, yielding a more accurate result than an unfused
      /// multiply-subtract.
      ///
      /// This always returns the precise result, even if there is no hardware
      /// `fma` support. In past versions, this function used to fallback to an
      /// unfused multiply-subtract, instead of a precise but slow software
      /// implementation. The previous behavior has been moved to
      /// [`fast_mul_sub`].
      ///
      /// # Unspecified precision
      ///
      /// The precision of this function is non-deterministic. This means it
      /// varies by platform, version, and can even differ within the same
      /// execution from one invocation to the next.
      ///
      /// [`fast_mul_sub`]: Self::fast_mul_sub
      #[must_use]
      $fn_mul_sub

      /// Fused multiply-subtract. Computes `(self * a) - b` with only one
      /// rounding error if possible.
      ///
      /// If there is hardware FMA support, this computes the result with only
      /// one rounding error. If not, this falls back to separate multiply and
      /// add operations, resulting in two rounding errors.
      #[must_use]
      $fn_fast_mul_sub

      /// Fused multiply-negate-add. Computes `-(self * a) + b` with only one
      /// rounding error, yielding a more accurate result than an unfused
      /// multiply-negate-add.
      ///
      /// This always returns the precise result, even if there is no hardware
      /// `fma` support. In past versions, this function used to fallback to an
      /// unfused multiply-negate-add, instead of a precise but slow software
      /// implementation. The previous behavior has been moved to
      /// [`fast_mul_neg_add`].
      ///
      /// # Unspecified precision
      ///
      /// The precision of this function is non-deterministic. This means it
      /// varies by platform, version, and can even differ within the same
      /// execution from one invocation to the next.
      ///
      /// [`fast_mul_neg_add`]: Self::fast_mul_neg_add
      #[must_use]
      $fn_mul_neg_add

      /// Fused multiply-negate-add. Computes `-(self * a) + b` with only one
      /// rounding error if possible.
      ///
      /// If there is hardware FMA support, this computes the result with only
      /// one rounding error. If not, this falls back to separate multiply and
      /// add operations, resulting in two rounding errors.
      #[must_use]
      $fn_fast_mul_neg_add

      /// Fused multiply-negate-subtract. Computes `-(self * a) - b` with only
      /// one rounding error, yielding a more accurate result than an unfused
      /// multiply-negate-subtract.
      ///
      /// This always returns the precise result, even if there is no hardware
      /// `fma` support. In past versions, this function used to fallback to an
      /// unfused multiply-negate-subtract, instead of a precise but slow
      /// software implementation. The previous behavior has been moved to
      /// [`fast_mul_neg_sub`].
      ///
      /// # Unspecified precision
      ///
      /// The precision of this function is non-deterministic. This means it
      /// varies by platform, version, and can even differ within the same
      /// execution from one invocation to the next.
      ///
      /// [`fast_mul_neg_sub`]: Self::fast_mul_neg_sub
      #[must_use]
      $fn_mul_neg_sub

      /// Fused multiply-negate-subtract. Computes `-(self * a) - b` with only
      /// one rounding error if possible.
      ///
      /// If there is hardware FMA support, this computes the result with only
      /// one rounding error. If not, this falls back to separate multiply and
      /// add operations, resulting in two rounding errors.
      #[must_use]
      $fn_fast_mul_neg_sub

      #[inline]
      #[must_use]
      pub fn div_euclid(self, rhs: Self) -> Self {
        let q = (self / rhs).trunc();
        (self % rhs)
          .simd_lt(Self::ZERO)
          .select(rhs.simd_gt(Self::ZERO).select(q - Self::ONE, q + Self::ONE), q)
      }

      #[inline]
      #[must_use]
      pub fn rem_euclid(self, rhs: Self) -> Self {
        let r = self % rhs;
        r.simd_lt(Self::ZERO).select(r + rhs.abs(), r)
      }

      /// Raises each element of the number `self` to the corresponding element
      /// of the floating point power `n`.
      ///
      /// This function cannot be named simply `powf`, because a now deprecated
      /// function already uses that name.
      ///
      /// # Unspecified precision
      ///
      /// The precision of this function is non-deterministic. This means it
      /// varies by platform, version, and can even differ within the same
      /// execution from one invocation to the next.
      #[must_use]
      $fn_powf_simd

      #[must_use]
      $fn_sqrt

      #[must_use]
      $fn_exp

      /// Returns `2^self`.
      #[must_use]
      $fn_exp2

      /// Natural log (ln(x))
      #[must_use]
      $fn_ln

      #[inline]
      #[must_use]
      pub fn log2(self) -> Self {
        Self::ln(self) * Self::LOG2_E
      }

      #[inline]
      #[must_use]
      pub fn log10(self) -> Self {
        Self::ln(self) * Self::LOG10_E
      }

      /// Calculates the cube root: `self^(1/3)`.
      #[must_use]
      $fn_cbrt

      #[inline]
      #[must_use]
      pub fn sin(self) -> Self {
        let (s, _) = self.sin_cos();
        s
      }

      #[inline]
      #[must_use]
      pub fn cos(self) -> Self {
        let (_, c) = self.sin_cos();
        c
      }

      #[inline]
      #[must_use]
      pub fn tan(self) -> Self {
        let (s, c) = self.sin_cos();
        s / c
      }

      #[must_use]
      $fn_asin

      #[must_use]
      $fn_acos

      #[must_use]
      $fn_atan

      #[must_use]
      $fn_atan2

      #[must_use]
      $fn_sin_cos

      #[must_use]
      $fn_asin_acos

      /// Calculate `e^self - 1` for each lane. Accurate even for very small
      /// values.
      #[must_use]
      $fn_exp_m1

      /// Calculate `ln(1 + self)` for each lane. Accurate even for very small
      /// values.
      #[must_use]
      $fn_ln_1p

      /// Calculates hyperbolic sine: `(e^self - e^(-self))/2`.
      #[must_use]
      $fn_sinh

      /// Calculates hyperbolic cosine: `(e^self + e^(-self))/2`.
      #[must_use]
      $fn_cosh

      /// Calculates hyperbolic tangent: `sinh(self)/cosh(self)`.
      #[must_use]
      $fn_tanh

      /// Raises each element of the number `self` to the corresponding element
      /// of the floating point power `n`.
      ///
      /// This function has been renamed to [`powf_simd`].
      ///
      /// # Unspecified precision
      ///
      /// The precision of this function is non-deterministic. This means it
      /// varies by platform, version, and can even differ within the same
      /// execution from one invocation to the next.
      ///
      /// [`powf_simd`]: Self::powf_simd
      #[deprecated(since = "1.6.0", note = "renamed to `powf_simd`")]
      #[inline]
      #[must_use]
      pub fn $old_powf_simd_fn_name(self, n: Self) -> Self {
        self.powf_simd(n)
      }

      /// Raises each element of the number `self` to the scalar floating point
      /// power `n`.
      ///
      /// This function has been deprecated because it raises all elements of
      /// `x` to the same power, even though that brings no performance benefit.
      #[doc = concat!("Use `x.powf_simd(", stringify!($Simd), "::splat(n))` instead.")]
      ///
      /// # Unspecified precision
      ///
      /// The precision of this function is non-deterministic. This means it
      /// varies by platform, version, and can even differ within the same
      /// execution from one invocation to the next.
      #[deprecated(since = "1.6.0", note = "use `x.powf_simd(splat(n))` instead")]
      #[inline]
      #[must_use]
      pub fn powf(self, n: $T) -> Self {
        self.powf_simd(Self::splat(n))
      }
    }
  };
}

/// Precise implementation of `mul_add` that is used when there is no hardware
/// support.
///
/// TODO(perf): For `f32` types, it might be faster to emulate `mul_add` using
/// `f64` operations. See this `libm` implementation:
///
/// <https://docs.rs/libm/0.2.16/src/libm/math/generic/fma_wide.rs.html>
macro_rules! software_mul_add {
  (
    $Float:ident,
    $Int:ident,
    $Uint:ident,
    $SimdFloat:ident,
    $SimdInt:ident,
    $SimdUint:ident,
    $self:ident,
    $a:ident,
    $b:ident
  ) => {
    // Based on `https://docs.rs/libm/0.2.16/src/libm/math/generic/fma.rs.html`.

    const BITS: $Uint = (size_of::<$Float>() * 8) as $Uint;
    const MANTISSA_DIGITS: $Uint = $Float::MANTISSA_DIGITS as $Uint;
    const SIG_BITS: $Uint = MANTISSA_DIGITS - 1;
    const EXP_BITS: $Uint = BITS - SIG_BITS - 1;

    const EXP_SAT: $Uint = (1 << EXP_BITS) - 1;
    const EXP_BIAS: $Uint = EXP_SAT >> 1;
    const EXP_UNBIAS: $Uint = EXP_BIAS + SIG_BITS + 1;

    const SIG_MASK: $Uint = (1 << SIG_BITS) - 1;
    const IMPLICIT_BIT: $Uint = 1 << SIG_BITS;

    // Scaling used temporarely to normalize subnormals
    const SUBNORMAL_SCALE: $Float = (BITS - 1) as $Float;
    const SUBNORMAL_SCALE_XOR_1: $Float =
      $Float::from_bits(SUBNORMAL_SCALE.to_bits() ^ (1 as $Float).to_bits());

    // Exponent adjustments
    const EXP_OFFSET: $Int = -(SUBNORMAL_SCALE as $Int) - EXP_UNBIAS as $Int;
    const SENTINEL_0_EXP: $Int = 1 << EXP_BITS;
    const SENTINEL_0_EXP_XOR_EXP_OFFSET: $Int = SENTINEL_0_EXP ^ EXP_OFFSET;
    /// Values greater than this had a saturated exponent (infinity or NaN), OR were zero and we
    /// adjusted the exponent such that it exceeds this threashold.
    const ZERO_INF_NAN: $Uint = EXP_SAT - EXP_UNBIAS;

    // Splatted SIMD constants
    const BITS_SIMD: $SimdUint = $SimdUint::splat(BITS);
    const EXP_SAT_SIMD: $SimdUint = $SimdUint::splat(EXP_SAT);
    const SIG_MASK_SIMD: $SimdUint = $SimdUint::splat(SIG_MASK);
    const IMPLICIT_BIT_SIMD: $SimdUint = $SimdUint::splat(IMPLICIT_BIT);
    const SUBNORMAL_SCALE_XOR_1_SIMD: $SimdFloat =
      $SimdFloat::splat(SUBNORMAL_SCALE_XOR_1);
    const EXP_OFFSET_SIMD: $SimdInt = $SimdInt::splat(EXP_OFFSET);
    const SENTINEL_0_EXP_XOR_EXP_OFFSET_SIMD: $SimdInt =
      $SimdInt::splat(SENTINEL_0_EXP_XOR_EXP_OFFSET);
    const ZERO_INF_NAN_SIMD: $SimdUint = $SimdUint::splat(ZERO_INF_NAN);

    /// Returns the exponent, not adjusting for bias, not accounting for
    /// subnormals or zero.
    #[inline]
    fn ex(x: $SimdFloat) -> $SimdUint {
      (x.to_bits() >> SIG_BITS) & EXP_SAT_SIMD
    }

    /// Converts to a float representation that has handled subnormals.
    ///
    /// Returns a tuple with:
    ///
    /// - The normalized significand with one guard bit, unsigned.
    ///
    /// - The exponent of the mantissa such that `m * 2^e = x`. Accounts for the
    ///   shift in the mantissa and the guard bit; that is, 1.0 will normalize
    ///   as `m = 1 << 53` and `e = -53`.
    #[inline]
    fn norm(x: $SimdFloat) -> ($SimdUint, $SimdInt) {
      let exp_bits = ex(x);

      // Normalize subnormals by multiplication
      let is_subnormal =
        $SimdFloat::from_bits(exp_bits.simd_eq($SimdUint::ZERO));
      // Compute select for constants
      let scale = $SimdFloat::ONE ^ (is_subnormal & SUBNORMAL_SCALE_XOR_1_SIMD);
      let x = x * scale;
      // Need to recompute exponent
      let exp_bits = ex(x);

      let sig = ((x.to_bits() & SIG_MASK_SIMD) | IMPLICIT_BIT_SIMD) << 1;

      // If the exponent is still zero, the input was zero. Artifically set this
      // value such that the final exponent will exceed `ZERO_INF_NAN`.
      let is_zero = exp_bits.simd_eq($SimdUint::ZERO).cast_signed();
      // Compute select for constants
      let exp_offset =
        EXP_OFFSET_SIMD ^ (is_zero & SENTINEL_0_EXP_XOR_EXP_OFFSET_SIMD);
      let exp = exp_bits.cast_signed() + exp_offset;

      (sig, exp)
    }

    /// Returns true if `exp` is neither zero, NaN, or infinite.
    #[inline]
    fn is_not_zero_nan_inf(exp: i32x4) -> f32x4 {
      $SimdFloat::from_bits(
        exp.simd_lt(ZERO_INF_NAN_SIMD.cast_signed()).cast_unsigned(),
      )
    }

    #[inline]
    fn is_zero(exp: i32x4) -> u32x4 {
      // The only exponent that strictly exceeds this value is our sentinel
      // value for zero.
      exp.simd_gt(ZERO_INF_NAN_SIMD.cast_signed()).cast_unsigned()
    }

    // Normalize such that the top of the mantissa is zero and we have a guard
    // bit.
    let (self_sig, self_exp) = norm($self);
    let (a_sig, a_exp) = norm($a);
    let (b_sig, b_exp) = norm($b);

    // Compute multiplication
    let (mul_sig_low, mul_sig_high) = self_sig.mul_keep_low_high(a_sig);
    let mul_exp = self_exp + a_exp;

    // Before addition can be done, the exponent of the multiplication and `b`
    // need to be adjusted to be the same
    let exp_diff = b_exp - mul_exp;

    let exp_diff_minus_bits = exp_diff - BITS_SIMD.cast_signed();
    let exp_diff_plus_bits = exp_diff + BITS_SIMD.cast_signed();
    let bits_minus_exp_diff = -exp_diff_minus_bits;
    let twobits_minus_exp_diff = BITS_SIMD.cast_signed() - exp_diff_minus_bits;
    let exp_diff_is_negative = exp_diff.is_negative().cast_unsigned();
    let exp_diff_is_positive = exp_diff.is_positive().cast_unsigned();
    let exp_diff_lt_bits = exp_diff_minus_bits.is_negative().cast_unsigned();
    let exp_diff_eq_bits = exp_diff_minus_bits.simd_eq($SimdInt::ZERO).cast_unsigned();
    let exp_diff_gt_bits = exp_diff_minus_bits.is_positive().cast_unsigned();
    let exp_diff_lt_2bits = exp_diff_minus_bits.simd_lt(BITS_SIMD.cast_signed()).cast_unsigned();
    let exp_diff_gt_neg_bits = exp_diff.simd_gt(-BITS_SIMD.cast_signed()).cast_unsigned();

    let exp = exp_diff_lt_bits.cast_signed().select(mul_exp, b_exp - BITS_SIMD.cast_signed());
    let b_sig_low = exp_diff_is_negative.select(
      b_sig.unbounded_shr(-exp_diff.cast_unsigned())
        | -((b_sig << exp_diff_plus_bits).simd_ne($SimdUint::ZERO) | exp_diff_gt_neg_bits),
      b_sig.unbounded_shl(exp_diff.cast_unsigned()),
    );
    let b_sig_high = b_sig.unbounded_shr(bits_minus_exp_diff.max($SimdInt::ZERO).cast_unsigned());
    let mul_sig_low = exp_diff_gt_bits.select(
      exp_diff_lt_2bits.select(
        (mul_sig_high << twobits_minus_exp_diff) | (mul_sig_low >> exp_diff_minus_bits),
        $SimdUint::ONE,
      ),
      mul_sig_low,
    );
    let mul_sig_low = exp_diff_is_positive.select(
      exp_diff_lt_bits.select(
        mul_sig_low,
        exp_diff_eq_bits.select(
          mul_sig_low,
          exp_diff_lt_2bits.select(
            mul_sig_low | (mul_sig_low << twobits_minus_exp_diff).simd_ne($SimdUint::ZERO) & $SimdUint::ONE,
            mul_sig_low,
          ),
        ),
      ),
      mul_sig_low,
    );
    let mul_sig_high = mul_sig_high.unbounded_shr(exp_diff_minus_bits.max($SimdInt::ZERO).cast_unsigned());

    let mul_neg = $self.is_sign_negative() ^ $a.is_sign_negative();
    let samesign = mul_neg ^ $b.is_sign_positive();

    let result = todo!();

    // If these are false, our algorithm breaks, but unfused mul add actually
    // works.
    let use_fused = is_not_zero_nan_inf(self_exp)
      & is_not_zero_nan_inf(a_exp)
      & is_not_zero_nan_inf(b_exp);

    use_fused.select(result, $self * $a + $b)
  };
}
