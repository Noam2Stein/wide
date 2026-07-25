use core::ops::{Add, BitAnd, BitOr, BitXor, Div, Mul, Neg, Not, Rem, Sub};

use bytemuck::{Pod, Zeroable};

use crate::AlignTo;
#[expect(deprecated)]
use crate::{CmpEq, CmpGe, CmpGt, CmpLe, CmpLt, CmpNe};

/// A generic SIMD vector with `N` elements of type `T`.
///
/// Note that only specific combinations of `T` and `N` are supported. Supported
/// [`Simd<T, N>`] types implement the [`SupportedSimd`] trait.
pub struct Simd<T, const N: usize>(<Self as SimdBackend>::Inner)
where
  T: SimdElement,
  Self: SupportedSimd;

/// A marker trait for types usable as SIMD vector elements.
#[expect(private_bounds)]
pub trait SimdElement: Sealed + Pod {}

/// A marker trait for supported [`Simd<T, N>`] types.
///
/// Ideally, for any supported `T` type, all [`Simd<T, N>`] types would be
/// supported. Unfortunately, that is impossible on stable Rust.
#[expect(private_bounds)]
pub trait SupportedSimd: SimdBackend {
  /// The type returned by [`Simd::to_bitmask`].
  ///
  /// This is currently [`u32`] for all SIMD types.
  type Bitmask;
}

/// Controls the internal implementation of [`Simd<T, N>`].
///
/// # Safety
///
/// - `Self` must be a `Simd<T, N>` type
/// - `Inner` must satisfy all requirements of `bytemuck::Pod`
/// - `Inner` must have a size and alignment equal to `size_of::<T>() * N`
pub(crate) unsafe trait SimdBackend {
  type Inner: Copy;

  fn neg(self) -> Self;

  fn not(self) -> Self;

  fn add(self, rhs: Self) -> Self;

  fn sub(self, rhs: Self) -> Self;

  fn mul(self, rhs: Self) -> Self;

  fn div(self, rhs: Self) -> Self;

  fn rem(self, rhs: Self) -> Self;

  fn bitand(self, rhs: Self) -> Self;

  fn bitor(self, rhs: Self) -> Self;

  fn bitxor(self, rhs: Self) -> Self;

  fn simd_eq(self, other: Self) -> Self;

  fn simd_ne(self, other: Self) -> Self;

  fn simd_lt(self, other: Self) -> Self;

  fn simd_gt(self, other: Self) -> Self;

  fn simd_le(self, other: Self) -> Self;

  fn simd_ge(self, other: Self) -> Self;

  fn reduce_add(self) -> Self::T
  where
    Self: SimdAssociatedTypes;

  fn reduce_mul(self) -> Self::T
  where
    Self: SimdAssociatedTypes;

  fn bitselect(self, if_one: Self, if_zero: Self) -> Self;

  fn select(self, if_true: Self, if_false: Self) -> Self;

  fn to_bitmask(self) -> Self::Bitmask
  where
    Self: SupportedSimd;

  fn any(self) -> bool;

  fn all(self) -> bool;

  fn transpose(matrix: Self::SelfxN) -> Self::SelfxN
  where
    Self: SimdAssociatedTypes;
}

/// A helper trait for [`SimdBackend`].
///
/// Since the [`SimdBackend`] trait is not generic over `T` and `N`, it needs
/// this trait in order to have access to types that depend on those parameters.
pub(crate) trait SimdAssociatedTypes {
  type T;

  type SelfxN;
}

/// An internal marker trait for element types of SIMD vectors that should
/// implement [`Neg`].
///
/// Currently, this is implemented for all SIMD element types, but this may not
/// be the case in the future.
pub(crate) trait NegOrUint {}

/// Seals the [`SimdElement`] trait.
trait Sealed {}

macro_rules! impl_formatting_trait {
  ($Trait:path) => {
    impl<T, const N: usize> $Trait for Simd<T, N>
    where
      T: SimdElement + $Trait,
      Self: SupportedSimd,
    {
      #[allow(clippy::missing_inline_in_public_items)]
      fn fmt(&self, f: &mut core::fmt::Formatter) -> core::fmt::Result {
        write!(f, "(")?;
        for (i, x) in self.to_array().iter().enumerate() {
          if i > 0 {
            write!(f, ", ")?;
          }
          <T as $Trait>::fmt(x, f)?;
        }
        write!(f, ")")
      }
    }
  };
}
impl_formatting_trait!(core::fmt::Debug);
impl_formatting_trait!(core::fmt::Display);
impl_formatting_trait!(core::fmt::LowerExp);
impl_formatting_trait!(core::fmt::UpperExp);

impl<T, const N: usize> Clone for Simd<T, N>
where
  T: SimdElement,
  Self: SupportedSimd,
{
  #[inline]
  fn clone(&self) -> Self {
    *self
  }
}

impl<T, const N: usize> Copy for Simd<T, N>
where
  T: SimdElement,
  Self: SupportedSimd,
{
}

impl<T, const N: usize> PartialEq for Simd<T, N>
where
  T: SimdElement + PartialEq,
  Self: SupportedSimd,
{
  #[inline]
  fn eq(&self, other: &Self) -> bool {
    self.simd_eq(*other).all()
  }
}

impl<T, const N: usize> Eq for Simd<T, N>
where
  T: SimdElement + Eq,
  Self: SupportedSimd,
{
}

impl<T, const N: usize> Default for Simd<T, N>
where
  T: SimdElement + Default,
  Self: SupportedSimd,
{
  #[inline]
  fn default() -> Self {
    // This works because coincidentally, all supported element types have a
    // default value of "all bits zero".
    Self::zeroed()
  }
}

impl<T, const N: usize> From<[T; N]> for Simd<T, N>
where
  T: SimdElement,
  Self: SupportedSimd,
{
  /// Converts an array to a SIMD vector.
  #[inline]
  fn from(value: [T; N]) -> Self {
    Self::new(value)
  }
}

impl<T, const N: usize> From<Simd<T, N>> for [T; N]
where
  T: SimdElement,
  Simd<T, N>: SupportedSimd,
{
  /// Converts a SIMD vector to an array.
  #[inline]
  fn from(value: Simd<T, N>) -> Self {
    value.to_array()
  }
}

impl<T, const N: usize> From<T> for Simd<T, N>
where
  T: SimdElement,
  Self: SupportedSimd,
{
  /// Converts a single value to a SIMD vector by setting all elements to that
  /// value.
  #[inline]
  fn from(value: T) -> Self {
    Self::splat(value)
  }
}

impl<T, const N: usize> From<&[T]> for Simd<T, N>
where
  T: SimdElement,
  Self: SupportedSimd,
{
  /// Converts a slice to a SIMD vector, filling in zeros if there are not
  /// enough elements, and panicking if there are too many elements.
  ///
  /// Note that in the future, handling of too many elements may change.
  #[inline]
  fn from(value: &[T]) -> Self {
    assert!(
      value.len() <= N,
      "slice has more elements than SIMD vector can store"
    );

    let mut result = Simd::zeroed();

    // SAFETY: `value` is valid for its own length, and `result` is valid
    // because its length is checked to be less than or equal to
    // `value.len()`. Both pointers are properly aligned because they
    // originate from a slice of `T`. The regions of memory do not overlap
    // because they originate from a shared reference and a mutable
    // reference.
    unsafe {
      core::ptr::copy_nonoverlapping::<T>(
        value.as_ptr(),
        result.as_mut_array().as_mut_ptr(),
        value.len(),
      );
    }

    result
  }
}

macro_rules! impl_unary_operator {
  (
    $Op:ident,
    $op:ident,
    $(Bound = $Bound:ident,)?
    $(#[$doc:meta])*
  ) => {
    impl<T, const N: usize> $Op for Simd<T, N>
    where
      T: SimdElement $(+ $Bound)?,
      Self: SupportedSimd,
    {
      type Output = Self;

      $(#[$doc])*
      #[inline]
      fn $op(self) -> Self::Output {
        SimdBackend::$op(self)
      }
    }

    impl<T, const N: usize> $Op for &Simd<T, N>
    where
      T: SimdElement $(+ $Bound)?,
      Simd<T, N>: SupportedSimd,
    {
      type Output = Simd<T, N>;

      $(#[$doc])*
      #[inline]
      fn $op(self) -> Self::Output {
        SimdBackend::$op(*self)
      }
    }
  };
}
impl_unary_operator!(
  Neg,
  neg,
  Bound = NegOrUint,
  /// Negates each element of `self`.
  ///
  /// Note that this is also implemented for unsigned integers, which do not
  /// implement the `-` operator themselves. Here it simply uses wrapping
  /// arithmetic.
);
impl_unary_operator!(
  Not,
  not,
  /// Flips all bits of all elements.
  ///
  /// Note that this is also implemented for floats, which do not implement the
  /// `!` operator themselves.
);

macro_rules! impl_binary_operator {
  (
    $Op:ident,
    $op:ident,
    $(Bound = $Bound:ident,)?
    $(#[$simd_simd_doc:meta])*,
    $(#[$simd_scalar_doc:meta])*,
    $(#[$scalar_simd_doc:meta])*,
    $(#[$extra_doc:meta])*
  ) => {
    impl<T, const N: usize> $Op for Simd<T, N>
    where
      T: SimdElement $(+ $Bound<Output = T>)?,
      Self: SupportedSimd,
    {
      type Output = Self;

      $(#[$simd_simd_doc])*
      $(#[$extra_doc])*
      #[inline]
      fn $op(self, rhs: Self) -> Self::Output {
        SimdBackend::$op(self, rhs)
      }
    }

    impl<T, const N: usize> $Op<T> for Simd<T, N>
    where
      T: SimdElement $(+ $Bound<Output = T>)?,
      Self: SupportedSimd,
    {
      type Output = Self;

      $(#[$simd_scalar_doc])*
      $(#[$extra_doc])*
      #[inline]
      fn $op(self, rhs: T) -> Self::Output {
        SimdBackend::$op(self, Self::splat(rhs))
      }
    }

    impl<T, const N: usize, Rhs> $Op<&Rhs> for Simd<T, N>
    where
      T: SimdElement,
      Self: SupportedSimd + $Op<Rhs>,
      Rhs: Copy,
    {
      type Output = <Self as $Op<Rhs>>::Output;

      $(#[$simd_simd_doc])*
      $(#[$extra_doc])*
      #[inline]
      fn $op(self, rhs: &Rhs) -> Self::Output {
        $Op::$op(self, *rhs)
      }
    }

    impl<T, const N: usize, Rhs> $Op<Rhs> for &Simd<T, N>
    where
      T: SimdElement,
      Simd<T, N>: SupportedSimd + $Op<Rhs>,
    {
      type Output = <Simd<T, N> as $Op<Rhs>>::Output;

      $(#[$simd_simd_doc])*
      $(#[$extra_doc])*
      #[inline]
      fn $op(self, rhs: Rhs) -> Self::Output {
        $Op::$op(*self, rhs)
      }
    }

    // The type system does not let us implement this for a generic `T` type, so
    // we must implement separately for each supported `T` type.
    macro_rules! impl_scalar_simd {
      ($T:ident) => {
        impl<const N: usize> $Op<Simd<$T, N>> for $T
        where
          $T: SimdElement $(+ $Bound<Output = $T>)?,
          Simd<$T, N>: SupportedSimd,
        {
          type Output = Simd<$T, N>;

          $(#[$scalar_simd_doc])*
          $(#[$extra_doc])*
          #[inline]
          fn $op(self, rhs: Simd<$T, N>) -> Self::Output {
            SimdBackend::$op(Simd::splat(self), rhs)
          }
        }

        impl<const N: usize> $Op<&Simd<$T, N>> for $T
        where
          $T: SimdElement $(+ $Bound<Output = $T>)?,
          Simd<$T, N>: SupportedSimd,
        {
          type Output = Simd<$T, N>;

          $(#[$scalar_simd_doc])*
          $(#[$extra_doc])*
          #[inline]
          fn $op(self, rhs: &Simd<$T, N>) -> Self::Output {
            SimdBackend::$op(Simd::splat(self), *rhs)
          }
        }

        impl<const N: usize> $Op<Simd<$T, N>> for &$T
        where
          $T: SimdElement $(+ $Bound<Output = $T>)?,
          Simd<$T, N>: SupportedSimd,
        {
          type Output = Simd<$T, N>;

          $(#[$scalar_simd_doc])*
          $(#[$extra_doc])*
          #[inline]
          fn $op(self, rhs: Simd<$T, N>) -> Self::Output {
            SimdBackend::$op(Simd::splat(*self), rhs)
          }
        }

        impl<const N: usize> $Op<&Simd<$T, N>> for &$T
        where
          $T: SimdElement $(+ $Bound<Output = $T>)?,
          Simd<$T, N>: SupportedSimd,
        {
          type Output = Simd<$T, N>;

          $(#[$scalar_simd_doc])*
          $(#[$extra_doc])*
          #[inline]
          fn $op(self, rhs: &Simd<$T, N>) -> Self::Output {
            SimdBackend::$op(Simd::splat(*self), *rhs)
          }
        }
      }
    }
    impl_scalar_simd!(f32);
    impl_scalar_simd!(f64);
    impl_scalar_simd!(i8);
    impl_scalar_simd!(i16);
    impl_scalar_simd!(i32);
    impl_scalar_simd!(i64);
    impl_scalar_simd!(u8);
    impl_scalar_simd!(u16);
    impl_scalar_simd!(u32);
    impl_scalar_simd!(u64);
  };
}
impl_binary_operator!(
  Add,
  add,
  Bound = Add,
  /// Computes addition for each element of `self` and the corresponding element
  /// of `rhs`.
  ,
  /// Computes addition for each element of `self` and the uniform scalar `rhs`.
  ,
  /// Computes addition for the uniform scalar `self` and each element of `rhs`.
  ,
  ///
  /// For floats, this always returns the precise result. For integers, this
  /// uses wrapping arithmetic.
);
impl_binary_operator!(
  Sub,
  sub,
  Bound = Sub,
  /// Computes subtraction for each element of `self` and the corresponding
  /// element of `rhs`.
  ,
  /// Computes subtraction for each element of `self` and the uniform scalar
  /// `rhs`.
  ,
  /// Computes subtraction for the uniform scalar `self` and each element of
  /// `rhs`.
  ,
  ///
  /// For floats, this always returns the precise result. For integers, this
  /// uses wrapping arithmetic.
);
impl_binary_operator!(
  Mul,
  mul,
  Bound = Mul,
  /// Computes multiplication for each element of `self` and the corresponding
  /// element of `rhs`.
  ,
  /// Computes multiplication for each element of `self` and the uniform scalar
  /// `rhs`.
  ,
  /// Computes multiplication for the uniform scalar `self` and each element of
  /// `rhs`.
  ,
  ///
  /// For floats, this always returns the precise result. For integers, this
  /// uses wrapping arithmetic.
);
impl_binary_operator!(
  Div,
  div,
  Bound = Div,
  /// Computes division for each element of `self` and the corresponding element
  /// of `rhs`.
  ,
  /// Computes division for each element of `self` and the uniform scalar `rhs`.
  ,
  /// Computes division for the uniform scalar `self` and each element of `rhs`.
  ,
  ///
  /// For floats, this always returns the precise result. For integers, this
  /// uses wrapping arithmetic and panics if any element of `rhs` is zero.
);
impl_binary_operator!(
  Rem,
  rem,
  Bound = Rem,
  /// Computes remainder for each element of `self` and the corresponding
  /// element of `rhs`.
  ,
  /// Computes remainder for each element of `self` and the uniform scalar
  /// `rhs`.
  ,
  /// Computes remainder for the uniform scalar `self` and each element of
  /// `rhs`.
  ,
  ///
  /// For floats, this always returns the precise result. For integers, this
  /// panics if any element of `rhs` is zero.
);
impl_binary_operator!(
  BitAnd,
  bitand,
  /// Computes bitwise AND for each element of `self` and the corresponding
  /// element of `rhs`.
  ,
  /// Computes bitwise AND for each element of `self` and the uniform scalar
  /// `rhs`.
  ,
  /// Computes bitwise AND for the uniform scalar `self` and each element of
  /// `rhs`.
  ,
  ///
  /// Note that this is also implemented for floats, which do not implement the
  /// `&` operator themselves.
);
impl_binary_operator!(
  BitOr,
  bitor,
  /// Computes bitwise OR for each element of `self` and the corresponding
  /// element of `rhs`.
  ,
  /// Computes bitwise OR for each element of `self` and the uniform scalar
  /// `rhs`.
  ,
  /// Computes bitwise OR for the uniform scalar `self` and each element of
  /// `rhs`.
  ,
  ///
  /// Note that this is also implemented for floats, which do not implement the
  /// `|` operator themselves.
);
impl_binary_operator!(
  BitXor,
  bitxor,
  /// Computes bitwise XOR for each element of `self` and the corresponding
  /// element of `rhs`.
  ,
  /// Computes bitwise XOR for each element of `self` and the uniform scalar
  /// `rhs`.
  ,
  /// Computes bitwise XOR for the uniform scalar `self` and each element of
  /// `rhs`.
  ,
  ///
  /// Note that this is also implemented for floats, which do not implement the
  /// `^` operator themselves.
);

// SAFETY: `Self::Inner` is required to satisfy all requirements of `Pod`.
unsafe impl<T, const N: usize> Zeroable for Simd<T, N>
where
  T: SimdElement,
  Self: SupportedSimd,
{
}

// SAFETY: `Self::Inner` is required to satisfy all requirements of `Pod`.
unsafe impl<T, const N: usize> Pod for Simd<T, N>
where
  T: SimdElement,
  Self: SupportedSimd,
{
}

macro_rules! impl_comparison_trait {
  ($Trait:ident, $Bound:ident, $fn:ident) => {
    #[expect(deprecated)]
    impl<T, const N: usize> $Trait for Simd<T, N>
    where
      T: SimdElement + $Bound,
      Self: SupportedSimd,
    {
      type Output = Self;

      #[inline]
      fn $fn(self, other: Self) -> Self::Output {
        SimdBackend::$fn(self, other)
      }
    }

    #[expect(deprecated)]
    impl<T, const N: usize> $Trait<T> for Simd<T, N>
    where
      T: SimdElement + $Bound,
      Self: SupportedSimd,
    {
      type Output = Self;

      #[inline]
      fn $fn(self, other: T) -> Self::Output {
        SimdBackend::$fn(self, Self::splat(other))
      }
    }
  };
}
impl_comparison_trait!(CmpEq, PartialEq, simd_eq);
impl_comparison_trait!(CmpNe, PartialEq, simd_ne);
impl_comparison_trait!(CmpLt, PartialOrd, simd_lt);
impl_comparison_trait!(CmpGt, PartialOrd, simd_gt);
impl_comparison_trait!(CmpLe, PartialOrd, simd_le);
impl_comparison_trait!(CmpGe, PartialOrd, simd_ge);

impl<T, const N: usize> AlignTo for Simd<T, N>
where
  T: SimdElement + PartialEq + Default,
  Self: SupportedSimd,
{
  type Elem = T;
}

impl<T, const N: usize> Simd<T, N>
where
  T: SimdElement,
  Self: SupportedSimd,
{
  /// Converts an array to a SIMD vector.
  #[inline]
  #[must_use]
  pub const fn new(array: [T; N]) -> Self {
    // SAFETY: Both types implement `Pod`.
    unsafe { core::mem::transmute_copy::<[T; N], Simd<T, N>>(&array) }
  }

  /// Constructs a new SIMD vector with all elements set to the given value.
  #[inline]
  #[must_use]
  pub const fn splat(value: T) -> Self {
    // SAFETY: Both types implement `Pod`.
    unsafe { core::mem::transmute_copy::<[T; N], Simd<T, N>>(&[value; N]) }
  }

  /// Converts a SIMD vector to an array.
  #[inline]
  #[must_use]
  pub const fn to_array(self) -> [T; N] {
    // SAFETY: Both types implement `Pod`.
    unsafe { core::mem::transmute_copy::<Simd<T, N>, [T; N]>(&self) }
  }

  /// Returns an array reference containing the entire SIMD vector.
  #[inline]
  #[must_use]
  pub const fn as_array(&self) -> &[T; N] {
    // SAFETY: The input type has greater alignment than the output type,
    // and both pointed-at types have the same size, accept all bit-patterns
    // and only contain initialized memory.
    unsafe { core::mem::transmute::<&Simd<T, N>, &[T; N]>(self) }
  }

  /// Returns a mutable array reference containing the entire SIMD vector.
  #[inline]
  #[must_use]
  pub const fn as_mut_array(&mut self) -> &mut [T; N] {
    // SAFETY: The input type has greater alignment than the output type,
    // and both pointed-at types have the same size, accept all bit-patterns
    // and only contain initialized memory.
    unsafe { core::mem::transmute::<&mut Simd<T, N>, &mut [T; N]>(self) }
  }

  /// Returns a [mask] that checks if each element of `self` is equal to the
  /// corresponding element of `other`.
  ///
  /// [mask]: crate#masks
  #[inline]
  #[must_use]
  #[expect(deprecated)]
  pub fn simd_eq<Rhs>(self, other: Rhs) -> <Self as CmpEq<Rhs>>::Output
  where
    Self: CmpEq<Rhs>,
  {
    CmpEq::simd_eq(self, other)
  }

  /// Returns a [mask] that checks if each element of `self` is not equal to the
  /// corresponding element of `other`.
  ///
  /// [mask]: crate#masks
  #[inline]
  #[must_use]
  #[expect(deprecated)]
  pub fn simd_ne<Rhs>(self, other: Rhs) -> <Self as CmpNe<Rhs>>::Output
  where
    Self: CmpNe<Rhs>,
  {
    CmpNe::simd_ne(self, other)
  }

  /// Returns a [mask] that checks if each element of `self` is less than the
  /// corresponding element of `other`.
  ///
  /// [mask]: crate#masks
  #[inline]
  #[must_use]
  #[expect(deprecated)]
  pub fn simd_lt<Rhs>(self, other: Rhs) -> <Self as CmpLt<Rhs>>::Output
  where
    Self: CmpLt<Rhs>,
  {
    CmpLt::simd_lt(self, other)
  }

  /// Returns a [mask] that checks if each element of `self` is greater than the
  /// corresponding element of `other`.
  ///
  /// [mask]: crate#masks
  #[inline]
  #[must_use]
  #[expect(deprecated)]
  pub fn simd_gt<Rhs>(self, other: Rhs) -> <Self as CmpGt<Rhs>>::Output
  where
    Self: CmpGt<Rhs>,
  {
    CmpGt::simd_gt(self, other)
  }

  /// Returns a [mask] that checks if each element of `self` is less than or
  /// equal to the corresponding element of `other`.
  ///
  /// [mask]: crate#masks
  #[inline]
  #[must_use]
  #[expect(deprecated)]
  pub fn simd_le<Rhs>(self, other: Rhs) -> <Self as CmpLe<Rhs>>::Output
  where
    Self: CmpLe<Rhs>,
  {
    CmpLe::simd_le(self, other)
  }

  /// Returns a [mask] that checks if each element of `self` is greater than or
  /// equal to the corresponding element of `other`.
  ///
  /// [mask]: crate#masks
  #[inline]
  #[must_use]
  #[expect(deprecated)]
  pub fn simd_ge<Rhs>(self, other: Rhs) -> <Self as CmpGe<Rhs>>::Output
  where
    Self: CmpGe<Rhs>,
  {
    CmpGe::simd_ge(self, other)
  }

  /// Reducing addition. Returns the sum of the vector's elements.
  ///
  /// Equivalent to `self[0] + self[1] + ...`.
  ///
  /// For integers, this uses wrapping arithmetic.
  ///
  /// # Unspecified precision (for floats)
  ///
  /// For floats, the order of addition is non-deterministic. This means it
  /// varies by platform, version, and can even differ within the same execution
  /// from one invocation to the next.
  #[inline]
  #[must_use]
  pub fn reduce_add(self) -> T
  where
    T: Add<Output = T>,
  {
    SimdBackend::reduce_add(self)
  }

  /// Reducing multiplication. Returns the product of the vector's elements.
  ///
  /// Equivalent to `self[0] * self[1] * ...`.
  ///
  /// For integers, this uses wrapping arithmetic.
  ///
  /// # Unspecified precision (for floats)
  ///
  /// For floats, the order of multiplication is non-deterministic. This means
  /// it varies by platform, version, and can even differ within the same
  /// execution from one invocation to the next.
  #[inline]
  #[must_use]
  pub fn reduce_mul(self) -> T
  where
    T: Mul<Output = T>,
  {
    SimdBackend::reduce_mul(self)
  }

  /// Bitwise selection.
  ///
  /// For each bit of `self`:
  ///
  /// - If the bit is one, return the corresponding bit of `if_one`
  /// - If the bit is zero, return the corresponding bit of `if_zero`
  ///
  /// If you know `self` is a [mask], meaning each element is either all zeros
  /// or all ones, consider using [`select`] which is faster.
  ///
  /// [mask]: crate#masks
  /// [`select`]: Self::select
  #[inline]
  #[must_use]
  pub fn bitselect(self, if_one: Self, if_zero: Self) -> Self {
    SimdBackend::bitselect(self, if_one, if_zero)
  }

  /// Elementwise selection.
  ///
  /// For each element of `self`:
  ///
  /// - If all bits are one, return the corresponding element of `if_true`
  /// - If all bits are zero, return the corresponding element of `if_false`
  ///
  /// This function assumes `self` is a [mask], meaning each element is either
  /// all zeros or all ones. For bitwise selection use [`bitselect`].
  ///
  /// [mask]: crate#masks
  /// [`bitselect`]: Self::bitselect
  #[inline]
  #[must_use]
  pub fn select(self, if_true: Self, if_false: Self) -> Self {
    SimdBackend::select(self, if_true, if_false)
  }

  /// Converts to a bitmask, where each bit is `1` if the element of `self` is
  /// true or `0` if the element of `self` is false.
  ///
  /// Each bit of the output corresponds to an element of `self`. The least
  /// significant bit corresponds to the lowest element. Remaining bits are `0`.
  ///
  /// This currently returns [`u32`] for all SIMD types.
  ///
  /// This function assumes `self` is a [mask], meaning each element is either
  /// all zeros or all ones. If the input is not a mask, the result is
  /// unspecified.
  ///
  /// [mask]: crate#masks
  #[inline]
  #[must_use]
  #[doc(alias("movemask", "move_mask"))]
  pub fn to_bitmask(self) -> <Self as SupportedSimd>::Bitmask {
    SimdBackend::to_bitmask(self)
  }

  /// Returns `true` if any element of `self` is true.
  ///
  /// This function assumes `self` is a [mask], meaning each element is either
  /// all zeros or all ones. If the input is not a mask, the result is
  /// unspecified.
  ///
  /// [mask]: crate#masks
  #[inline]
  #[must_use]
  pub fn any(self) -> bool {
    SimdBackend::any(self)
  }

  /// Returns `true` if all elements of `self` are true.
  ///
  /// This function assumes `self` is a [mask], meaning each element is either
  /// all zeros or all ones. If the input is not a mask, the result is
  /// unspecified.
  ///
  /// [mask]: crate#masks
  #[inline]
  #[must_use]
  pub fn all(self) -> bool {
    SimdBackend::all(self)
  }

  /// Returns `true` if none of the elements of `self` are true.
  ///
  /// This function assumes `self` is a [mask], meaning each element is either
  /// all zeros or all ones. If the input is not a mask, the result is
  /// unspecified.
  ///
  /// [mask]: crate#masks
  #[inline]
  #[must_use]
  pub fn none(self) -> bool {
    !self.any()
  }

  /// Transposes an array of SIMD vectors interpreted as a square matrix.
  #[inline]
  #[must_use]
  pub fn transpose(matrix: [Self; N]) -> [Self; N] {
    <Self as SimdBackend>::transpose(matrix)
  }

  /// Elementwise selection.
  ///
  /// For each element of `self`:
  ///
  /// - If all bits are one, return the corresponding element of `if_true`
  /// - If all bits are zero, return the corresponding element of `if_false`
  ///
  /// Originally this function did not specify whether it supported per-bit
  /// selection, or if it assumed `self` is a [mask], meaning each element is
  /// either all zeros or all ones (for better performance). Because of this,
  /// [`blend`] has been split into two new functions: [`select`] and
  /// [`bitselect`].
  ///
  /// [mask]: crate#masks
  /// [`blend`]: Self::blend
  /// [`select`]: Self::select
  /// [`bitselect`]: Self::bitselect
  #[deprecated(
    since = "1.6.0",
    note = "split into `select` and `bitselect` functions"
  )]
  #[inline]
  #[must_use]
  pub fn blend(self, if_true: Self, if_false: Self) -> Self {
    SimdBackend::select(self, if_true, if_false)
  }
}

impl SimdElement for f32 {}
impl SimdElement for f64 {}
impl SimdElement for i8 {}
impl SimdElement for i16 {}
impl SimdElement for i32 {}
impl SimdElement for i64 {}
impl SimdElement for i128 {}
impl SimdElement for isize {}
impl SimdElement for u8 {}
impl SimdElement for u16 {}
impl SimdElement for u32 {}
impl SimdElement for u64 {}
impl SimdElement for u128 {}
impl SimdElement for usize {}

impl<T, const N: usize> SimdAssociatedTypes for Simd<T, N>
where
  T: SimdElement,
  Self: SupportedSimd,
{
  type T = T;

  type SelfxN = [Self; N];
}

impl NegOrUint for f32 {}
impl NegOrUint for f64 {}
impl NegOrUint for i8 {}
impl NegOrUint for i16 {}
impl NegOrUint for i32 {}
impl NegOrUint for i64 {}
impl NegOrUint for i128 {}
impl NegOrUint for isize {}
impl NegOrUint for u8 {}
impl NegOrUint for u16 {}
impl NegOrUint for u32 {}
impl NegOrUint for u64 {}
impl NegOrUint for u128 {}
impl NegOrUint for usize {}

impl Sealed for f32 {}
impl Sealed for f64 {}
impl Sealed for i8 {}
impl Sealed for i16 {}
impl Sealed for i32 {}
impl Sealed for i64 {}
impl Sealed for i128 {}
impl Sealed for isize {}
impl Sealed for u8 {}
impl Sealed for u16 {}
impl Sealed for u32 {}
impl Sealed for u64 {}
impl Sealed for u128 {}
impl Sealed for usize {}

#[cfg(feature = "serde")]
mod impl_serde {
  use core::marker::PhantomData;

  use serde_core::{
    Deserialize, Serialize,
    de::{Error, Visitor},
    ser::SerializeTuple,
  };

  use crate::{Simd, SimdElement, SupportedSimd};

  impl<T: Serialize, const N: usize> Serialize for Simd<T, N>
  where
    T: SimdElement,
    Self: SupportedSimd,
  {
    #[inline]
    fn serialize<S>(&self, serializer: S) -> Result<S::Ok, S::Error>
    where
      S: serde_core::Serializer,
    {
      let array = self.as_array();
      let mut seq = serializer.serialize_tuple(N)?;
      for e in array {
        seq.serialize_element(e)?;
      }
      seq.end()
    }
  }

  impl<'de, T: Deserialize<'de>, const N: usize> Deserialize<'de> for Simd<T, N>
  where
    T: SimdElement,
    Self: SupportedSimd,
  {
    #[inline]
    fn deserialize<D>(deserializer: D) -> Result<Self, D::Error>
    where
      D: serde_core::Deserializer<'de>,
    {
      struct SimdVisitor<T, const N: usize>(PhantomData<[T; N]>);

      impl<'de, T: Deserialize<'de>, const N: usize> Visitor<'de>
        for SimdVisitor<T, N>
      where
        T: SimdElement,
        Simd<T, N>: SupportedSimd,
      {
        type Value = Simd<T, N>;

        #[inline]
        fn expecting(
          &self,
          formatter: &mut core::fmt::Formatter,
        ) -> core::fmt::Result {
          formatter.write_fmt(format_args!("a tuple of size {N}"))
        }

        #[inline]
        fn visit_seq<A>(self, mut seq: A) -> Result<Self::Value, A::Error>
        where
          A: serde_core::de::SeqAccess<'de>,
        {
          // SAFETY: All supported SIMD types satisfy the requirements of
          // `bytemuck::Pod`.
          let mut result = unsafe { core::mem::zeroed::<Simd<T, N>>() };

          for i in 0..N {
            result.as_mut_array()[i] = match seq.next_element()? {
              Some(value) => value,
              None => return Err(Error::invalid_length(i, &self)),
            };
          }

          Ok(result)
        }
      }

      deserializer.deserialize_tuple(N, SimdVisitor(PhantomData))
    }
  }
}
