use bytemuck::{Pod, pod_align_to, pod_align_to_mut};

#[expect(deprecated)]
use crate::{AlignTo, CmpEq, CmpGe, CmpGt, CmpLe, CmpLt, CmpNe};

/// A generic SIMD vector with `N` elements of type `T`.
///
/// Note that only specific combinations of `T` and `N` are supported. Supported types implement the [`SupportedSimd`] trait.
/// TODO NOW improve this doc.
///
/// See the [crate level documentation] for more information about SIMD
/// vectors.
///
/// [crate level documentation]: crate
#[derive(Clone, Copy)]
pub struct Simd<T, const N: usize>(pub(crate) <Self as SimdBackend>::Inner)
where
  Self: SupportedSimd;

/// A marker trait for SIMD TODO NOW doc this.
#[expect(private_bounds)]
#[diagnostic::on_unimplemented(
  message = "`{Self}` is not a supported SIMD type",
  note = "see type aliases for a list of supported types"
)]
pub trait SupportedSimd:
  Copy + SimdBackend<Bitmask = <Self as SupportedSimd>::Bitmask>
{
  /// The type returned by [`Simd::to_bitmask`].
  ///
  /// Currently this is [`u32`] for all SIMD vectors.
  type Bitmask;
}

/// Controls the internal implementation of a [`Simd`] type. This includes
/// controlling its internal field type and its function implementations.
///
/// # Safety
///
/// The following requirements must be met:
///
/// - `Self` must be [`Simd`] of a certain `T` and `N`
/// - `T` must satisty all requirements of `bytemuck::Pod`
/// - `N` must be a power of two
/// - `Inner` must satisty all requirements of `bytemuck::Pod`
/// - `Inner` must have a size and alignment equal to `size_of::<T>() * N`
pub(crate) unsafe trait SimdBackend: SimdAssociatedTypes {
  /// Specifies the internal representation of this [`Simd`] type.
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

  fn reduce_add(self) -> Self::T;

  fn reduce_mul(self) -> Self::T;

  fn bitselect(self, if_one: Self, if_zero: Self) -> Self;

  fn select(self, if_true: Self, if_false: Self) -> Self;

  fn to_bitmask(self) -> Self::Bitmask;

  fn any(self) -> bool;

  fn all(self) -> bool;

  fn transpose(matrix: Self::Matrix) -> Self::Matrix;
}

/// TODO doc this
pub(crate) trait SimdAssociatedTypes {
  type T;
  type Matrix;
  type Bitmask;
}

macro_rules! impl_formatting_trait {
  ($Trait:path) => {
    impl<T: Copy + $Trait, const N: usize> $Trait for Simd<T, N>
    where
      Self: SupportedSimd,
    {
      #[allow(clippy::missing_inline_in_public_items)]
      fn fmt(&self, f: &mut core::fmt::Formatter) -> core::fmt::Result {
        write!(f, "(")?;
        for (i, x) in self.to_array().iter().enumerate() {
          if i > 0 {
            write!(f, ", ")?;
          }
          T::fmt(x, f)?;
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

impl<T: Copy + PartialEq, const N: usize> PartialEq for Simd<T, N>
where
  Self: SupportedSimd,
{
  #[inline]
  fn eq(&self, other: &Self) -> bool {
    self.simd_eq(*other).all()
  }
}

impl<T: Copy + Eq, const N: usize> Eq for Simd<T, N> where Self: SupportedSimd {}

impl<T: Default, const N: usize> Default for Simd<T, N>
where
  Self: SupportedSimd,
{
  #[inline]
  fn default() -> Self {
    // This works because coincidentally, all supported `T` types have a default
    // value of "all bits zero".

    // SAFETY: All SIMD types are expected to satisfy the requirements of
    // `bytemuck::Pod`.
    unsafe { core::mem::zeroed::<Simd<T, N>>() }
  }
}

impl<T: Copy, const N: usize> From<[T; N]> for Simd<T, N>
where
  Self: SupportedSimd,
{
  /// Converts an array to a SIMD vector.
  #[inline]
  fn from(array: [T; N]) -> Self {
    Self::new(array)
  }
}

impl<T: Copy, const N: usize> From<Simd<T, N>> for [T; N]
where
  Simd<T, N>: SupportedSimd,
{
  /// Converts a SIMD vector to an array.
  #[inline]
  fn from(simd: Simd<T, N>) -> Self {
    simd.to_array()
  }
}

impl<T: Copy, const N: usize> From<T> for Simd<T, N>
where
  Self: SupportedSimd,
{
  /// Converts a single value to a SIMD vector by setting all elements to
  /// that value.
  #[inline]
  fn from(value: T) -> Self {
    Self::splat(value)
  }
}

impl<T: Copy, const N: usize> From<&[T]> for Simd<T, N>
where
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
      concat!(
        "slice has more elements than `",
        stringify!($Simd),
        "` can store",
      ),
    );

    // SAFETY: `Simd<T, N>` accepts all bit-patterns, including all zeros.
    let mut result = unsafe { core::mem::zeroed::<Simd<T, N>>() };

    // SAFETY: `value` is valid for its own length, and `result` is valid
    // because its length is checked to be less than or equal to
    // `value.len()`. Both pointers are properly aligned because they
    // originate from a slice of `$T`. The regions of memory do not overlap
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

impl<T, const N: usize> Neg for Simd<T, N> where Self: SupportedSimd {}

#[expect(deprecated)]
impl<T: PartialEq, const N: usize> CmpEq for Simd<T, N>
where
  Self: SupportedSimd,
{
  type Output = Self;

  #[inline]
  fn simd_eq(self, other: Self) -> Self::Output {
    SimdBackend::simd_eq(self, other)
  }
}

#[expect(deprecated)]
impl<T: Copy + PartialEq, const N: usize> CmpEq<T> for Simd<T, N>
where
  Self: SupportedSimd,
{
  type Output = Self;

  #[inline]
  fn simd_eq(self, other: T) -> Self::Output {
    SimdBackend::simd_eq(self, Self::splat(other))
  }
}

#[expect(deprecated)]
impl<T: PartialEq, const N: usize> CmpNe for Simd<T, N>
where
  Self: SupportedSimd,
{
  type Output = Self;

  #[inline]
  fn simd_ne(self, other: Self) -> Self::Output {
    SimdBackend::simd_ne(self, other)
  }
}

#[expect(deprecated)]
impl<T: Copy + PartialEq, const N: usize> CmpNe<T> for Simd<T, N>
where
  Self: SupportedSimd,
{
  type Output = Self;

  #[inline]
  fn simd_ne(self, other: T) -> Self::Output {
    SimdBackend::simd_ne(self, Self::splat(other))
  }
}

#[expect(deprecated)]
impl<T: PartialOrd, const N: usize> CmpLt for Simd<T, N>
where
  Self: SupportedSimd,
{
  type Output = Self;

  #[inline]
  fn simd_lt(self, other: Self) -> Self::Output {
    SimdBackend::simd_lt(self, other)
  }
}

#[expect(deprecated)]
impl<T: Copy + PartialOrd, const N: usize> CmpLt<T> for Simd<T, N>
where
  Self: SupportedSimd,
{
  type Output = Self;

  #[inline]
  fn simd_lt(self, other: T) -> Self::Output {
    SimdBackend::simd_lt(self, Self::splat(other))
  }
}

#[expect(deprecated)]
impl<T: PartialOrd, const N: usize> CmpGt for Simd<T, N>
where
  Self: SupportedSimd,
{
  type Output = Self;

  #[inline]
  fn simd_gt(self, other: Self) -> Self::Output {
    SimdBackend::simd_gt(self, other)
  }
}

#[expect(deprecated)]
impl<T: Copy + PartialOrd, const N: usize> CmpGt<T> for Simd<T, N>
where
  Self: SupportedSimd,
{
  type Output = Self;

  #[inline]
  fn simd_gt(self, other: T) -> Self::Output {
    SimdBackend::simd_gt(self, Self::splat(other))
  }
}

#[expect(deprecated)]
impl<T: PartialOrd, const N: usize> CmpLe for Simd<T, N>
where
  Self: SupportedSimd,
{
  type Output = Self;

  #[inline]
  fn simd_le(self, other: Self) -> Self::Output {
    SimdBackend::simd_le(self, other)
  }
}

#[expect(deprecated)]
impl<T: Copy + PartialOrd, const N: usize> CmpLe<T> for Simd<T, N>
where
  Self: SupportedSimd,
{
  type Output = Self;

  #[inline]
  fn simd_le(self, other: T) -> Self::Output {
    SimdBackend::simd_le(self, Self::splat(other))
  }
}

#[expect(deprecated)]
impl<T: PartialOrd, const N: usize> CmpGe for Simd<T, N>
where
  Self: SupportedSimd,
{
  type Output = Self;

  #[inline]
  fn simd_ge(self, other: Self) -> Self::Output {
    SimdBackend::simd_ge(self, other)
  }
}

#[expect(deprecated)]
impl<T: Copy + PartialOrd, const N: usize> CmpGe<T> for Simd<T, N>
where
  Self: SupportedSimd,
{
  type Output = Self;

  #[inline]
  fn simd_ge(self, other: T) -> Self::Output {
    SimdBackend::simd_ge(self, Self::splat(other))
  }
}

#[expect(deprecated)]
impl<T, const N: usize> AlignTo for Simd<T, N>
where
  Self: SupportedSimd,
  T: Pod + Default + PartialEq,
{
  type Elem = T;
}

impl<T: Copy, const N: usize> Simd<T, N>
where
  Self: SupportedSimd,
{
  /// Converts an array to a SIMD vector.
  #[inline]
  #[must_use]
  pub const fn new(array: [T; N]) -> Self {
    // SAFETY: Both types satisfy all requirements of `bytemuck::Pod` and have
    // the same size.
    unsafe { core::mem::transmute_copy::<[T; N], Simd<T, N>>(&array) }
  }

  /// Constructs a new SIMD vector with all elements set to the given value.
  #[inline]
  #[must_use]
  pub const fn splat(value: T) -> Self {
    // SAFETY: Both types satisfy all requirements of `bytemuck::Pod` and have
    // the same size.
    unsafe { core::mem::transmute_copy::<[T; N], Simd<T, N>>(&[value; N]) }
  }

  /// Converts a SIMD vector to an array.
  #[inline]
  #[must_use]
  pub const fn to_array(self) -> [T; N] {
    // SAFETY: Both types satisfy all requirements of `bytemuck::Pod` and have
    // the same size.
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

  /// Returns a [mask] that checks if each element of `self` is not equal to
  /// the corresponding element of `other`.
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

  /// Returns a [mask] that checks if each element of `self` is less than
  /// the corresponding element of `other`.
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

  /// Returns a [mask] that checks if each element of `self` is greater than
  /// the corresponding element of `other`.
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

  /// Returns a [mask] that checks if each element of `self` is greater than
  /// or equal to the corresponding element of `other`.
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

  /// Bitwise selection.
  ///
  /// For each bit of `self`:
  ///
  /// - If the bit is one, return the corresponding bit of `if_one`
  /// - If the bit is zero, return the corresponding bit of `if_zero`
  ///
  /// If you know `self` is a [mask], meaning each element is either all
  /// zeros or all ones, consider using [`select`] which is faster.
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
  /// This function assumes `self` is a [mask], meaning each element is
  /// either all zeros or all ones. For bitwise selection use [`bitselect`].
  ///
  /// [mask]: crate#masks
  /// [`bitselect`]: Self::bitselect
  #[inline]
  #[must_use]
  pub fn select(self, if_true: Self, if_false: Self) -> Self {
    SimdBackend::select(self, if_true, if_false)
  }

  /// Converts to a bitmask, where each bit is `1` if the element of `self`
  /// is true or `0` if the element of `self` is false.
  ///
  /// This currently returns [`u32`] for all SIMD vectors.
  ///
  /// Each bit of the output corresponds to an element of `self`. The least
  /// significant bit corresponds to the lowest element. Remaining bits are
  /// `0`.
  ///
  /// This function assumes `self` is a [mask], meaning each element is
  /// either all zeros or all ones. If the input is not a mask, the result
  /// is unspecified.
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
  /// This function assumes `self` is a [mask], meaning each element is
  /// either all zeros or all ones. If the input is not a mask, the result
  /// is unspecified.
  ///
  /// [mask]: crate#masks
  #[inline]
  #[must_use]
  pub fn any(self) -> bool {
    SimdBackend::any(self)
  }

  /// Returns `true` if all elements of `self` are true.
  ///
  /// This function assumes `self` is a [mask], meaning each element is
  /// either all zeros or all ones. If the input is not a mask, the result
  /// is unspecified.
  ///
  /// [mask]: crate#masks
  #[inline]
  #[must_use]
  pub fn all(self) -> bool {
    SimdBackend::all(self)
  }

  /// Returns `true` if none of the elements of `self` are true.
  ///
  /// This function assumes `self` is a [mask], meaning each element is
  /// either all zeros or all ones. If the input is not a mask, the result
  /// is unspecified.
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
    // TODO BEFORE MERGE: This trick is awkward. Is there a better way?

    // SAFETY: The compile cannot understand this, but the `Matrix` associated
    // type always equals `[Self; N]`.
    unsafe {
      let matrix = core::mem::transmute_copy::<
        [Simd<T, N>; N],
        <Simd<T, N> as SimdAssociatedTypes>::Matrix,
      >(&matrix);

      let result = <Self as SimdBackend>::transpose(matrix);

      core::mem::transmute_copy::<
        <Simd<T, N> as SimdAssociatedTypes>::Matrix,
        [Simd<T, N>; N],
      >(&result)
    }
  }

  /// A SIMD variant of [`align_to`].
  ///
  /// [`align_to`]: https://doc.rust-lang.org/std/primitive.slice.html#method.align_to
  #[inline]
  pub fn simd_align_to(slice: &[T]) -> (&[T], &[Self], &[T])
  where
    T: Pod,
  {
    pod_align_to(slice)
  }

  /// A SIMD variant of [`align_to_mut`].
  ///
  /// [`align_to_mut`]: https://doc.rust-lang.org/std/primitive.slice.html#method.align_to_mut
  #[inline]
  pub fn simd_align_to_mut(slice: &mut [T]) -> (&mut [T], &mut [Self], &mut [T])
  where
    T: Pod,
  {
    pod_align_to_mut(slice)
  }

  /// Elementwise selection.
  ///
  /// For each element of `self`:
  ///
  /// - If all bits are one, return the corresponding element of `if_true`
  /// - If all bits are zero, return the corresponding element of `if_false`
  ///
  /// Originally this function did not specify whether it supported per-bit
  /// selection, or if it assumed `self` is a [mask], meaning each element
  /// is either all zeros or all ones (for better performance). Because of
  /// this, [`blend`] has been split into two new functions: [`select`] and
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

impl<T, const N: usize> SimdAssociatedTypes for Simd<T, N>
where
  Self: SupportedSimd,
{
  type T = T;
  type Matrix = [Self; N];
  type Bitmask = <Self as SupportedSimd>::Bitmask;
}

mod impl_bytemuck {
  use bytemuck::{Pod, Zeroable};

  use crate::{Simd, SupportedSimd};

  // SAFETY: All SIMD types in this library contain fully initialized memory
  // and accept all bits patterns.
  unsafe impl<T: Zeroable, const N: usize> Zeroable for Simd<T, N> where
    Self: SupportedSimd
  {
  }

  // SAFETY: All SIMD types in this library contain fully initialized memory
  // and accept all bits patterns.
  unsafe impl<T: Pod, const N: usize> Pod for Simd<T, N> where Self: SupportedSimd {}
}

#[cfg(feature = "serde")]
mod impl_serde {
  use serde_core::{Deserialize, Serialize, ser::SerializeTuple};

  use crate::{Simd, SupportedSimd};

  impl<T: Serialize, const N: usize> Serialize for Simd<T, N>
  where
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
    Self: SupportedSimd,
  {
    #[inline]
    fn deserialize<D>(deserializer: D) -> Result<Self, D::Error>
    where
      D: serde_core::Deserializer<'de>,
    {
      Ok(<[T; N]>::deserialize(deserializer)?.into())
    }
  }
}

macro_rules! impl_unary_operator {
  ($Simd:ident, $Op:ident, $op:ident, $impl:item $(, $(#[$doc:meta])*)?) => {
    impl $Op for $Simd {
      type Output = Self;

      $($(#[$doc])*)?
      $impl
    }

    impl $Op for &$Simd {
      type Output = $Simd;

      $($(#[$doc])*)?
      #[inline]
      fn $op(self) -> Self::Output {
        (*self).$op()
      }
    }
  }
}

macro_rules! impl_binary_operator {
  (
    $T:ident,
    $Simd:ident,
    $Op:ident,
    $op:ident,
    $OpAssign:ident,
    $op_assign:ident,
    $impl:item
    $(,
      $(#[$doc:meta])*,
      $(#[$doc_scalar:meta])*,
      $(#[$scalar_doc:meta])*
    )?
  ) => {
    impl $Op for $Simd {
      type Output = Self;

      $($(#[$doc])*)?
      $impl
    }

    impl $Op<$T> for $Simd {
      type Output = Self;

      $($(#[$doc_scalar])*)?
      #[inline]
      fn $op(self, rhs: $T) -> Self::Output {
        self.$op(Self::splat(rhs))
      }
    }

    impl $Op<$Simd> for $T {
      type Output = $Simd;

      $($(#[$scalar_doc])*)?
      #[inline]
      fn $op(self, rhs: $Simd) -> Self::Output {
        $Simd::splat(self).$op(rhs)
      }
    }

    impl $OpAssign for $Simd {
      $($(#[$doc])*)?
      #[inline]
      fn $op_assign(&mut self, rhs: Self) {
        *self = (*self).$op(rhs);
      }
    }

    impl $OpAssign<$T> for $Simd {
      $($(#[$doc_scalar])*)?
      #[inline]
      fn $op_assign(&mut self, rhs: $T) {
        *self = (*self).$op(Self::splat(rhs));
      }
    }

    impl $Op<&Self> for $Simd {
      type Output = Self;

      $($(#[$doc])*)?
      #[inline]
      fn $op(self, rhs: &Self) -> Self::Output {
        self.$op(*rhs)
      }
    }

    impl $Op<&$T> for $Simd {
      type Output = Self;

      $($(#[$doc_scalar])*)?
      #[inline]
      fn $op(self, rhs: &$T) -> Self::Output {
        self.$op(Self::splat(*rhs))
      }
    }

    impl $Op<&$Simd> for $T {
      type Output = $Simd;

      $($(#[$scalar_doc])*)?
      #[inline]
      fn $op(self, rhs: &$Simd) -> Self::Output {
        $Simd::splat(self).$op(*rhs)
      }
    }

    impl $OpAssign<&Self> for $Simd {
      $($(#[$doc])*)?
      #[inline]
      fn $op_assign(&mut self, rhs: &Self) {
        *self = (*self).$op(*rhs);
      }
    }

    impl $OpAssign<&$T> for $Simd {
      $($(#[$doc_scalar])*)?
      #[inline]
      fn $op_assign(&mut self, rhs: &$T) {
        *self = (*self).$op(Self::splat(*rhs));
      }
    }

    impl $Op<$Simd> for &$Simd {
      type Output = $Simd;

      $($(#[$doc])*)?
      #[inline]
      fn $op(self, rhs: $Simd) -> Self::Output {
        (*self).$op(rhs)
      }
    }

    impl $Op<$T> for &$Simd {
      type Output = $Simd;

      $($(#[$doc_scalar])*)?
      #[inline]
      fn $op(self, rhs: $T) -> Self::Output {
        (*self).$op($Simd::splat(rhs))
      }
    }

    impl $Op<$Simd> for &$T {
      type Output = $Simd;

      $($(#[$scalar_doc])*)?
      #[inline]
      fn $op(self, rhs: $Simd) -> Self::Output {
        $Simd::splat(*self).$op(rhs)
      }
    }

    impl $Op<&$Simd> for &$Simd {
      type Output = $Simd;

      $($(#[$doc])*)?
      #[inline]
      fn $op(self, rhs: &$Simd) -> Self::Output {
        (*self).$op(*rhs)
      }
    }

    impl $Op<&$T> for &$Simd {
      type Output = $Simd;

      $($(#[$doc_scalar])*)?
      #[inline]
      fn $op(self, rhs: &$T) -> Self::Output {
        (*self).$op($Simd::splat(*rhs))
      }
    }

    impl $Op<&$Simd> for &$T {
      type Output = $Simd;

      $($(#[$scalar_doc])*)?
      #[inline]
      fn $op(self, rhs: &$Simd) -> Self::Output {
        $Simd::splat(*self).$op(*rhs)
      }
    }
  }
}

macro_rules! impl_shift_operator {
  (
    $T:ident,
    $Simd:ident,
    $UnsignedSimd:ident,
    $SignedSimd:ident,
    $Op:ident,
    $op:ident,
    $OpAssign:ident,
    $op_assign:ident,
    $impl_unsigned_simd:item,
    $impl_u32:item
    $(,
      $(#[$doc:meta])*,
      $(#[$doc_scalar:meta])*,
      $(#[$scalar_doc:meta])*
    )?
  ) => {
    impl $Op<$UnsignedSimd> for $Simd {
      type Output = Self;

      $($(#[$doc])*)?
      $impl_unsigned_simd
    }

    impl $Op<$SignedSimd> for $Simd {
      type Output = Self;

      $($(#[$doc])*)?
      #[inline]
      fn $op(self, rhs: $SignedSimd) -> Self::Output {
        self.$op(cast::<$SignedSimd, $UnsignedSimd>(rhs))
      }
    }

    impl $Op<$UnsignedSimd> for $T {
      type Output = $Simd;

      $($(#[$scalar_doc])*)?
      #[inline]
      fn $op(self, rhs: $UnsignedSimd) -> Self::Output {
        $Simd::splat(self).$op(rhs)
      }
    }

    impl $Op<$SignedSimd> for $T {
      type Output = $Simd;

      $($(#[$scalar_doc])*)?
      #[inline]
      fn $op(self, rhs: $SignedSimd) -> Self::Output {
        $Simd::splat(self).$op(rhs)
      }
    }

    impl<Rhs> $OpAssign<Rhs> for $Simd
    where
      Self: $Op<Rhs, Output = Self>,
    {
      #[inline]
      fn $op_assign(&mut self, rhs: Rhs) {
        *self = (*self).$op(rhs);
      }
    }

    impl<Rhs> $Op<&Rhs> for $Simd
    where
      Self: $Op<Rhs, Output = Self>,
      Rhs: Copy,
    {
      type Output = Self;

      #[inline]
      fn $op(self, rhs: &Rhs) -> Self::Output {
        self.$op(*rhs)
      }
    }

    impl $Op<&$UnsignedSimd> for $T {
      type Output = $Simd;

      $($(#[$scalar_doc])*)?
      #[inline]
      fn $op(self, rhs: &$UnsignedSimd) -> Self::Output {
        $Simd::splat(self).$op(*rhs)
      }
    }

    impl $Op<&$SignedSimd> for $T {
      type Output = $Simd;

      $($(#[$scalar_doc])*)?
      #[inline]
      fn $op(self, rhs: &$SignedSimd) -> Self::Output {
        $Simd::splat(self).$op(*rhs)
      }
    }

    impl<Rhs> $Op<Rhs> for &$Simd
    where
      $Simd: $Op<Rhs, Output = $Simd>,
    {
      type Output = $Simd;

      #[inline]
      fn $op(self, rhs: Rhs) -> Self::Output {
        (*self).$op(rhs)
      }
    }

    impl $Op<$UnsignedSimd> for &$T {
      type Output = $Simd;

      $($(#[$scalar_doc])*)?
      #[inline]
      fn $op(self, rhs: $UnsignedSimd) -> Self::Output {
        $Simd::splat(*self).$op(rhs)
      }
    }

    impl $Op<$SignedSimd> for &$T {
      type Output = $Simd;

      $($(#[$scalar_doc])*)?
      #[inline]
      fn $op(self, rhs: $SignedSimd) -> Self::Output {
        $Simd::splat(*self).$op(rhs)
      }
    }

    impl $Op<&$UnsignedSimd> for &$T {
      type Output = $Simd;

      $($(#[$scalar_doc])*)?
      #[inline]
      fn $op(self, rhs: &$UnsignedSimd) -> Self::Output {
        $Simd::splat(*self).$op(*rhs)
      }
    }

    impl $Op<&$SignedSimd> for &$T {
      type Output = $Simd;

      $($(#[$scalar_doc])*)?
      #[inline]
      fn $op(self, rhs: &$SignedSimd) -> Self::Output {
        $Simd::splat(*self).$op(*rhs)
      }
    }

    impl $Op<u32> for $Simd {
      type Output = Self;

      $($(#[$doc_scalar])*)?
      $impl_u32
    }

    macro_rules! impl_scalar_with_cast {
      ($T2:ident) => {
        impl $Op<$T2> for $Simd {
          type Output = Self;

          $($(#[$doc_scalar])*)?
          #[inline]
          fn $op(self, rhs: $T2) -> Self::Output {
            self.$op(rhs as u32)
          }
        }
      }
    }
    impl_scalar_with_cast!(i8);
    impl_scalar_with_cast!(i16);
    impl_scalar_with_cast!(i32);
    impl_scalar_with_cast!(i64);
    impl_scalar_with_cast!(i128);
    impl_scalar_with_cast!(isize);
    impl_scalar_with_cast!(u8);
    impl_scalar_with_cast!(u16);
    impl_scalar_with_cast!(u64);
    impl_scalar_with_cast!(u128);
    impl_scalar_with_cast!(usize);
  }
}
