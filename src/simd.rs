use bytemuck::Pod;

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

/// Seals the [`SimdElement`] trait.
trait Sealed {}

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
