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
pub trait SupportedSimd: SimdBackend {}

/// Controls the internal implementation of [`Simd<T, N>`].
///
/// # Safety
///
/// - `Self` must be a `Simd<T, N>` type
/// - `Inner` must satisfy all requirements of `bytemuck::Pod`
/// - `Inner` must have a size and alignment equal to `size_of::<T>() * N`
pub(crate) unsafe trait SimdBackend {
  type Inner: Copy;
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
