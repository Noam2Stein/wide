/*
NOTE FOR CONTRIBUTORS:

These primitive traits should remain without any associated items.

In the future, when Rust gains more type system features, these traits could be
replaced with the traits from the `num-primitive` crate. To make that possible,
our traits need to stay compatible, and ideally remain fully item-less.
*/

use crate::SimdElement;

/// A marker trait for all primitive floating-point types.
///
/// Implemented for [`f64`] and [`f64`].
///
/// This trait can be used in generic contexts to access float-specific SIMD
/// functionality. This trait does not expose any functions directly.
///
/// # Example
///
/// ```
/// # use wide::{PrimitiveFloat, Simd, SupportedSimd};
/// #
/// fn generic<T, const N: usize>(x: Simd<T, N>) -> Simd<T, N>
/// where
///     T: PrimitiveFloat,
///     Simd<T, N>: SupportedSimd,
/// {
///     // Use float-specific SIMD functions.
///     x.sin()
/// }
/// ```
pub trait PrimitiveFloat: SimdElement {}

/// A marker trait for all primitive integer types.
///
/// Implemented for [`i8`], [`i16`], [`i32`], [`i64`], [`i128`], [`isize`],
/// [`u8`], [`u16`], [`u32`], [`u64`], [`u128`] and [`usize`].
///
/// This trait can be used in generic contexts to access integer-specific SIMD
/// functionality. This trait does not expose any functions directly.
///
/// Note that currently, some integer functions cannot be used in generic
/// contexts, because their names conflict with generic float functions. This
/// will be fixed once Rust supports [disjoint implementations].
///
/// # Example
///
/// ```
/// # use wide::{PrimitiveInteger, Simd, SupportedSimd};
/// #
/// fn generic<T, const N: usize>(x: Simd<T, N>) -> Simd<T, N>
/// where
///     T: PrimitiveInteger,
///     Simd<T, N>: SupportedSimd,
/// {
///     // Use integer-specific SIMD functions.
///     x.saturating_add(x)
/// }
/// ```
///
/// [disjoint implementations]: https://github.com/rust-lang/rust/issues/20400
pub trait PrimitiveInteger: SimdElement {}

/// A marker trait for all primitive signed-integer types.
///
/// Implemented for [`i8`], [`i16`], [`i32`], [`i64`], [`i128`] and [`isize`].
///
/// This trait can be used in generic contexts to access signed-integer-specific
/// SIMD functionality. This trait does not expose any functions directly.
///
/// Note that currently, some integer functions cannot be used in generic
/// contexts, because their names conflict with generic float functions. This
/// will be fixed once Rust supports [disjoint implementations].
///
/// # Example
///
/// ```
/// # use wide::{PrimitiveSigned, Simd, SupportedSimd};
/// #
/// fn generic<T, const N: usize>(x: Simd<T, N>) -> Simd<T, N>
/// where
///     T: PrimitiveSigned,
///     Simd<T, N>: SupportedSimd,
/// {
///     // Use signed-integer-specific SIMD functions.
///     x.is_positive()
/// }
/// ```
///
/// [disjoint implementations]: https://github.com/rust-lang/rust/issues/20400
pub trait PrimitiveSigned: PrimitiveInteger {}

/// A marker trait for all primitive unsigned-integer types.
///
/// Implemented for [`u8`], [`u16`], [`u32`], [`u64`], [`u128`] and [`usize`].
///
/// This trait can be used in generic contexts to access
/// unsigned-integer-specific SIMD functionality. This trait does not expose any
/// functions directly.
///
/// Note that currently, some integer functions cannot be used in generic
/// contexts, because their names conflict with generic float functions. This
/// will be fixed once Rust supports [disjoint implementations].
///
/// # Example
///
/// ```
/// # use wide::{PrimitiveUnsigned, Simd, SimdElement, SupportedSimd};
/// #
/// fn generic<T, const N: usize>(x: Simd<T, N>) -> Simd<T::Int, N>
/// where
///     T: PrimitiveUnsigned,
///     Simd<T, N>: SupportedSimd,
/// {
///     // Use unsigned-integer-specific SIMD functions.
///     x.cast_signed()
/// }
/// ```
///
/// [disjoint implementations]: https://github.com/rust-lang/rust/issues/20400
pub trait PrimitiveUnsigned: PrimitiveInteger {}

/// A marker trait for types that are either floats or unsigned-integers.
///
/// This is used for the blanket implementation of `SimdSignedBackend`.
pub(crate) trait PrimitiveFloatOrPrimitiveUnsigned: SimdElement {}

impl PrimitiveFloat for f32 {}
impl PrimitiveFloat for f64 {}

impl PrimitiveInteger for i8 {}
impl PrimitiveInteger for i16 {}
impl PrimitiveInteger for i32 {}
impl PrimitiveInteger for i64 {}
impl PrimitiveInteger for i128 {}
impl PrimitiveInteger for isize {}
impl PrimitiveInteger for u8 {}
impl PrimitiveInteger for u16 {}
impl PrimitiveInteger for u32 {}
impl PrimitiveInteger for u64 {}
impl PrimitiveInteger for u128 {}
impl PrimitiveInteger for usize {}

impl PrimitiveSigned for i8 {}
impl PrimitiveSigned for i16 {}
impl PrimitiveSigned for i32 {}
impl PrimitiveSigned for i64 {}
impl PrimitiveSigned for i128 {}
impl PrimitiveSigned for isize {}

impl PrimitiveUnsigned for u8 {}
impl PrimitiveUnsigned for u16 {}
impl PrimitiveUnsigned for u32 {}
impl PrimitiveUnsigned for u64 {}
impl PrimitiveUnsigned for u128 {}
impl PrimitiveUnsigned for usize {}

impl PrimitiveFloatOrPrimitiveUnsigned for f32 {}
impl PrimitiveFloatOrPrimitiveUnsigned for f64 {}
impl PrimitiveFloatOrPrimitiveUnsigned for u8 {}
impl PrimitiveFloatOrPrimitiveUnsigned for u16 {}
impl PrimitiveFloatOrPrimitiveUnsigned for u32 {}
impl PrimitiveFloatOrPrimitiveUnsigned for u64 {}
impl PrimitiveFloatOrPrimitiveUnsigned for u128 {}
impl PrimitiveFloatOrPrimitiveUnsigned for usize {}
