#[cfg(feature = "dfdx")]
pub mod dfdx;
#[cfg(feature = "dfdx")]
pub use self::dfdx::Gpt2;

#[cfg(not(feature = "dfdx"))]
pub mod raw;
#[cfg(not(feature = "dfdx"))]
pub use raw::Gpt2;
