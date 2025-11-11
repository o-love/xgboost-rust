use crate::sys;
use std::ffi::CStr;
use std::fmt;

pub type XGBoostResult<T> = std::result::Result<T, XGBoostError>;

#[derive(Debug, Eq, PartialEq)]
pub struct XGBoostError {
    pub description: String,
}

impl XGBoostError {
    /// Check the return value from an XGBoost FFI call, and return the last error message on error.
    /// Return values of 0 are treated as success, non-zero values are treated as errors.
    pub fn check_return_value(ret_val: i32) -> XGBoostResult<()> {
        if ret_val == 0 {
            Ok(())
        } else {
            Err(XGBoostError::fetch_xgboost_error())
        }
    }

    /// Fetch current error message from XGBoost.
    fn fetch_xgboost_error() -> Self {
        let c_str = unsafe { CStr::from_ptr(sys::XGBGetLastError()) };
        let str_slice = c_str.to_str().unwrap_or("Unknown error");
        XGBoostError {
            description: str_slice.to_owned(),
        }
    }
}

impl fmt::Display for XGBoostError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "{}", self.description)
    }
}

impl std::error::Error for XGBoostError {}
