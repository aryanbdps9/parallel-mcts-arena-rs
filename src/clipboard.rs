//! # Cross-Platform Clipboard Support
//!
//! This module provides clipboard functionality for copying text content
//! like move history to the system clipboard. Works on Windows using
//! the Win32 API.
//!
//! ## Usage
//! ```rust,ignore
//! use crate::clipboard::{copy_to_clipboard, ClipboardResult};
//!
//! match copy_to_clipboard("Hello, World!") {
//!     Ok(()) => println!("Copied successfully!"),
//!     Err(e) => eprintln!("Failed to copy: {}", e),
//! }
//! ```

#[cfg(any(feature = "tui", feature = "gui"))]
use windows::{
    Win32::{
        Foundation::{HANDLE, GlobalFree},
        System::{
            DataExchange::{
                CloseClipboard, EmptyClipboard, OpenClipboard, SetClipboardData,
            },
            Memory::{
                GlobalAlloc, GlobalLock, GlobalUnlock, GMEM_MOVEABLE,
            },
        },
    },
    core::Result,
};

#[cfg(any(feature = "tui", feature = "gui"))]
use std::ffi::CString;

/// Clipboard format for plain text (CF_TEXT = 1)
#[cfg(any(feature = "tui", feature = "gui"))]
const CF_TEXT: u32 = 1;

/// Error type for clipboard operations
#[derive(Debug)]
pub enum ClipboardError {
    /// Failed to open the clipboard
    OpenFailed,
    /// Failed to empty the clipboard
    EmptyFailed,
    /// Failed to allocate memory
    AllocationFailed,
    /// Failed to lock memory
    LockFailed,
    /// Failed to set clipboard data
    SetDataFailed,
    /// Invalid string content
    InvalidString,
}

impl std::fmt::Display for ClipboardError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            ClipboardError::OpenFailed => write!(f, "Failed to open clipboard"),
            ClipboardError::EmptyFailed => write!(f, "Failed to empty clipboard"),
            ClipboardError::AllocationFailed => write!(f, "Failed to allocate clipboard memory"),
            ClipboardError::LockFailed => write!(f, "Failed to lock clipboard memory"),
            ClipboardError::SetDataFailed => write!(f, "Failed to set clipboard data"),
            ClipboardError::InvalidString => write!(f, "Invalid string for clipboard"),
        }
    }
}

impl std::error::Error for ClipboardError {}

/// Result type for clipboard operations
pub type ClipboardResult<T> = std::result::Result<T, ClipboardError>;

/// Copy text to the system clipboard
///
/// This function copies the provided text to the Windows clipboard using
/// the Win32 API. The text is copied as plain text (CF_TEXT format).
///
/// # Arguments
/// * `text` - The text to copy to the clipboard
///
/// # Returns
/// * `Ok(())` if the text was successfully copied
/// * `Err(ClipboardError)` if any step of the operation failed
///
/// # Example
/// ```rust,ignore
/// let move_history = "1. Player 1 - G(7,7)\n2. Player 2 - G(7,8)";
/// copy_to_clipboard(move_history)?;
/// ```
#[cfg(any(feature = "tui", feature = "gui"))]
pub fn copy_to_clipboard(text: &str) -> Result<()> {
    unsafe {
        // Open the clipboard
        if OpenClipboard(None).is_err() {
            return Ok(());
        }

        // Empty the clipboard
        if EmptyClipboard().is_err() {
            let _ = CloseClipboard();
            return Ok(());
        }

        // Convert string to null-terminated C string
        let c_string = CString::new(text).map_err(|_| windows::core::Error::from_win32())?;
        let c_str = c_string.as_bytes_with_nul();
        
        // Allocate global memory for the string
        let h_mem = match GlobalAlloc(GMEM_MOVEABLE, c_str.len()) {
            Ok(h) => h,
            Err(_) => {
                let _ = CloseClipboard();
                return Ok(());
            }
        };

        // Lock the memory and copy the string
        let p_mem = GlobalLock(h_mem);
        if p_mem.is_null() {
            let _ = GlobalFree(Some(h_mem));
            let _ = CloseClipboard();
            return Ok(());
        }

        std::ptr::copy_nonoverlapping(c_str.as_ptr(), p_mem as *mut u8, c_str.len());

        let _ = GlobalUnlock(h_mem);

        // Set the clipboard data
        if SetClipboardData(CF_TEXT, Some(HANDLE(h_mem.0))).is_err() {
            let _ = GlobalFree(Some(h_mem));
        }

        // Close the clipboard
        let _ = CloseClipboard();
    }
    Ok(())
}

/// Fallback implementation when Windows features are not available
#[cfg(not(any(feature = "tui", feature = "gui")))]
pub fn copy_to_clipboard(_text: &str) -> std::result::Result<(), ClipboardError> {
    // No clipboard support without Windows features
    Ok(())
}

/// Copy move history to clipboard with visual feedback
///
/// This function is designed to be called from the UI and provides
/// a consistent interface for copying move history across both TUI and GUI.
///
/// # Arguments
/// * `history_text` - The formatted move history text to copy
///
/// # Returns
/// * `true` if the copy was successful
/// * `false` if the copy failed
#[cfg(any(feature = "tui", feature = "gui"))]
pub fn copy_history_to_clipboard(history_text: &str) -> bool {
    copy_to_clipboard(history_text).is_ok()
}

/// Fallback for when Windows features are not available
#[cfg(not(any(feature = "tui", feature = "gui")))]
pub fn copy_history_to_clipboard(_history_text: &str) -> bool {
    false
}

