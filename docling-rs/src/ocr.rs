//! OCR support via Tesseract for extracting text from image-based PDFs.
//!
//! OCR can be disabled by setting the environment variable `DOCLING_OCR=0`.

use image::DynamicImage;

/// Check if OCR is enabled via environment variable.
/// OCR is enabled by default; set DOCLING_OCR=0 to disable.
fn is_ocr_enabled() -> bool {
    match std::env::var("DOCLING_OCR") {
        Ok(val) => val != "0" && val.to_lowercase() != "false",
        Err(_) => true, // Default to enabled
    }
}

/// Perform OCR on an image and return the extracted text.
///
/// Returns `None` if:
/// - Tesseract is not installed
/// - OCR fails for any reason
/// - No text is found in the image
///
/// The image is processed at its current resolution. For best results,
/// render PDF pages at 200-300 DPI before calling this function.
#[cfg(feature = "ocr")]
pub fn ocr_image_to_text(image: &DynamicImage) -> Option<String> {
    use rusty_tesseract::{Args, Image};
    use std::collections::HashMap;

    // Check if OCR is disabled via environment variable
    if !is_ocr_enabled() {
        log::debug!("OCR disabled via DOCLING_OCR environment variable");
        return None;
    }

    // Convert DynamicImage to rusty_tesseract Image
    let tess_image = match Image::from_dynamic_image(image) {
        Ok(img) => img,
        Err(e) => {
            log::debug!("Failed to convert image for OCR: {}", e);
            return None;
        }
    };

    // Configure Tesseract with sensible defaults for document text
    let args = Args {
        lang: "eng".to_string(),
        config_variables: HashMap::new(),
        dpi: Some(200), // Hint for text size estimation
        psm: Some(3),   // Fully automatic page segmentation
        oem: Some(3),   // Default OCR engine mode (LSTM + legacy)
    };

    // Run OCR
    match rusty_tesseract::image_to_string(&tess_image, &args) {
        Ok(text) => {
            let trimmed = text.trim();
            if trimmed.is_empty() {
                log::debug!("OCR returned empty text");
                None
            } else {
                log::info!("OCR extracted {} characters", trimmed.len());
                Some(trimmed.to_string())
            }
        }
        Err(e) => {
            // Check if this is a "tesseract not found" error
            let err_str = e.to_string();
            if err_str.contains("not found") || err_str.contains("No such file") {
                log::warn!("Tesseract not installed, OCR unavailable: {}", e);
            } else {
                log::debug!("OCR failed: {}", e);
            }
            None
        }
    }
}

/// Check if Tesseract OCR is available on the system.
#[cfg(feature = "ocr")]
pub fn is_tesseract_available() -> bool {
    use std::process::Command;

    match Command::new("tesseract").arg("--version").output() {
        Ok(output) => output.status.success(),
        Err(_) => false,
    }
}

/// Stub function when OCR feature is disabled.
#[cfg(not(feature = "ocr"))]
pub fn ocr_image_to_text(_image: &DynamicImage) -> Option<String> {
    None
}

/// Stub function when OCR feature is disabled.
#[cfg(not(feature = "ocr"))]
pub fn is_tesseract_available() -> bool {
    false
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_tesseract_check() {
        // This test just verifies the function doesn't panic
        let available = is_tesseract_available();
        println!("Tesseract available: {}", available);
    }
}
