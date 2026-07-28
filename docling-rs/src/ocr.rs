//! OCR support via Tesseract for extracting text from image-based PDFs.
//!
//! Provides two levels of OCR output:
//! - `ocr_image_to_text`: plain text (backward-compatible convenience wrapper)
//! - `ocr_image_to_blocks`: structured output with bounding boxes, block/paragraph/line
//!   hierarchy, confidence scores, and text heights for layout-aware processing.
//!
//! OCR can be disabled by setting the environment variable `DOCLING_OCR=0`.

use image::DynamicImage;

// ---------------------------------------------------------------------------
// Structured OCR types
// ---------------------------------------------------------------------------

#[derive(Debug, Clone)]
pub struct OcrWord {
    pub text: String,
    pub bbox: (i32, i32, i32, i32), // (left, top, width, height)
    pub confidence: f32,
}

#[derive(Debug, Clone)]
pub struct OcrLine {
    pub words: Vec<OcrWord>,
    pub bbox: (i32, i32, i32, i32),
}

impl OcrLine {
    pub fn text(&self) -> String {
        self.words
            .iter()
            .map(|w| w.text.as_str())
            .collect::<Vec<_>>()
            .join(" ")
    }

    pub fn median_word_height(&self) -> f64 {
        let mut heights: Vec<i32> = self.words.iter().map(|w| w.bbox.3).collect();
        if heights.is_empty() {
            return 0.0;
        }
        heights.sort();
        heights[heights.len() / 2] as f64
    }
}

#[derive(Debug, Clone)]
pub struct OcrParagraph {
    pub lines: Vec<OcrLine>,
    pub bbox: (i32, i32, i32, i32),
}

impl OcrParagraph {
    pub fn text(&self) -> String {
        self.lines
            .iter()
            .map(|l| l.text())
            .collect::<Vec<_>>()
            .join(" ")
    }
}

#[derive(Debug, Clone)]
pub struct OcrBlock {
    pub paragraphs: Vec<OcrParagraph>,
    pub bbox: (i32, i32, i32, i32),
    pub median_text_height: f64,
    pub median_confidence: f32,
}

impl OcrBlock {
    pub fn text(&self) -> String {
        self.paragraphs
            .iter()
            .map(|p| p.text())
            .collect::<Vec<_>>()
            .join("\n\n")
    }

    pub fn word_count(&self) -> usize {
        self.paragraphs
            .iter()
            .flat_map(|p| &p.lines)
            .flat_map(|l| &l.words)
            .count()
    }

    pub fn low_confidence_word_ratio(&self) -> f64 {
        let words: Vec<&OcrWord> = self
            .paragraphs
            .iter()
            .flat_map(|p| &p.lines)
            .flat_map(|l| &l.words)
            .collect();
        if words.is_empty() {
            return 1.0;
        }
        let low = words.iter().filter(|w| w.confidence < 30.0).count();
        low as f64 / words.len() as f64
    }
}

#[derive(Debug, Clone)]
pub struct OcrPageResult {
    pub blocks: Vec<OcrBlock>,
    pub page_width: i32,
    pub page_height: i32,
    pub median_text_height: f64,
}

// ---------------------------------------------------------------------------
// Internal helpers
// ---------------------------------------------------------------------------

fn is_ocr_enabled() -> bool {
    match std::env::var("DOCLING_OCR") {
        Ok(val) => val != "0" && val.to_lowercase() != "false",
        Err(_) => true,
    }
}

fn merge_bbox(a: (i32, i32, i32, i32), b: (i32, i32, i32, i32)) -> (i32, i32, i32, i32) {
    let al = a.0;
    let at = a.1;
    let ar = a.0 + a.2;
    let ab = a.1 + a.3;
    let bl = b.0;
    let bt = b.1;
    let br = b.0 + b.2;
    let bb = b.1 + b.3;
    let l = al.min(bl);
    let t = at.min(bt);
    let r = ar.max(br);
    let b = ab.max(bb);
    (l, t, r - l, b - t)
}

fn median_i32(values: &mut Vec<i32>) -> f64 {
    if values.is_empty() {
        return 0.0;
    }
    values.sort();
    values[values.len() / 2] as f64
}

fn median_f32(values: &mut Vec<f32>) -> f32 {
    if values.is_empty() {
        return 0.0;
    }
    values.sort_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal));
    values[values.len() / 2]
}

#[cfg(feature = "ocr")]
fn build_tesseract_args(dpi: u32) -> rusty_tesseract::Args {
    use std::collections::HashMap;
    rusty_tesseract::Args {
        lang: "eng".to_string(),
        config_variables: HashMap::new(),
        dpi: Some(dpi as i32),
        psm: Some(3), // fully automatic page segmentation
        oem: Some(3), // LSTM + legacy
    }
}

// ---------------------------------------------------------------------------
// Structured OCR: image_to_data based
// ---------------------------------------------------------------------------

/// Perform OCR and return structured block/paragraph/line/word data with
/// bounding boxes, confidence scores, and text height metrics.
///
/// Returns `None` if OCR is disabled, Tesseract is missing, or no text found.
#[cfg(feature = "ocr")]
pub fn ocr_image_to_blocks(image: &DynamicImage, dpi: u32) -> Option<OcrPageResult> {
    use rusty_tesseract::Image;
    use std::collections::BTreeMap;

    if !is_ocr_enabled() {
        log::debug!("OCR disabled via DOCLING_OCR environment variable");
        return None;
    }

    let tess_image = match Image::from_dynamic_image(image) {
        Ok(img) => img,
        Err(e) => {
            log::debug!("Failed to convert image for OCR: {}", e);
            return None;
        }
    };

    let args = build_tesseract_args(dpi);

    let data_output = match rusty_tesseract::image_to_data(&tess_image, &args) {
        Ok(d) => d,
        Err(e) => {
            let err_str = e.to_string();
            if err_str.contains("not found") || err_str.contains("No such file") {
                log::warn!("Tesseract not installed, OCR unavailable: {}", e);
            } else {
                log::debug!("OCR image_to_data failed: {}", e);
            }
            return None;
        }
    };

    // Group word-level entries by (block_num, par_num, line_num).
    // BTreeMap keeps blocks/paragraphs/lines in order.
    type LineKey = (i32, i32, i32);
    let mut line_map: BTreeMap<LineKey, Vec<OcrWord>> = BTreeMap::new();

    for entry in &data_output.data {
        if entry.level != 5 {
            continue; // only word-level entries
        }
        let text = entry.text.trim();
        if text.is_empty() {
            continue;
        }
        // Skip very low-confidence garbage
        if entry.conf < 0.0 {
            continue;
        }
        let key = (entry.block_num, entry.par_num, entry.line_num);
        line_map.entry(key).or_default().push(OcrWord {
            text: text.to_string(),
            bbox: (entry.left, entry.top, entry.width, entry.height),
            confidence: entry.conf,
        });
    }

    if line_map.is_empty() {
        log::debug!("OCR returned no word-level data");
        return None;
    }

    // Build hierarchy: block -> paragraph -> line
    type ParKey = (i32, i32);
    let mut par_map: BTreeMap<ParKey, Vec<(i32, Vec<OcrWord>)>> = BTreeMap::new();
    for ((block_num, par_num, line_num), words) in line_map {
        par_map
            .entry((block_num, par_num))
            .or_default()
            .push((line_num, words));
    }

    let mut block_map: BTreeMap<i32, Vec<(i32, Vec<(i32, Vec<OcrWord>)>)>> = BTreeMap::new();
    for ((block_num, par_num), lines) in par_map {
        block_map
            .entry(block_num)
            .or_default()
            .push((par_num, lines));
    }

    let mut blocks: Vec<OcrBlock> = Vec::new();
    let mut all_word_heights: Vec<i32> = Vec::new();

    for (_block_num, paragraphs_data) in block_map {
        let mut paragraphs: Vec<OcrParagraph> = Vec::new();
        let mut block_bbox: Option<(i32, i32, i32, i32)> = None;
        let mut block_word_heights: Vec<i32> = Vec::new();
        let mut block_confidences: Vec<f32> = Vec::new();

        for (_par_num, lines_data) in paragraphs_data {
            let mut lines: Vec<OcrLine> = Vec::new();
            let mut par_bbox: Option<(i32, i32, i32, i32)> = None;

            for (_line_num, words) in lines_data {
                if words.is_empty() {
                    continue;
                }
                let mut line_bbox = words[0].bbox;
                for w in &words[1..] {
                    line_bbox = merge_bbox(line_bbox, w.bbox);
                }
                for w in &words {
                    block_word_heights.push(w.bbox.3);
                    all_word_heights.push(w.bbox.3);
                    block_confidences.push(w.confidence);
                }
                par_bbox = Some(match par_bbox {
                    Some(pb) => merge_bbox(pb, line_bbox),
                    None => line_bbox,
                });
                lines.push(OcrLine {
                    words,
                    bbox: line_bbox,
                });
            }

            if !lines.is_empty() {
                let pb = par_bbox.unwrap_or((0, 0, 0, 0));
                block_bbox = Some(match block_bbox {
                    Some(bb) => merge_bbox(bb, pb),
                    None => pb,
                });
                paragraphs.push(OcrParagraph {
                    lines,
                    bbox: pb,
                });
            }
        }

        if !paragraphs.is_empty() {
            let bb = block_bbox.unwrap_or((0, 0, 0, 0));
            let median_h = median_i32(&mut block_word_heights);
            let median_conf = median_f32(&mut block_confidences);
            blocks.push(OcrBlock {
                paragraphs,
                bbox: bb,
                median_text_height: median_h,
                median_confidence: median_conf,
            });
        }
    }

    if blocks.is_empty() {
        log::debug!("OCR produced no blocks");
        return None;
    }

    let global_median_height = median_i32(&mut all_word_heights);
    let page_width = image.width() as i32;
    let page_height = image.height() as i32;

    let total_words: usize = blocks.iter().map(|b| b.word_count()).sum();
    log::info!(
        "OCR extracted {} blocks, {} words, median text height {:.1}px",
        blocks.len(),
        total_words,
        global_median_height
    );

    Some(OcrPageResult {
        blocks,
        page_width,
        page_height,
        median_text_height: global_median_height,
    })
}

// ---------------------------------------------------------------------------
// Plain-text convenience wrapper (backward compatible)
// ---------------------------------------------------------------------------

/// Perform OCR on an image and return the extracted text as a plain string.
#[cfg(feature = "ocr")]
pub fn ocr_image_to_text(image: &DynamicImage) -> Option<String> {
    use rusty_tesseract::Image;

    if !is_ocr_enabled() {
        log::debug!("OCR disabled via DOCLING_OCR environment variable");
        return None;
    }

    let tess_image = match Image::from_dynamic_image(image) {
        Ok(img) => img,
        Err(e) => {
            log::debug!("Failed to convert image for OCR: {}", e);
            return None;
        }
    };

    let args = build_tesseract_args(200);

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

// ---------------------------------------------------------------------------
// Stubs when OCR feature is disabled
// ---------------------------------------------------------------------------

#[cfg(not(feature = "ocr"))]
pub fn ocr_image_to_blocks(_image: &DynamicImage, _dpi: u32) -> Option<OcrPageResult> {
    None
}

#[cfg(not(feature = "ocr"))]
pub fn ocr_image_to_text(_image: &DynamicImage) -> Option<String> {
    None
}

#[cfg(not(feature = "ocr"))]
pub fn is_tesseract_available() -> bool {
    false
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_tesseract_check() {
        let available = is_tesseract_available();
        println!("Tesseract available: {}", available);
    }
}
