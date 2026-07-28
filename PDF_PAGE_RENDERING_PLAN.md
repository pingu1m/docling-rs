# Plan: PDF Page Rendering for Vector Diagram Extraction

## Problem

Rust currently extracts only **raster XObject images** from PDFs via `pdf_oxide::extract_images()`. This correctly captures embedded screenshots and photographs, but **misses vector-drawn diagrams** — content composed of PDF path operations, shapes, fills, and text that together form a visual diagram.

In the SageMaker PDF benchmark:
- **Python docling** captures 11 large content images including 3 key vector diagrams (MLOps Venn diagram, Foundation block diagram, CI/CD Architecture diagram)
- **Rust** captures 8 raster images (all correct) but the 3 vector diagrams are invisible to `extract_images()`

Python solves this by (1) running an ML layout model on the rendered page to identify picture regions, then (2) cropping those regions from the rendered page using pypdfium2.

## Solution: Hybrid Image Extraction with pdfium Page Rendering

Add a page rendering capability using **pdfium** (via `pdfium-render` crate) to detect and rasterize regions that contain vector graphics. This complements the existing `pdf_oxide::extract_images()` for raster images.

### Architecture

```
PDF Page
  ├── pdf_oxide::extract_images()  →  Raster XObject images (screenshots, photos)
  │     [existing, with size/dedup/entropy filters]
  │
  └── pdfium page rendering         →  Vector diagram regions
        1. Render full page at 150 DPI
        2. Identify "visual content gaps" between text blocks
        3. Render those regions at 200 DPI
        4. Filter out blank/near-blank regions
        5. Emit as picture items
```

### Why pdfium-render

| Criterion | pdfium-render | mupdf | hayro |
|-----------|---------------|-------|-------|
| License | MIT/Apache-2.0 | AGPL-3.0 | MIT/Apache-2.0 |
| Region rendering | Yes (crop box + clip) | Workaround only | No |
| Maturity | 775k downloads, 26 reverse deps | 526k downloads | 208k early-stage |
| Binary distribution | pdfium-bind crate auto-downloads | mupdf-sys build | Pure Rust |
| Image format support | Returns `image::DynamicImage` | Custom Pixmap | Custom pixmap |

`pdfium-render` is the clear winner: permissive license, mature bindings, region rendering support, and returns `image::DynamicImage` which we already use.

### Binary distribution strategy

Use `pdfium-bind` crate (or `pdfium-auto`) which auto-downloads prebuilt PDFium binaries from [bblanchon/pdfium-binaries](https://github.com/bblanchon/pdfium-binaries). Supports static linking for self-contained deployment. Platforms: macOS arm64/x86_64, Linux x86_64/aarch64, Windows.

## Implementation Steps

### Step 1: Add pdfium-render dependency

**File: `docling-rs/Cargo.toml`**

Add `pdfium-render` as an optional dependency behind a feature flag:

```toml
[dependencies]
pdfium-render = { version = "0.8", optional = true }

[features]
default = ["pdf-oxide", "pdfium-render"]
pdf-oxide = ["dep:pdf_oxide"]
pdfium-render = ["dep:pdfium-render"]
```

For PDFium binary resolution at runtime, the `pdfium-render` crate's `Pdfium::bind_to_system_library()` or `Pdfium::bind_to_library()` handles dynamic loading. For static linking, use the `static` feature or bundle via `pdfium-bind`.

### Step 2: Create page rendering infrastructure

**File: `docling-rs/src/backend/pdf.rs`**

Add a function that initializes a pdfium instance and renders a page:

```rust
#[cfg(feature = "pdfium-render")]
fn render_page_region(
    pdfium: &Pdfium,
    pdf_bytes: &[u8],
    page_index: u32,
    region: (f64, f64, f64, f64),  // (l, t, r, b) in TOPLEFT coords
    page_height: f64,
    dpi: u16,
) -> Option<image::DynamicImage> {
    // 1. Open document with pdfium
    // 2. Get the page
    // 3. Set crop box to the region (convert TOPLEFT to PDF bottom-left coords)
    // 4. Render at specified DPI
    // 5. Return as DynamicImage
}
```

### Step 3: Detect vector diagram regions (gap analysis)

The key heuristic: **find large rectangular areas on each page that are between text blocks but contain no text.** These are likely diagram/figure regions.

**Algorithm:**

```
Input: page dimensions (W, H), text block bounding boxes
Output: candidate diagram regions

1. Create a vertical "occupancy map" of the page:
   - For each text block, mark its vertical span as "occupied"
   - Find large vertical gaps (> 15% of page height)

2. For each vertical gap:
   - Determine the horizontal extent (full page width or narrower)
   - This is a candidate diagram region

3. Filter candidates:
   - Skip regions that are too narrow (< 20% of page width)
   - Skip regions that are too short (< 10% of page height)
   - Skip regions at the very top or bottom of the page (likely headers/footers)
   - Merge overlapping/adjacent regions

4. Return candidate regions as (l, t, r, b) bounding boxes
```

This is inspired by how document layout analysis works — text blocks define structure, and the spaces between them contain figures/diagrams.

**File: `docling-rs/src/backend/pdf.rs`**

```rust
fn detect_diagram_regions(
    blocks: &[AssembledBlock],
    page_width: f64,
    page_height: f64,
) -> Vec<(f64, f64, f64, f64)> {
    // Implementation of gap analysis algorithm
}
```

### Step 4: Filter blank rendered regions

After rendering a candidate region, check if it actually contains visual content:

```rust
fn region_has_visual_content(img: &image::DynamicImage) -> bool {
    // Sample pixels across the image
    // If >95% of pixels are near-identical (solid color), return false
    // Use standard deviation of pixel values as a simplicity metric
    // A diagram will have significant color variation; a blank area won't
}
```

### Step 5: Integrate into emit_images pipeline

**File: `docling-rs/src/backend/pdf.rs`**

Add a new function `emit_rendered_diagrams` that:

1. Takes the assembled text blocks for the page
2. Calls `detect_diagram_regions()` to find candidate areas
3. Renders each region with pdfium at 200 DPI
4. Filters out blank regions with `region_has_visual_content()`
5. Applies content-hash dedup (shared with raster dedup)
6. Emits remaining images as `doc.add_picture()` with provenance

Call this alongside `emit_images_oxide()` in the `convert()` loop.

### Step 6: Avoid duplicate extraction

Raster images (from `pdf_oxide`) and rendered regions may overlap. To prevent duplicates:

1. After extracting raster images, record their bounding boxes
2. Before rendering a diagram region, check if it substantially overlaps (>50% IoU) with any raster image bbox
3. Skip the rendered region if a raster image already covers it

### Step 7: Update convert() orchestration

```rust
// Phase 3: Classify and emit
let mut seen_image_hashes: HashSet<u64> = HashSet::new();
let pdfium = init_pdfium(); // cached once

for page_data in &all_pages {
    // ... existing text classification ...

    // Emit raster images (existing)
    let raster_bboxes = emit_images_oxide(...);

    // Emit rendered vector diagrams (new)
    emit_rendered_diagrams(
        &mut doc,
        &pdfium,
        &data,
        page_data,
        &raster_bboxes,
        &mut seen_image_hashes,
    );
}
```

### Step 8: Regenerate groundtruth and verify

1. Rebuild release binary
2. Run on SageMaker PDF and compare with Python
3. Regenerate all PDF groundtruth files
4. Run E2E tests
5. Run `compare.sh` to validate

## Expected Outcome

After implementation:

| Metric | Before | After (expected) | Python |
|--------|--------|-------------------|--------|
| Raster images | 8 | 8 | — |
| Vector diagrams | 0 | 3-5 | 3 |
| Total images | 8 | 11-13 | 11 large + 8 medium |
| Total size | 6.3 MB | ~8-10 MB | 3.0 MB |
| Junk images | 0 | 0 | 22 |
| Speed | <1s | ~2-3s | 31s |

The Rust version should capture all meaningful content images (both raster and vector) while remaining 10-15x faster than Python and producing zero junk images.

## Risk Mitigation

1. **PDFium binary distribution**: Use `pdfium-render`'s dynamic binding with fallback. If PDFium is not available at runtime, gracefully degrade to raster-only extraction (current behavior). This keeps the feature additive.

2. **False positive regions**: The gap analysis may identify regions that are just whitespace or decorative backgrounds. The `region_has_visual_content()` filter handles this by checking pixel variance.

3. **Performance**: Full-page rendering at 150 DPI adds ~100-200ms per page. Region rendering at 200 DPI is faster (smaller area). Total overhead should be <1s for a 30-page document.

4. **Feature flag isolation**: All pdfium code is behind `#[cfg(feature = "pdfium-render")]`. Without the feature, behavior is identical to current (raster-only). This enables clean CI and optional deployment.

## Files Modified

- `docling-rs/Cargo.toml` — add `pdfium-render` dependency + feature flag
- `docling-rs/src/backend/pdf.rs` — add rendering functions, gap analysis, integration
- `e2e/tests/pdf.rs` — update thresholds if needed after groundtruth regen
- `tests/data/groundtruth/docling_v2/*.json` and `*.md` — regenerated

## Dependencies Added

- `pdfium-render = "0.8"` (MIT/Apache-2.0) — Rust bindings to PDFium
- Runtime: PDFium shared library (auto-resolved via `Pdfium::bind_to_system_library()` or bundled)
