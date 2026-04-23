//! Slide puzzle pattern detection for gap and background images.
//!
//! The crate exposes [`PuzzleCaptchaSolver`], a small utility that mirrors the
//! behavior of the original Python implementation:
//!
//! - load the gap image and crop away surrounding whitespace
//! - run edge detection on the gap and background images
//! - locate the best matching position on the background
//! - write a result image with the detected rectangle
//!
//! # Example
//!
//! ```no_run
//! use puzzler::PuzzleCaptchaSolver;
//!
//! let solver = PuzzleCaptchaSolver::new(
//!   "fixtures/slider/gap.png",
//!   "fixtures/slider/background.png",
//!   "artifacts/slider-match.png",
//! );
//! let position = solver.discern()?;
//!
//! println!("{position}");
//! # Ok::<(), puzzler::PuzzlePatternError>(())
//! ```

use image::imageops::crop_imm;
use image::{GrayImage, ImageError, ImageReader, Rgb, RgbImage};
use rayon::prelude::*;
use std::error::Error;
use std::fmt;
use std::path::{Path, PathBuf};
use std::sync::OnceLock;

/// Errors returned while loading, processing, or matching puzzle images.
#[derive(Debug)]
pub enum PuzzlePatternError {
  /// The underlying image decoder or encoder failed.
  Image(ImageError),

  /// The gap image did not contain any non-whitespace pixels after scanning.
  NoPatternPixels,

  /// The cropped template image is larger than the background image.
  TemplateLargerThanBackground {
    template_width: u32,
    template_height: u32,
    background_width: u32,
    background_height: u32,
  },

  /// The extracted template feature map is empty, so matching cannot proceed.
  ZeroVarianceTemplate,
}

/// Finds the horizontal position of a puzzle gap inside a background image.
///
/// The solver stores three paths:
///
/// - the original gap image
/// - the original background image
/// - the output image that receives the drawn match rectangle
#[derive(Clone, Debug, Eq, PartialEq)]
pub struct PuzzleCaptchaSolver {
  gap: PathBuf,
  background: PathBuf,
  output: PathBuf,
}

#[derive(Clone, Copy, Debug)]
struct MatchCandidate {
  numerator: u64,
  region_square_sum: u64,
  x: u32,
  y: u32,
}

#[derive(Debug)]
struct PreparedTemplate {
  width: u32,
  height: u32,
  runs: Vec<TemplateRun>,
  values: Vec<u8>,
}

#[derive(Debug)]
struct PreparedBackground {
  width: u32,
  pixels: Vec<u8>,
  square_integral: Vec<u64>,
}

#[derive(Clone, Copy, Debug)]
struct SearchRegion {
  x_start: u32,
  x_end: u32,
  y_start: u32,
  y_end: u32,
}

#[derive(Clone, Debug)]
struct SearchLevel {
  template: GrayImage,
  background: GrayImage,
}

#[derive(Clone, Copy, Debug)]
struct TemplateRun {
  relative_offset: usize,
  value_offset: usize,
  len: usize,
}

#[derive(Clone, Copy)]
enum FeatureMode {
  Sobel,
  Laplacian,
}

impl fmt::Display for PuzzlePatternError {
  fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
    match self {
      Self::Image(error) => write!(formatter, "{error}"),
      Self::NoPatternPixels => {
        formatter.write_str("pattern image did not contain any non-whitespace pixels")
      }
      Self::TemplateLargerThanBackground {
        template_width,
        template_height,
        background_width,
        background_height,
      } => write!(
        formatter,
        "template size {template_width}x{template_height} exceeds background size {background_width}x{background_height}"
      ),
      Self::ZeroVarianceTemplate => {
        formatter.write_str("template feature map is empty after preprocessing")
      }
    }
  }
}

impl Error for PuzzlePatternError {
  fn source(&self) -> Option<&(dyn Error + 'static)> {
    match self {
      Self::Image(error) => Some(error),
      Self::NoPatternPixels
      | Self::TemplateLargerThanBackground { .. }
      | Self::ZeroVarianceTemplate => None,
    }
  }
}

impl From<ImageError> for PuzzlePatternError {
  fn from(error: ImageError) -> Self {
    Self::Image(error)
  }
}

impl PuzzleCaptchaSolver {
  /// Creates a new solver from the gap, background, and output image paths.
  #[must_use]
  pub fn new<P1, P2, P3>(gap_image_path: P1, bg_image_path: P2, output_image_path: P3) -> Self
  where
    P1: Into<PathBuf>,
    P2: Into<PathBuf>,
    P3: Into<PathBuf>,
  {
    Self {
      gap: gap_image_path.into(),
      background: bg_image_path.into(),
      output: output_image_path.into(),
    }
  }

  /// Runs the complete detection pipeline and returns the detected `x` offset.
  ///
  /// The method crops the gap image, extracts feature maps from both images,
  /// performs template matching, and writes an annotated output image.
  ///
  /// # Errors
  ///
  /// Returns:
  ///
  /// - [`PuzzlePatternError::Image`] if the gap image, background image, or
  ///   output image cannot be read or written
  /// - [`PuzzlePatternError::NoPatternPixels`] if the gap image contains no
  ///   non-whitespace pixels to crop
  /// - [`PuzzlePatternError::TemplateLargerThanBackground`] if the cropped gap
  ///   image is larger than the background image
  /// - [`PuzzlePatternError::ZeroVarianceTemplate`] if the processed template
  ///   contains no usable feature pixels
  pub fn discern(&self) -> Result<u32, PuzzlePatternError> {
    let gap_image = Self::remove_whitespace(self.gap.as_path())?;
    let bg_image = load_rgb_image(self.background.as_path())?;
    let primary_mode = select_feature_mode(&gap_image, &bg_image);
    let primary_gap = apply_edge_detection_with_mode(&gap_image, primary_mode);
    let primary_bg = apply_edge_detection_with_mode(&bg_image, primary_mode);
    let primary_candidates = find_match_candidates(&primary_gap, &primary_bg, 2)?;
    let (best_point, feature_gap, feature_bg) = if should_try_alternate_mode(&primary_candidates) {
      let alternate_mode = match primary_mode {
        FeatureMode::Sobel => FeatureMode::Laplacian,
        FeatureMode::Laplacian => FeatureMode::Sobel,
      };
      let alternate_gap = apply_edge_detection_with_mode(&gap_image, alternate_mode);
      let alternate_bg = apply_edge_detection_with_mode(&bg_image, alternate_mode);
      let alternate_candidates = find_match_candidates(&alternate_gap, &alternate_bg, 2)?;

      if prefer_alternate_mode(&primary_candidates, &alternate_candidates) {
        (
          best_match_point(&alternate_candidates),
          alternate_gap,
          alternate_bg,
        )
      } else {
        (
          best_match_point(&primary_candidates),
          primary_gap,
          primary_bg,
        )
      }
    } else {
      (
        best_match_point(&primary_candidates),
        primary_gap,
        primary_bg,
      )
    };
    let mut output_image = gray_to_rgb(&feature_bg);

    draw_rectangle(
      &mut output_image,
      best_point,
      feature_gap.width(),
      feature_gap.height(),
    );
    output_image.save(self.output.as_path())?;

    Ok(best_point.0)
  }

  /// Matches a preprocessed slide image against a preprocessed background.
  ///
  /// The return value is the `x` coordinate of the best match. The output file
  /// configured in the solver is updated with a red rectangle drawn over the
  /// matched region.
  ///
  /// # Errors
  ///
  /// Returns:
  ///
  /// - [`PuzzlePatternError::TemplateLargerThanBackground`] if `slide_pic`
  ///   exceeds `background_pic` in either dimension
  /// - [`PuzzlePatternError::ZeroVarianceTemplate`] if `slide_pic` contains no
  ///   usable feature pixels
  /// - [`PuzzlePatternError::Image`] if the annotated output image cannot be
  ///   written to disk
  pub fn find_position_of_slide(
    &self,
    slide_pic: &GrayImage,
    background_pic: &GrayImage,
  ) -> Result<u32, PuzzlePatternError> {
    let best_location = self.locate_slide(slide_pic, background_pic)?;
    let mut output_image = gray_to_rgb(background_pic);

    draw_rectangle(
      &mut output_image,
      best_location,
      slide_pic.width(),
      slide_pic.height(),
    );
    output_image.save(self.output.as_path())?;

    Ok(best_location.0)
  }

  /// Locates the top-left corner of a preprocessed slide image inside a
  /// preprocessed background image.
  ///
  /// This method performs matching only. It does not write the annotated
  /// output image.
  ///
  /// # Errors
  ///
  /// Returns:
  ///
  /// - [`PuzzlePatternError::TemplateLargerThanBackground`] if `slide_pic`
  ///   exceeds `background_pic` in either dimension
  /// - [`PuzzlePatternError::ZeroVarianceTemplate`] if `slide_pic` contains no
  ///   usable feature pixels
  pub fn locate_slide(
    &self,
    slide_pic: &GrayImage,
    background_pic: &GrayImage,
  ) -> Result<(u32, u32), PuzzlePatternError> {
    let (tpl_width, tpl_height) = slide_pic.dimensions();
    let (bg_width, bg_height) = background_pic.dimensions();

    if tpl_width > bg_width || tpl_height > bg_height {
      return Err(PuzzlePatternError::TemplateLargerThanBackground {
        template_width: tpl_width,
        template_height: tpl_height,
        background_width: bg_width,
        background_height: bg_height,
      });
    }

    find_best_match_location(slide_pic, background_pic)
  }

  /// Crops the gap image to the smallest rectangle that contains
  /// non-whitespace pixels.
  ///
  /// A pixel is treated as whitespace when all RGB channels have the same
  /// value. This mirrors the behavior of the original Python code.
  ///
  /// # Errors
  ///
  /// Returns:
  ///
  /// - [`PuzzlePatternError::Image`] if `image_path` cannot be decoded
  /// - [`PuzzlePatternError::NoPatternPixels`] if no non-whitespace pixel is
  ///   found in the scanned image area
  pub fn remove_whitespace(image_path: &Path) -> Result<RgbImage, PuzzlePatternError> {
    let image = load_rgb_image(image_path)?;
    let mut bounds: Option<(u32, u32, u32, u32)> = None;

    for y in 1..image.height() {
      for x in 1..image.width() {
        let [red, green, blue] = image.get_pixel(x, y).0;

        if red != green || green != blue {
          match bounds.as_mut() {
            Some((min_x, min_y, max_x, max_y)) => {
              *min_x = (*min_x).min(x);
              *min_y = (*min_y).min(y);
              *max_x = (*max_x).max(x);
              *max_y = (*max_y).max(y);
            }
            None => {
              bounds = Some((x, y, x, y));
            }
          }
        }
      }
    }

    let Some((min_x, min_y, max_x, max_y)) = bounds else {
      return Err(PuzzlePatternError::NoPatternPixels);
    };

    if max_x <= min_x || max_y <= min_y {
      return Err(PuzzlePatternError::NoPatternPixels);
    }

    let width = max_x - min_x;
    let height = max_y - min_y;

    Ok(crop_imm(&image, min_x, min_y, width, height).to_image())
  }

  /// Converts an RGB image to grayscale and extracts a Sobel-magnitude feature
  /// map.
  ///
  /// This preprocessing is intentionally simpler than full Canny edge
  /// detection. In practice it preserves the slider outline while being much
  /// cheaper to compute in pure Rust.
  #[must_use]
  pub fn apply_edge_detection(image: &RgbImage) -> GrayImage {
    apply_edge_detection_with_mode(image, FeatureMode::Sobel)
  }

  /// Loads both images and prepares the feature maps used for matching.
  ///
  /// # Errors
  ///
  /// Returns:
  ///
  /// - [`PuzzlePatternError::Image`] if either image cannot be read
  /// - [`PuzzlePatternError::NoPatternPixels`] if the gap image contains no
  ///   non-whitespace pixels
  pub fn prepare_feature_maps(&self) -> Result<(GrayImage, GrayImage), PuzzlePatternError> {
    let gap_image = Self::remove_whitespace(self.gap.as_path())?;
    let bg_image = load_rgb_image(self.background.as_path())?;
    let mode = select_feature_mode(&gap_image, &bg_image);
    let feature_gap = apply_edge_detection_with_mode(&gap_image, mode);
    let feature_bg = apply_edge_detection_with_mode(&bg_image, mode);

    Ok((feature_gap, feature_bg))
  }
}

impl PreparedTemplate {
  fn from_image(image: &GrayImage, background_width: u32) -> Result<Self, PuzzlePatternError> {
    let width = image.width();
    let height = image.height();
    let raw_pixels = image.as_raw();
    let mut runs = Vec::new();
    let mut values = Vec::new();
    let background_width =
      usize::try_from(background_width).expect("background width should fit usize");

    for row in 0..height {
      let row_start = usize::try_from(row).expect("row should fit usize")
        * usize::try_from(width).expect("width should fit usize");
      let background_row_offset =
        usize::try_from(row).expect("row should fit usize") * background_width;
      let mut run_start = None;

      for column in 0..width {
        let index = row_start + usize::try_from(column).expect("column should fit usize");
        let value = raw_pixels[index];

        if value == 0 {
          if let Some(start_column) = run_start.take() {
            let end_column = usize::try_from(column).expect("column should fit usize");

            push_template_run(
              &mut runs,
              &mut values,
              raw_pixels,
              row_start,
              background_row_offset,
              start_column,
              end_column,
            );
          }

          continue;
        }

        run_start.get_or_insert_with(|| usize::try_from(column).expect("column should fit usize"));
      }

      if let Some(start_column) = run_start.take() {
        push_template_run(
          &mut runs,
          &mut values,
          raw_pixels,
          row_start,
          background_row_offset,
          start_column,
          usize::try_from(width).expect("width should fit usize"),
        );
      }
    }

    if values.is_empty() {
      return Err(PuzzlePatternError::ZeroVarianceTemplate);
    }

    Ok(Self {
      width,
      height,
      runs,
      values,
    })
  }
}

impl PreparedBackground {
  fn from_image(image: &GrayImage) -> Self {
    let width = image.width();
    let height = image.height();
    let integral_width = usize::try_from(width + 1).expect("background width should fit usize");
    let integral_height = usize::try_from(height + 1).expect("background height should fit usize");
    let mut square_integral = vec![0u64; integral_width * integral_height];
    let pixels = image.as_raw().clone();

    for y in 0..height {
      let mut row_square_sum = 0u64;

      for x in 0..width {
        let pixel = u32::from(pixels[background_index(width, x, y)]);

        row_square_sum += u64::from(pixel) * u64::from(pixel);

        let current_index = integral_index(width, x + 1, y + 1);

        square_integral[current_index] =
          square_integral[integral_index(width, x + 1, y)] + row_square_sum;
      }
    }

    Self {
      width,
      pixels,
      square_integral,
    }
  }
}

fn load_rgb_image(image_path: &Path) -> Result<RgbImage, PuzzlePatternError> {
  let reader = ImageReader::open(image_path)
    .map_err(ImageError::IoError)?
    .with_guessed_format()
    .map_err(ImageError::from)?;

  Ok(reader.decode()?.into_rgb8())
}

fn gray_to_rgb(image: &GrayImage) -> RgbImage {
  RgbImage::from_fn(image.width(), image.height(), |x, y| {
    let value = image.get_pixel(x, y)[0];

    Rgb([value, value, value])
  })
}

fn draw_rectangle(image: &mut RgbImage, top_left: (u32, u32), width: u32, height: u32) {
  if width == 0 || height == 0 {
    return;
  }

  let max_x = top_left.0 + width - 1;
  let max_y = top_left.1 + height - 1;

  for x in top_left.0..=max_x {
    image.put_pixel(x, top_left.1, Rgb([255, 0, 0]));
    image.put_pixel(x, max_y, Rgb([255, 0, 0]));
  }

  for y in top_left.1..=max_y {
    image.put_pixel(top_left.0, y, Rgb([255, 0, 0]));
    image.put_pixel(max_x, y, Rgb([255, 0, 0]));
  }
}

fn select_feature_mode(gap: &RgbImage, background: &RgbImage) -> FeatureMode {
  if gap.height() * 8 >= background.height() * 3 {
    FeatureMode::Laplacian
  } else {
    FeatureMode::Sobel
  }
}

fn apply_edge_detection_with_mode(image: &RgbImage, mode: FeatureMode) -> GrayImage {
  let grayscale_image = rgb_to_grayscale(image);

  match mode {
    FeatureMode::Sobel => sobel_magnitude(&grayscale_image),
    FeatureMode::Laplacian => laplacian_abs(&grayscale_image),
  }
}

fn rgb_to_grayscale(image: &RgbImage) -> GrayImage {
  let width = usize::try_from(image.width()).expect("image width should fit usize");
  let height = usize::try_from(image.height()).expect("image height should fit usize");
  let pixels = image.as_raw();
  let mut output = vec![0u8; width * height];

  if width * height >= 32 * 1024 {
    output
      .par_chunks_mut(width)
      .enumerate()
      .for_each(|(row, output_row)| fill_grayscale_row(output_row, &pixels[row * width * 3..]));
  } else {
    for (row, output_row) in output.chunks_mut(width).enumerate() {
      fill_grayscale_row(output_row, &pixels[row * width * 3..]);
    }
  }

  GrayImage::from_raw(image.width(), image.height(), output).expect("grayscale image should fit")
}

fn sobel_magnitude(image: &GrayImage) -> GrayImage {
  let width = usize::try_from(image.width()).expect("image width should fit usize");
  let height = usize::try_from(image.height()).expect("image height should fit usize");
  let pixels = image.as_raw();
  let mut output = vec![0u8; width * height];

  if width < 3 || height < 3 {
    return GrayImage::from_raw(image.width(), image.height(), output)
      .expect("sobel image should fit");
  }

  let sqrt_table = sobel_sqrt_table();

  if width * height >= 32 * 1024 {
    output
      .par_chunks_mut(width)
      .enumerate()
      .for_each(|(row, output_row)| {
        fill_sobel_row(output_row, pixels, width, height, row, sqrt_table);
      });
  } else {
    for (row, output_row) in output.chunks_mut(width).enumerate() {
      fill_sobel_row(output_row, pixels, width, height, row, sqrt_table);
    }
  }

  GrayImage::from_raw(image.width(), image.height(), output).expect("sobel image should fit")
}

fn laplacian_abs(image: &GrayImage) -> GrayImage {
  let width = usize::try_from(image.width()).expect("image width should fit usize");
  let height = usize::try_from(image.height()).expect("image height should fit usize");
  let pixels = image.as_raw();
  let mut output = vec![0u8; width * height];

  if width < 3 || height < 3 {
    return GrayImage::from_raw(image.width(), image.height(), output)
      .expect("laplacian image should fit");
  }

  if width * height >= 32 * 1024 {
    output
      .par_chunks_mut(width)
      .enumerate()
      .for_each(|(row, output_row)| fill_laplacian_row(output_row, pixels, width, height, row));
  } else {
    for (row, output_row) in output.chunks_mut(width).enumerate() {
      fill_laplacian_row(output_row, pixels, width, height, row);
    }
  }

  GrayImage::from_raw(image.width(), image.height(), output).expect("laplacian image should fit")
}

fn find_best_match_location(
  template: &GrayImage,
  background: &GrayImage,
) -> Result<(u32, u32), PuzzlePatternError> {
  let best = find_match_candidates(template, background, 1)?
    .into_iter()
    .next()
    .expect("search should yield one candidate");

  Ok((best.x, best.y))
}

fn find_match_candidates(
  template: &GrayImage,
  background: &GrayImage,
  limit: usize,
) -> Result<Vec<MatchCandidate>, PuzzlePatternError> {
  if !should_use_pyramid(template, background) {
    return find_top_candidates(
      template,
      background,
      &[full_search_region(template, background)],
      limit,
    );
  }

  let levels = build_search_levels(template, background);
  let coarsest = levels.last().expect("pyramid should contain a level");
  let mut candidates = find_top_candidates(
    &coarsest.template,
    &coarsest.background,
    &[full_search_region(&coarsest.template, &coarsest.background)],
    1usize.max(limit),
  )?;

  for level_index in (0..levels.len() - 1).rev() {
    let level = &levels[level_index];
    let mut regions = project_search_regions(&candidates, &level.template, &level.background, 2);

    if regions_cover_most_positions(&regions, &level.template, &level.background) {
      regions.clear();
      regions.push(full_search_region(&level.template, &level.background));
    }

    candidates = find_top_candidates(
      &level.template,
      &level.background,
      &regions,
      1usize.max(limit),
    )?;
  }

  candidates.truncate(limit);

  Ok(candidates)
}

fn match_score(
  background: &PreparedBackground,
  template: &PreparedTemplate,
  offset_x: u32,
  offset_y: u32,
) -> MatchCandidate {
  let region_square_sum = region_sum_u64(
    &background.square_integral,
    background.width,
    offset_x,
    offset_y,
    template.width,
    template.height,
  );

  if region_square_sum == 0 {
    return MatchCandidate {
      numerator: 0,
      region_square_sum: 0,
      x: offset_x,
      y: offset_y,
    };
  }

  let mut numerator = 0u64;
  let base_offset = background_index(background.width, offset_x, offset_y);

  for run in &template.runs {
    let template_values = &template.values[run.value_offset..run.value_offset + run.len];
    let background_values = &background.pixels
      [base_offset + run.relative_offset..base_offset + run.relative_offset + run.len];

    numerator += dot_product_bytes(template_values, background_values);
  }

  MatchCandidate {
    numerator,
    region_square_sum,
    x: offset_x,
    y: offset_y,
  }
}

fn should_use_pyramid(template: &GrayImage, background: &GrayImage) -> bool {
  if template.width() < 12 || template.height() < 12 {
    return false;
  }

  let max_x = background.width() - template.width();
  let max_y = background.height() - template.height();
  let full_positions = u64::from(max_x + 1) * u64::from(max_y + 1);

  full_positions >= 10_000
}

fn build_search_levels(template: &GrayImage, background: &GrayImage) -> Vec<SearchLevel> {
  let mut levels = vec![SearchLevel {
    template: template.clone(),
    background: background.clone(),
  }];

  while levels.len() < 3 {
    let current = levels.last().expect("level list should not be empty");
    let next_template = downsample_feature_map(&current.template);
    let next_background = downsample_feature_map(&current.background);

    if next_template.width() < 12 || next_template.height() < 12 {
      break;
    }

    if next_background.width() <= next_template.width()
      || next_background.height() <= next_template.height()
    {
      break;
    }

    levels.push(SearchLevel {
      template: next_template,
      background: next_background,
    });
  }

  levels
}

fn downsample_feature_map(image: &GrayImage) -> GrayImage {
  let next_width = image.width() / 2;
  let next_height = image.height() / 2;

  if next_width == 0 || next_height == 0 {
    return image.clone();
  }

  GrayImage::from_fn(next_width, next_height, |x, y| {
    let source_x = x * 2;
    let source_y = y * 2;
    let a = image.get_pixel(source_x, source_y)[0];
    let b = image.get_pixel(source_x + 1, source_y)[0];
    let c = image.get_pixel(source_x, source_y + 1)[0];
    let d = image.get_pixel(source_x + 1, source_y + 1)[0];

    image::Luma([a.max(b).max(c).max(d)])
  })
}

fn find_top_candidates(
  template: &GrayImage,
  background: &GrayImage,
  regions: &[SearchRegion],
  limit: usize,
) -> Result<Vec<MatchCandidate>, PuzzlePatternError> {
  let prepared_template = PreparedTemplate::from_image(template, background.width())?;
  let prepared_background = PreparedBackground::from_image(background);
  let positions = total_region_positions(regions);
  let top = if positions < 8_192 {
    let mut best = Vec::new();

    for region in regions {
      search_region_candidates(
        &prepared_background,
        &prepared_template,
        region,
        limit,
        &mut best,
      );
    }

    best
  } else {
    regions
      .par_iter()
      .map(|region| {
        let mut best = Vec::new();

        search_region_candidates(
          &prepared_background,
          &prepared_template,
          region,
          limit,
          &mut best,
        );

        best
      })
      .reduce(Vec::new, |mut left, right| {
        for candidate in right {
          insert_candidate(&mut left, candidate, limit);
        }

        left
      })
  };

  Ok(top)
}

fn full_search_region(template: &GrayImage, background: &GrayImage) -> SearchRegion {
  SearchRegion {
    x_start: 0,
    x_end: background.width() - template.width(),
    y_start: 0,
    y_end: background.height() - template.height(),
  }
}

fn project_search_regions(
  candidates: &[MatchCandidate],
  template: &GrayImage,
  background: &GrayImage,
  margin: u32,
) -> Vec<SearchRegion> {
  let max_x = background.width() - template.width();
  let max_y = background.height() - template.height();
  let mut regions = Vec::with_capacity(candidates.len());

  for candidate in candidates {
    let projected_x = candidate.x.saturating_mul(2);
    let projected_y = candidate.y.saturating_mul(2);
    let x_start = projected_x.saturating_sub(margin);
    let y_start = projected_y.saturating_sub(margin);
    let x_end = projected_x.saturating_add(margin + 2).min(max_x);
    let y_end = projected_y.saturating_add(margin + 2).min(max_y);

    regions.push(SearchRegion {
      x_start,
      x_end,
      y_start,
      y_end,
    });
  }

  regions
}

fn regions_cover_most_positions(
  regions: &[SearchRegion],
  template: &GrayImage,
  background: &GrayImage,
) -> bool {
  let full_region = full_search_region(template, background);
  let full_positions = u64::from(full_region.x_end - full_region.x_start + 1)
    * u64::from(full_region.y_end - full_region.y_start + 1);
  let covered_positions = regions
    .iter()
    .map(|region| {
      u64::from(region.x_end - region.x_start + 1) * u64::from(region.y_end - region.y_start + 1)
    })
    .sum::<u64>();

  covered_positions * 2 >= full_positions
}

fn total_region_positions(regions: &[SearchRegion]) -> u64 {
  regions
    .iter()
    .map(|region| {
      u64::from(region.x_end - region.x_start + 1) * u64::from(region.y_end - region.y_start + 1)
    })
    .sum::<u64>()
}

fn insert_candidate(best: &mut Vec<MatchCandidate>, candidate: MatchCandidate, limit: usize) {
  if candidate.region_square_sum == 0 {
    return;
  }

  if let Some(index) = best
    .iter()
    .position(|existing| existing.x == candidate.x && existing.y == candidate.y)
  {
    if is_better_match(&candidate, &best[index]) {
      best[index] = candidate;
    } else {
      return;
    }
  } else {
    best.push(candidate);
  }

  best.sort_by(|left, right| {
    if is_better_match(left, right) {
      std::cmp::Ordering::Less
    } else if is_better_match(right, left) {
      std::cmp::Ordering::Greater
    } else {
      std::cmp::Ordering::Equal
    }
  });

  if best.len() > limit {
    best.truncate(limit);
  }
}

fn search_region_candidates(
  background: &PreparedBackground,
  template: &PreparedTemplate,
  region: &SearchRegion,
  limit: usize,
  best: &mut Vec<MatchCandidate>,
) {
  for offset_y in region.y_start..=region.y_end {
    for offset_x in region.x_start..=region.x_end {
      let candidate = match_score(background, template, offset_x, offset_y);

      insert_candidate(best, candidate, limit);
    }
  }
}

fn background_index(width: u32, x: u32, y: u32) -> usize {
  usize::try_from((y * width) + x).expect("background index should fit usize")
}

fn integral_index(width: u32, x: u32, y: u32) -> usize {
  usize::try_from(y * (width + 1) + x).expect("integral index should fit usize")
}

fn region_sum_u64(
  integral: &[u64],
  image_width: u32,
  offset_x: u32,
  offset_y: u32,
  width: u32,
  height: u32,
) -> u64 {
  let left = integral_index(image_width, offset_x, offset_y);
  let right = integral_index(image_width, offset_x + width, offset_y);
  let bottom_left = integral_index(image_width, offset_x, offset_y + height);
  let bottom_right = integral_index(image_width, offset_x + width, offset_y + height);
  let rectangle_sum = u128::from(integral[bottom_right]) + u128::from(integral[left])
    - u128::from(integral[right])
    - u128::from(integral[bottom_left]);

  u64::try_from(rectangle_sum).expect("rectangle sum should fit u64")
}

fn is_better_match(candidate: &MatchCandidate, current: &MatchCandidate) -> bool {
  if candidate.region_square_sum == 0 {
    return false;
  }

  if current.region_square_sum == 0 {
    return true;
  }

  let left = u128::from(candidate.numerator)
    * u128::from(candidate.numerator)
    * u128::from(current.region_square_sum);
  let right = u128::from(current.numerator)
    * u128::from(current.numerator)
    * u128::from(candidate.region_square_sum);

  if left > right {
    return true;
  }

  if left < right {
    return false;
  }

  candidate.y < current.y || (candidate.y == current.y && candidate.x < current.x)
}

fn best_match_point(candidates: &[MatchCandidate]) -> (u32, u32) {
  let best = &candidates[0];

  (best.x, best.y)
}

fn should_try_alternate_mode(candidates: &[MatchCandidate]) -> bool {
  !is_confident_match(candidates)
}

fn prefer_alternate_mode(primary: &[MatchCandidate], alternate: &[MatchCandidate]) -> bool {
  !is_confident_match(primary) && is_confident_match(alternate)
}

fn is_confident_match(candidates: &[MatchCandidate]) -> bool {
  if candidates.len() < 2 {
    return true;
  }

  let best = &candidates[0];
  let second = &candidates[1];

  if second.region_square_sum == 0 {
    return true;
  }

  let left = u128::from(best.numerator)
    * u128::from(best.numerator)
    * u128::from(second.region_square_sum)
    * 100;
  let right = u128::from(second.numerator)
    * u128::from(second.numerator)
    * u128::from(best.region_square_sum)
    * 101;

  left > right
}

fn integer_sqrt(value: u32) -> u32 {
  let mut lower = 0u32;
  let mut upper = value.min(65_535);
  let mut answer = 0u32;

  while lower <= upper {
    let middle = lower + ((upper - lower) / 2);
    let square = middle.saturating_mul(middle);

    if square <= value {
      answer = middle;
      lower = middle.saturating_add(1);
    } else {
      upper = middle.saturating_sub(1);
    }
  }

  answer
}

fn fill_grayscale_row(output_row: &mut [u8], input_row: &[u8]) {
  for (column, output) in output_row.iter_mut().enumerate() {
    let input_offset = column * 3;
    let value = ((u32::from(input_row[input_offset]) * 77)
      + (u32::from(input_row[input_offset + 1]) * 150)
      + (u32::from(input_row[input_offset + 2]) * 29)
      + 128)
      >> 8;

    *output = u8::try_from(value).expect("grayscale value should fit u8");
  }
}

fn fill_sobel_row(
  output_row: &mut [u8],
  pixels: &[u8],
  width: usize,
  height: usize,
  row: usize,
  sqrt_table: &[u8],
) {
  if row == 0 || row + 1 == height {
    output_row.fill(0);

    return;
  }

  output_row[0] = 0;
  output_row[width - 1] = 0;

  let row_above = (row - 1) * width;
  let row_current = row * width;
  let row_below = (row + 1) * width;

  for column in 1..width - 1 {
    let gradient_x = -i32::from(pixels[row_above + column - 1])
      + i32::from(pixels[row_above + column + 1])
      - 2 * i32::from(pixels[row_current + column - 1])
      + 2 * i32::from(pixels[row_current + column + 1])
      - i32::from(pixels[row_below + column - 1])
      + i32::from(pixels[row_below + column + 1]);
    let gradient_y = i32::from(pixels[row_above + column - 1])
      + 2 * i32::from(pixels[row_above + column])
      + i32::from(pixels[row_above + column + 1])
      - i32::from(pixels[row_below + column - 1])
      - 2 * i32::from(pixels[row_below + column])
      - i32::from(pixels[row_below + column + 1]);
    let squared_magnitude = u32::try_from(gradient_x * gradient_x + gradient_y * gradient_y)
      .expect("squared magnitude should fit u32");

    output_row[column] =
      sqrt_table[usize::try_from(squared_magnitude).expect("lookup index should fit usize")];
  }
}

fn fill_laplacian_row(
  output_row: &mut [u8],
  pixels: &[u8],
  width: usize,
  height: usize,
  row: usize,
) {
  if row == 0 || row + 1 == height {
    output_row.fill(0);

    return;
  }

  output_row[0] = 0;
  output_row[width - 1] = 0;

  let row_above = (row - 1) * width;
  let row_current = row * width;
  let row_below = (row + 1) * width;

  for column in 1..width - 1 {
    let value = -i32::from(pixels[row_above + column - 1])
      - i32::from(pixels[row_above + column])
      - i32::from(pixels[row_above + column + 1])
      - i32::from(pixels[row_current + column - 1])
      + 8 * i32::from(pixels[row_current + column])
      - i32::from(pixels[row_current + column + 1])
      - i32::from(pixels[row_below + column - 1])
      - i32::from(pixels[row_below + column])
      - i32::from(pixels[row_below + column + 1]);
    let magnitude = u32::try_from(value.abs())
      .expect("laplacian magnitude should fit")
      .min(255);
    let magnitude = u8::try_from(magnitude).expect("laplacian magnitude should fit u8");

    output_row[column] = if magnitude <= 16 { 0 } else { magnitude };
  }
}

fn sobel_sqrt_table() -> &'static [u8] {
  static TABLE: OnceLock<Box<[u8]>> = OnceLock::new();

  TABLE.get_or_init(|| {
    let mut table = vec![0u8; usize::try_from(1 + 1020 * 1020 * 2).expect("table size should fit")];

    for (index, entry) in table.iter_mut().enumerate() {
      let squared = u32::try_from(index).expect("table index should fit u32");
      let magnitude = integer_sqrt(squared).min(255);
      let magnitude = u8::try_from(magnitude).expect("magnitude should fit u8");

      *entry = if magnitude <= 16 { 0 } else { magnitude };
    }

    table.into_boxed_slice()
  })
}

fn push_template_run(
  runs: &mut Vec<TemplateRun>,
  values: &mut Vec<u8>,
  raw_pixels: &[u8],
  row_start: usize,
  background_row_offset: usize,
  start_column: usize,
  end_column: usize,
) {
  let value_offset = values.len();
  let len = end_column - start_column;

  values.extend_from_slice(&raw_pixels[row_start + start_column..row_start + end_column]);
  runs.push(TemplateRun {
    relative_offset: background_row_offset + start_column,
    value_offset,
    len,
  });
}

fn dot_product_bytes(left: &[u8], right: &[u8]) -> u64 {
  debug_assert_eq!(left.len(), right.len());

  left
    .iter()
    .zip(right.iter())
    .map(|(left, right)| u64::from(*left) * u64::from(*right))
    .sum::<u64>()
}
