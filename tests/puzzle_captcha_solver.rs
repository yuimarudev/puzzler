use image::{Rgb, RgbImage};
use puzzler::{PuzzleCaptchaSolver, PuzzlePatternError};
use std::fs;
use std::path::{Path, PathBuf};
use std::time::{SystemTime, UNIX_EPOCH};

#[test]
fn remove_whitespace_crops_the_pattern_region() {
  let test_dir = create_test_dir("remove_whitespace");
  let gap_path = test_dir.join("gap.png");
  let background_path = test_dir.join("background.png");
  let output_path = test_dir.join("result.png");

  create_gap_image()
    .save(&gap_path)
    .expect("gap image should be saved");
  create_background_image(30, 8)
    .save(&background_path)
    .expect("background image should be saved");

  let cropped = PuzzleCaptchaSolver::remove_whitespace(&gap_path).expect("cropping should succeed");

  assert_eq!(cropped.dimensions(), (11, 9));

  let solver = PuzzleCaptchaSolver::new(&gap_path, &background_path, &output_path);
  let position = solver.discern().expect("discern should succeed");

  assert_eq!(position, 31);
  assert!(output_path.exists());

  remove_test_dir(&test_dir);
}

#[test]
fn find_position_of_slide_rejects_a_template_larger_than_background() {
  let test_dir = create_test_dir("oversized_template");
  let output_path = test_dir.join("result.png");
  let solver = PuzzleCaptchaSolver::new("gap.png", "background.png", &output_path);
  let template = RgbImage::from_pixel(20, 20, Rgb([255, 0, 0]));
  let background = RgbImage::from_pixel(10, 10, Rgb([255, 0, 0]));
  let template_edges = PuzzleCaptchaSolver::apply_edge_detection(&template);
  let background_edges = PuzzleCaptchaSolver::apply_edge_detection(&background);
  let error = solver
    .find_position_of_slide(&template_edges, &background_edges)
    .expect_err("oversized template should fail");

  match error {
    PuzzlePatternError::TemplateLargerThanBackground {
      template_width,
      template_height,
      background_width,
      background_height,
    } => {
      assert_eq!((template_width, template_height), (20, 20));
      assert_eq!((background_width, background_height), (10, 10));
    }
    other => panic!("unexpected error: {other}"),
  }

  remove_test_dir(&test_dir);
}

#[test]
fn discern_writes_a_marked_output_image() {
  let test_dir = create_test_dir("discern_output");
  let gap_path = test_dir.join("gap.png");
  let background_path = test_dir.join("background.png");
  let output_path = test_dir.join("result.png");

  create_gap_image()
    .save(&gap_path)
    .expect("gap image should be saved");
  create_background_image(37, 9)
    .save(&background_path)
    .expect("background image should be saved");

  let solver = PuzzleCaptchaSolver::new(&gap_path, &background_path, &output_path);
  let position = solver.discern().expect("discern should succeed");
  let output = image::open(&output_path)
    .expect("result image should be readable")
    .into_rgb8();

  assert_eq!(position, 38);
  assert_eq!(output.get_pixel(38, 9), &Rgb([255, 0, 0]));

  remove_test_dir(&test_dir);
}

fn create_gap_image() -> RgbImage {
  let mut image = RgbImage::from_pixel(32, 28, Rgb([255, 255, 255]));

  draw_pattern(&mut image, 10, 8);

  image
}

fn create_background_image(pattern_x: u32, pattern_y: u32) -> RgbImage {
  let mut image = RgbImage::from_pixel(96, 40, Rgb([255, 255, 255]));

  draw_pattern(&mut image, pattern_x, pattern_y);

  image
}

fn draw_pattern(image: &mut RgbImage, origin_x: u32, origin_y: u32) {
  for y in origin_y..origin_y + 10 {
    for x in origin_x..origin_x + 4 {
      image.put_pixel(x, y, Rgb([255, 0, 0]));
    }
  }

  for y in origin_y..origin_y + 4 {
    for x in origin_x + 4..origin_x + 10 {
      image.put_pixel(x, y, Rgb([255, 0, 0]));
    }
  }

  for y in origin_y + 6..origin_y + 10 {
    for x in origin_x + 4..origin_x + 12 {
      image.put_pixel(x, y, Rgb([255, 0, 0]));
    }
  }
}

fn create_test_dir(label: &str) -> PathBuf {
  let unique = SystemTime::now()
    .duration_since(UNIX_EPOCH)
    .expect("system time should be after epoch")
    .as_nanos();
  let path = std::env::temp_dir().join(format!("puzzler-{label}-{unique}"));

  fs::create_dir_all(&path).expect("temporary test directory should be created");

  path
}

fn remove_test_dir(path: &Path) {
  fs::remove_dir_all(path).expect("temporary test directory should be removed");
}
