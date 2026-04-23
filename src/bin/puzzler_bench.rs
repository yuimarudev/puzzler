use puzzler::PuzzleCaptchaSolver;
use std::env;
use std::process::ExitCode;
use std::time::Instant;

fn main() -> ExitCode {
  match run() {
    Ok(()) => ExitCode::SUCCESS,
    Err(message) => {
      eprintln!("{message}");
      ExitCode::FAILURE
    }
  }
}

fn run() -> Result<(), String> {
  let mut args = env::args().skip(1);
  let Some(gap) = args.next() else {
    return Err(
      "usage: puzzler_bench GAP_IMAGE BACKGROUND_IMAGE OUTPUT_IMAGE EXPECTED_X [ROUNDS] [LOCATE_ROUNDS]"
        .to_owned(),
    );
  };
  let Some(background) = args.next() else {
    return Err(
      "usage: puzzler_bench GAP_IMAGE BACKGROUND_IMAGE OUTPUT_IMAGE EXPECTED_X [ROUNDS] [LOCATE_ROUNDS]"
        .to_owned(),
    );
  };
  let Some(output) = args.next() else {
    return Err(
      "usage: puzzler_bench GAP_IMAGE BACKGROUND_IMAGE OUTPUT_IMAGE EXPECTED_X [ROUNDS] [LOCATE_ROUNDS]"
        .to_owned(),
    );
  };
  let Some(expected) = args.next() else {
    return Err(
      "usage: puzzler_bench GAP_IMAGE BACKGROUND_IMAGE OUTPUT_IMAGE EXPECTED_X [ROUNDS] [LOCATE_ROUNDS]"
        .to_owned(),
    );
  };
  let rounds = match args.next() {
    Some(value) => value
      .parse::<u32>()
      .map_err(|error| format!("invalid ROUNDS `{value}`: {error}"))?,
    None => 100,
  };
  let locate_rounds = match args.next() {
    Some(value) => value
      .parse::<u32>()
      .map_err(|error| format!("invalid LOCATE_ROUNDS `{value}`: {error}"))?,
    None => 500,
  };
  let expected = expected
    .parse::<u32>()
    .map_err(|error| format!("invalid EXPECTED_X `{expected}`: {error}"))?;
  let solver = PuzzleCaptchaSolver::new(&gap, &background, &output);

  benchmark_discern(&solver, expected, rounds)?;
  benchmark_locate(&solver, expected, locate_rounds)?;

  Ok(())
}

fn benchmark_discern(
  solver: &PuzzleCaptchaSolver,
  expected: u32,
  rounds: u32,
) -> Result<(), String> {
  let warmup_start = Instant::now();
  let warmup_position = solver
    .discern()
    .map_err(|error| format!("discern warmup failed: {error}"))?;
  let warmup_elapsed = warmup_start.elapsed();
  let start = Instant::now();
  let mut position = warmup_position;

  if warmup_position != expected {
    return Err(format!(
      "discern warmup mismatch: expected {expected}, got {warmup_position}"
    ));
  }

  for _ in 0..rounds {
    position = solver
      .discern()
      .map_err(|error| format!("discern failed: {error}"))?;
  }

  if position != expected {
    return Err(format!(
      "discern mismatch: expected {expected}, got {position}"
    ));
  }

  let elapsed = start.elapsed();
  let average = elapsed / rounds;

  println!("discern_position={position}");
  println!(
    "discern_warmup_ms={:.3}",
    warmup_elapsed.as_secs_f64() * 1_000.0
  );
  println!("discern_total_ms={:.3}", elapsed.as_secs_f64() * 1_000.0);
  println!("discern_avg_ms={:.3}", average.as_secs_f64() * 1_000.0);

  Ok(())
}

fn benchmark_locate(
  solver: &PuzzleCaptchaSolver,
  expected: u32,
  rounds: u32,
) -> Result<(), String> {
  let (gap_edges, background_edges) = solver
    .prepare_feature_maps()
    .map_err(|error| format!("feature preparation failed: {error}"))?;
  let warmup_start = Instant::now();
  let warmup_position = solver
    .locate_slide(&gap_edges, &background_edges)
    .map_err(|error| format!("locate warmup failed: {error}"))?
    .0;
  let warmup_elapsed = warmup_start.elapsed();
  let start = Instant::now();
  let mut position = warmup_position;

  if warmup_position != expected {
    println!("locate_position={warmup_position}");
    println!(
      "locate_warmup_ms={:.3}",
      warmup_elapsed.as_secs_f64() * 1_000.0
    );
    println!("locate_skipped=true");

    return Ok(());
  }

  for _ in 0..rounds {
    position = solver
      .locate_slide(&gap_edges, &background_edges)
      .map_err(|error| format!("locate failed: {error}"))?
      .0;
  }

  if position != expected {
    return Err(format!(
      "locate mismatch: expected {expected}, got {position}"
    ));
  }

  let elapsed = start.elapsed();
  let average = elapsed / rounds;

  println!("locate_position={position}");
  println!(
    "locate_warmup_ms={:.3}",
    warmup_elapsed.as_secs_f64() * 1_000.0
  );
  println!("locate_total_ms={:.3}", elapsed.as_secs_f64() * 1_000.0);
  println!("locate_avg_ms={:.3}", average.as_secs_f64() * 1_000.0);

  Ok(())
}
