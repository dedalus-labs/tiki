// Copyright © 2026 Dedalus Labs, Inc.

//! Replays every integer layout-algebra call that PyCuTe's own test suite makes.
//!
//! `reference/cases.txt` holds one call per line with PyCuTe's result, recorded by
//! `reference/record.py`. tiki-cute must return the same layout, structurally equal, and must
//! refuse exactly where PyCuTe raises. Replaying the reference's own cases catches a divergence
//! in any branch those tests reach, including the normal form of each result.

use tiki_cute::{Int, Layout, LayoutError, Shape, Tiler, Tuple};

const CASES: &str = include_str!("reference/cases.txt");
/// PyCuTe accepts `None` as a whole tiler and returns the layout unchanged. tiki-cute's tilers
/// hole only individual modes: a caller that tiles nothing does not call the operation.
const WHOLE_HOLE: &str = "_";
/// Mismatches printed before the test fails, enough to see a pattern.
const SHOWN: usize = 40;

fn layout(text: &str) -> Result<Layout, LayoutError> {
    text.parse()
}

/// Runs one recorded operation. An argument tiki-cute's types reject counts as a refusal.
fn run(operation: &str, args: &[&str]) -> Result<Layout, LayoutError> {
    let a = || layout(args[0]);
    match operation {
        "coalesce" => Ok(a()?.coalesce()),
        "coalesce_z" => Ok(a()?.coalesce_z()),
        "composition" => a()?.compose(&args[1].parse::<Tiler>()?),
        "complement" => a()?.complement(),
        "complement_in" => a()?.complement_in(&args[1].parse::<Shape>()?),
        "right_inverse" => Ok(a()?.right_inverse()),
        "left_inverse" => a()?.left_inverse(),
        "nullspace" => Ok(a()?.nullspace()),
        "logical_product" => a()?.logical_product(&args[1].parse::<Tiler>()?),
        "logical_divide" => a()?.logical_divide(&args[1].parse::<Tiler>()?),
        "zipped_divide" => a()?.zipped_divide(&args[1].parse::<Tiler>()?),
        "blocked_product" => a()?.blocked_product(&layout(args[1])?),
        "raked_product" => a()?.raked_product(&layout(args[1])?),
        "recast" => a()?.recast(&args[1].parse::<Tuple<Int>>()?),
        "make_layout_like" => Ok(a()?.compact_like()),
        "make_ordered_layout" => {
            Layout::from_order(args[0].parse::<Shape>()?, &args[1].parse::<Tuple<Int>>()?)
        }
        "layout_add" => a()?.add(&layout(args[1])?),
        _ => panic!("reference case names unknown operation {operation}"),
    }
}

// Every recorded call against its recorded outcome.
#[test]
fn every_reference_case_matches_pycute() {
    let mut mismatches = Vec::new();
    let mut count = 0;
    for line in CASES.lines().filter(|line| !line.starts_with('#')) {
        count += 1;
        let fields: Vec<&str> = line.split('\t').collect();
        let (operation, rest) = fields.split_first().expect("a case names its operation");
        let (outcome, args) = rest.split_last().expect("a case has an outcome");
        if args.get(1) == Some(&WHOLE_HOLE) {
            continue;
        }
        let got = std::panic::catch_unwind(|| run(operation, args));
        let Ok(got) = got else {
            mismatches.push(format!("{operation}({}) panicked", args.join(", ")));
            continue;
        };
        let matches = match (outcome.strip_prefix("=> "), &got) {
            (Some(expected), Ok(got)) => layout(expected).as_ref() == Ok(got),
            (None, Err(_)) => true,
            _ => false,
        };
        if !matches {
            let got = got.map(|l| l.cute().to_string());
            mismatches
                .push(format!("{operation}({}) expected {outcome}, got {got:?}", args.join(", ")));
        }
    }
    let shown: Vec<&String> = mismatches.iter().take(SHOWN).collect();
    assert!(
        mismatches.is_empty(),
        "{} of {count} reference cases differ:\n{}",
        mismatches.len(),
        shown.iter().map(|m| m.as_str()).collect::<Vec<_>>().join("\n")
    );
}
