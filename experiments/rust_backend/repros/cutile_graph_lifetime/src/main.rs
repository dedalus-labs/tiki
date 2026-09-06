// Copyright (c) 2026 Dedalus Labs, Inc. All rights reserved.

//! Exercise a scan, its reverse-mode rule, and captured-buffer ownership.

use anyhow::{ensure, Result};
use cutile::prelude::*;
use std::time::Instant;

#[cutile::module]
mod kernels {
    use cutile::core::*;

    #[cutile::entry()]
    fn forward<const N: i32>(out: &mut Tensor<f32, { [N] }>, input: &Tensor<f32, { [-1] }>) {
        let values = load_tile_like(input, out);
        out.store(scan_sum(values, 0i32, reverse::Forward, 0.0f32));
    }

    #[cutile::entry()]
    fn backward<const N: i32>(out: &mut Tensor<f32, { [N] }>, cotangent: &Tensor<f32, { [-1] }>) {
        let values = load_tile_like(cotangent, out);
        out.store(scan_sum(values, 0i32, reverse::Reverse, 0.0f32));
    }
}

fn main() -> Result<()> {
    let start = Instant::now();
    let stream = Device::new(0)?.new_stream()?;
    let input = api::arange::<f32>(128).sync_on(&stream)?;
    let mut output = api::zeros::<f32>(&[128]).sync_on(&stream)?;
    let mut gradient = api::zeros::<f32>(&[128]).sync_on(&stream)?;
    let first = Instant::now();
    kernels::forward((&mut output).partition([128]), &input).sync_on(&stream)?;
    let first_us = first.elapsed().as_micros();
    kernels::backward((&mut gradient).partition([128]), &input).sync_on(&stream)?;
    let backward = gradient.to_host_vec().sync_on(&stream)?;
    let expected_forward: Vec<f32> = (0..128).map(|i| (i * (i + 1) / 2) as f32).collect();
    for i in 0..128 {
        ensure!(
            backward[i] == (8128 - i * i.saturating_sub(1) / 2) as f32,
            "backward mismatch at {i}"
        );
    }
    println!(
        "{}",
        serde_json::json!({"stage":"scan", "first_forward_us":first_us,
        "process_setup_and_checks_us":start.elapsed().as_micros(), "correct":true})
    );

    let graph = CudaGraph::scope(&stream, |scope| {
        scope.record(kernels::forward((&mut output).partition([128]), &input))?;
        Ok(())
    })?;
    if std::env::args().any(|arg| arg == "--drop-input") {
        let input_pointer = input.device_pointer().cu_deviceptr();
        drop(input);
        let mut replacements = Vec::new();
        let replacement = loop {
            let candidate = api::full::<f32>(1000.0, &[128]).sync_on(&stream)?;
            if candidate.device_pointer().cu_deviceptr() == input_pointer {
                break candidate;
            }
            ensure!(
                replacements.len() < 4096,
                "allocator did not reuse captured input"
            );
            replacements.push(candidate);
        };
        graph.launch().sync_on(&stream)?;
        let actual = output.to_host_vec().sync_on(&stream)?;
        ensure!(
            actual == expected_forward,
            "replay changed after dropping input: first={}, last={}",
            actual[0],
            actual[127]
        );
        drop(replacement);
        println!("graph input-drop ownership: PASS");
    } else {
        let replay = Instant::now();
        for _ in 0..100 {
            graph.launch().sync_on(&stream)?;
        }
        ensure!(
            output.to_host_vec().sync_on(&stream)? == expected_forward,
            "replay output mismatch"
        );
        println!(
            "{}",
            serde_json::json!({"stage":"graph_replay",
            "warm_host_sync_us":replay.elapsed().as_secs_f64()*1e4})
        );
    }
    Ok(())
}
