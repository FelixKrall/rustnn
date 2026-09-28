//! The public graph accessor reports its actual load, without changing prediction.
#![cfg(all(target_os = "macos", feature = "coreml-runtime"))]

use rustnn::mlcontext::{
    Backend, LoadDiagnostics, MLContext, MLContextOptions, MLGraphBuilder, MLNamedOperands,
    MLNamedTensors, MLOperandDescriptor, MLPowerPreference, MLTensorDescriptor,
};
use rustnn::operator_enums::MLOperandDataType;

#[test]
fn cpu_graph_retains_load_diagnostics_and_predicts_exact_values() {
    let mut context = MLContext::create(
        &MLContextOptions::new(MLPowerPreference::Default, false)
            .with_rustnn_backend_hint(Backend::Coreml),
    )
    .unwrap();
    let mut builder = MLGraphBuilder::new(&mut context).unwrap();
    let input = builder
        .input(
            "input",
            &MLOperandDescriptor::new(MLOperandDataType::Float32, vec![2]),
        )
        .unwrap();
    let output = builder.identity(input).unwrap();
    let mut graph = builder
        .build(&MLNamedOperands::from([("output", output)]))
        .unwrap();
    let Some(LoadDiagnostics::Coreml(diagnostic)) = graph.rustnn_load_diagnostics() else {
        panic!("CoreML graph did not report CoreML load diagnostics");
    };
    assert_eq!(diagnostic.requested_compute_units, "CPU_ONLY");
    assert_eq!(diagnostic.loaded_compute_units, "CPU_ONLY");
    assert!(diagnostic.failures.iter().all(|failure| {
        failure.compute_units.is_none() || failure.compute_units == Some("CPU_ONLY")
    }));
    let descriptor = MLTensorDescriptor::new(MLOperandDataType::Float32, vec![2])
        .to_readable()
        .to_writable();
    let input = context.create_tensor(&descriptor).unwrap();
    let output = context.create_tensor(&descriptor).unwrap();
    context.write_tensor(&input, &[2f32, -4.]).unwrap();
    context
        .dispatch(
            &mut graph,
            &MLNamedTensors::from([("input", &input)]),
            &MLNamedTensors::from([("output", &output)]),
        )
        .unwrap();
    let mut actual = [0f32; 2];
    context.read_tensor(&output, &mut actual).unwrap();
    assert_eq!(actual, [2., -4.]);
    assert_eq!(
        graph.rustnn_load_diagnostics(),
        Some(LoadDiagnostics::Coreml(diagnostic))
    );
}
