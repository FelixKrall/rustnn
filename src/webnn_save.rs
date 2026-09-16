/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 * http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */

//! Save a rustnn source graph to the `.webnn` text format plus a sibling `.safetensors`
//! weights file.
//!
//! This is a **rustnn-specific** extension, not part of the W3C WebNN API. Constants are written
//! as external `@weights(...)` references so the emitted `.webnn` file stays compact and the pair
//! round-trips through [`crate::load_graph_from_path`].
//!
//! Safetensors does not define native 4-bit dtypes. RustNN therefore stores Int4 and Uint4
//! constants as their existing low-nibble-first packed bytes in a one-dimensional U8 tensor and
//! adds the versioned `rustnn.webnn.packed4` archive metadata marker. The `.webnn` declaration
//! remains authoritative for the logical 4-bit dtype and shape.
//!
//! The exporter operates on a backend-agnostic [`GraphInfo`], so it is available from
//! [`crate::mlgraphbuilder::MLGraphBuilder`] regardless of the execution backend, as long as the
//! constants are held in host memory (which is the case for graphs authored via the builder API).

use std::collections::HashMap;
use std::path::Path;

use webnn_graph::ast::DataType as AstDataType;
use webnn_graph::serialize::{SerializeOptions, serialize_graph_to_wg_text};

use crate::error::{Error, GraphBuilderError, Result};
use crate::graph::{GraphInfo, OperandKind};
use crate::webnn_json::{
    ConstExport, OutputNameOverrides, operand_export_name, to_graph_json_with_consts,
};

pub(crate) const PACKED_4BIT_METADATA_KEY: &str = "rustnn.webnn.packed4";
pub(crate) const PACKED_4BIT_METADATA_VERSION: &str = "1";

fn ast_dtype_to_safetensors(dt: &AstDataType) -> Option<safetensors::Dtype> {
    use safetensors::Dtype as S;
    Some(match dt {
        AstDataType::Float32 => S::F32,
        AstDataType::Float16 => S::F16,
        AstDataType::Int32 => S::I32,
        AstDataType::Uint32 => S::U32,
        AstDataType::Int64 => S::I64,
        AstDataType::Uint64 => S::U64,
        AstDataType::Int8 => S::I8,
        AstDataType::Uint8 => S::U8,
        // Packed 4-bit tensors are handled separately by the versioned U8 extension below.
        AstDataType::Int4 | AstDataType::Uint4 => return None,
    })
}

fn save_err(msg: impl Into<String>) -> Error {
    Error::GraphSaveError {
        source: msg.into().into(),
    }
}

fn logical_element_count(name: &str, shape: &[u32]) -> Result<usize> {
    shape.iter().try_fold(1usize, |count, &dimension| {
        count
            .checked_mul(dimension as usize)
            .ok_or_else(|| save_err(format!("constant `{name}` element count overflows usize")))
    })
}

/// Reject output names that collide with an exported input or constant name.
///
/// The `.webnn` format references operands by global name strings in nodes, so an output cannot
/// share a name with an input or constant even though WebNN dispatch keeps inputs and outputs in
/// separate maps at runtime.
pub(crate) fn validate_save_output_names(
    graph: &GraphInfo,
    output_names: &OutputNameOverrides,
) -> std::result::Result<(), GraphBuilderError> {
    for (idx, operand) in graph.operands.iter().enumerate() {
        let (operand_kind, exported) = match operand.kind {
            OperandKind::Input => ("input", operand_export_name(operand, idx)),
            OperandKind::Constant => ("constant", operand_export_name(operand, idx)),
            _ => continue,
        };
        for out_name in output_names.values() {
            if out_name == &exported {
                return Err(GraphBuilderError::OutputNameConflictsWithOperand {
                    name: out_name.clone(),
                    operand_kind,
                });
            }
        }
    }
    Ok(())
}

/// Serialize `graph_info` to `webnn_path` (text) and its constants to `<stem>.safetensors`.
///
/// `output_names` maps each output operand id to the name it should be exported under. The exporter
/// uses it to render the graph's `outputs` section without mutating `graph_info`, so callers (e.g.
/// the builder) can save while keeping the graph intact.
///
/// Constant operand names mirror [`to_graph_json`] (`operand.name`, else `operand_<idx>`), which is
/// also the tensor key used in the safetensors archive and the `@weights(...)` reference.
pub(crate) fn write_webnn_and_safetensors(
    graph_info: &GraphInfo,
    output_names: &OutputNameOverrides,
    webnn_path: &Path,
) -> Result<()> {
    // `to_graph_json_with_consts` returns `webnn_graph::ast::GraphJson`, which is the typed AST
    // (not a JSON file on disk). The name is historical; see
    // https://github.com/rustnn/webnn-graph/issues/17
    // Build the AST with constants emitted as external `@weights(...)` references. This avoids
    // copying the constant bytes into the AST; we serialize them straight from `graph_info` below.
    // The outputs section is taken from `output_names` so we never mutate `graph_info`.
    let webnn_ast = to_graph_json_with_consts(
        graph_info,
        graph_info.quantized,
        ConstExport::ExternalWeights,
        Some(output_names),
    )
    .map_err(|e| Error::GraphSaveError { source: e.into() })?;

    // Map constant operand name -> raw bytes, using the same naming scheme as `to_graph_json`.
    let mut const_bytes: HashMap<String, &[u8]> = HashMap::new();
    for (idx, operand) in graph_info.operands.iter().enumerate() {
        if operand.kind != OperandKind::Constant {
            continue;
        }
        if let Some(constant) = graph_info
            .constant_operand_ids_to_handles
            .get(&(idx as u32))
        {
            const_bytes.insert(operand_export_name(operand, idx), constant.data.as_slice());
        }
    }

    // Build safetensors views by borrowing the bytes directly from `graph_info` (no copy). The AST
    // already references each constant via `@weights(<name>)`, so we only need matching tensors.
    let mut views: Vec<(String, safetensors::tensor::TensorView<'_>)> = Vec::new();
    let mut has_packed_4bit = false;
    for (name, decl) in webnn_ast.consts.iter() {
        let bytes = *const_bytes
            .get(name)
            .ok_or_else(|| save_err(format!("constant `{name}` has no data to save")))?;
        let (dtype, shape) = if matches!(decl.data_type, AstDataType::Int4 | AstDataType::Uint4) {
            let elements = logical_element_count(name, &decl.shape)?;
            let packed_len = elements.div_ceil(2);
            if bytes.len() != packed_len {
                return Err(save_err(format!(
                    "constant `{name}` packed 4-bit byte length {} does not match logical shape {:?} (expected {packed_len})",
                    bytes.len(),
                    decl.shape
                )));
            }
            has_packed_4bit = true;
            (safetensors::Dtype::U8, vec![packed_len])
        } else {
            let dtype = ast_dtype_to_safetensors(&decl.data_type).ok_or_else(|| {
                save_err(format!(
                    "constant `{name}` has data type {:?} which cannot be stored in safetensors",
                    decl.data_type
                ))
            })?;
            let shape = decl.shape.iter().map(|&d| d as usize).collect();
            (dtype, shape)
        };
        let view = safetensors::tensor::TensorView::new(dtype, shape, bytes)
            .map_err(|e| save_err(format!("constant `{name}`: {e}")))?;
        views.push((name.clone(), view));
    }

    let metadata = has_packed_4bit.then(|| {
        HashMap::from([(
            PACKED_4BIT_METADATA_KEY.to_string(),
            PACKED_4BIT_METADATA_VERSION.to_string(),
        )])
    });
    let st_bytes = safetensors::serialize(views, metadata)
        .map_err(|e| save_err(format!("failed to serialize safetensors: {e}")))?;

    let text = serialize_graph_to_wg_text(
        &webnn_ast,
        SerializeOptions {
            quantized: graph_info.quantized,
        },
    )
    .map_err(|e| Error::GraphSaveError { source: e.into() })?;

    std::fs::write(webnn_path, text)
        .map_err(|e| save_err(format!("failed to write `{}`: {e}", webnn_path.display())))?;

    let st_path = webnn_path.with_extension("safetensors");
    if let Err(e) = std::fs::write(&st_path, st_bytes) {
        let _ = std::fs::remove_file(webnn_path);
        return Err(save_err(format!(
            "failed to write `{}`: {e}",
            st_path.display()
        )));
    }

    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::graph::{ConstantData, DataType, Dimension, Operand, OperandDescriptor};
    use crate::operators::Operation;

    #[test]
    fn packed_4bit_element_count_overflow_is_explicit() {
        let error = logical_element_count("q", &[u32::MAX, u32::MAX, u32::MAX]).unwrap_err();
        assert!(error.to_string().contains("element count overflows usize"));
    }

    #[test]
    fn packed_4bit_export_rejects_wrong_byte_length() {
        let descriptor = OperandDescriptor {
            data_type: DataType::Uint4,
            shape: vec![Dimension::Static(3)],
            pending_permutation: Vec::new(),
        };
        let graph = GraphInfo {
            operands: vec![
                Operand {
                    kind: OperandKind::Constant,
                    descriptor: descriptor.clone(),
                    name: Some("q".to_string()),
                },
                Operand {
                    kind: OperandKind::Output,
                    descriptor,
                    name: Some("output".to_string()),
                },
            ],
            output_operands: vec![1],
            operations: vec![Operation::Identity {
                input: 0,
                options: None,
                outputs: vec![1],
            }],
            constant_operand_ids_to_handles: HashMap::from([(
                0,
                ConstantData {
                    data: vec![0x70],
                    label: None,
                },
            )]),
            ..GraphInfo::default()
        };
        let output_names = HashMap::from([(1, "output".to_string())]);
        let temp_dir = tempfile::tempdir().unwrap();
        let error =
            write_webnn_and_safetensors(&graph, &output_names, &temp_dir.path().join("bad.webnn"))
                .unwrap_err();

        assert!(error.to_string().contains("packed 4-bit byte length"));
        assert!(!temp_dir.path().join("bad.webnn").exists());
        assert!(!temp_dir.path().join("bad.safetensors").exists());
    }
}
