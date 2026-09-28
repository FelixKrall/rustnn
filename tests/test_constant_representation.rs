use rustnn::Operation;
use serde_json::json;

#[test]
fn constants_are_not_parsed_as_operations() {
    assert!(
        Operation::from_json_attributes(
            "constant",
            &[],
            &[0],
            &json!({ "dataType": "float32", "shape": [] }),
        )
        .is_none()
    );
}
