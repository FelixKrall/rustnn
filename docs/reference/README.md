# WebNN Specification Reference

This directory contains cached copies of the W3C WebNN specification for offline reference.

## Files

- **webnn-index.bs** - Bikeshed source file from the official WebNN spec repository
  - Source: https://github.com/webmachinelearning/webnn/blob/main/index.bs
  - Last updated: 2025-12-20
  - This is the authoritative source that gets compiled to the official W3C spec at https://www.w3.org/TR/webnn/

## Usage

The `.bs` (Bikeshed) file is a markup format used by W3C specifications. While it's primarily meant to be compiled to HTML, it's human-readable and contains all operation definitions, signatures, and specifications.

To find operations:
```bash
grep "MLOperand \w\+(" webnn-index.bs
```

To extract operation list:
```bash
grep -E "MLOperand \w+\(" webnn-index.bs | sed -E 's/.*MLOperand ([a-zA-Z0-9]+)\(.*/\1/' | sort -u
```

## Updating

To update the cached spec:
```bash
curl -s https://raw.githubusercontent.com/webmachinelearning/webnn/main/index.bs -o docs/reference/webnn-index.bs
# Update the "Last updated" date in this README
```

## Cross-References

- Implementation status: [Generated backend operator support](https://rustnn.github.io/rustnn/development/backend-operator-support/)
- Online spec: https://www.w3.org/TR/webnn/
- Spec repository: https://github.com/webmachinelearning/webnn
