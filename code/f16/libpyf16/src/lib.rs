use pyo3::prelude::*;
use pyo3::wrap_pyfunction;
use f16::test as f16_test;

#[pyfunction]
/// Formats the sum of two numbers as string
fn sum_as_string(a: usize, b: usize) -> PyResult<String> {
    Ok((a + b).to_string())
}

#[pyfunction]
fn test() -> PyResult<i32> {
    f16_test();
    Ok(42)
}

/// This module is a python module implemented in Rust.
#[pymodule]
fn libpyf16(_py: Python, m: &PyModule) -> PyResult<()> {
    m.add_wrapped(wrap_pyfunction!(sum_as_string))?;
    m.add_wrapped(wrap_pyfunction!(test))?;
    Ok(())
}
