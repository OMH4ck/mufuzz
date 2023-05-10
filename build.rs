use std::fs;

fn main() -> Result<(), Box<dyn std::error::Error>> {
    // Build all proto in proto/.
    let paths = fs::read_dir("./proto").unwrap();
    for path in paths {
        let path = path.unwrap().path();
        let path = path.to_str().unwrap();

        if path.ends_with("proto") {
            tonic_build::compile_protos(path)?;
        }
    }
    Ok(())
}
