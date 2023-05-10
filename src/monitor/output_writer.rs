use super::stats::FuzzerInfo;
use crate::datatype::TestCase;
use std::fs;
use std::fs::File;
use std::io::prelude::*;
use std::sync::{Arc, Mutex};

/// Save the fuzzing process/statistics on disk
pub struct OutputWriter {
    root_dir: String,
    crash_dir: String,
    queue_dir: String,
    hang_dir: String,
    fuzzer_stat_path: String,
    plot_data_file: Arc<Mutex<File>>, // We need to write this file often, so keep a file handler.
}

impl OutputWriter {
    pub fn new(root_dir: String) -> std::io::Result<Self> {
        fs::create_dir_all(root_dir.clone())?;
        let crash_dir = format!("{}/crash", root_dir);
        let queue_dir = format!("{}/queue", root_dir);
        let hang_dir = format!("{}/hang", root_dir);
        let plot_data_path = format!("{}/plot_data", root_dir);
        let fuzzer_stat_path = format!("{}/fuzzer_stat", root_dir);
        fs::create_dir(crash_dir.clone())?;
        fs::create_dir(queue_dir.clone())?;
        fs::create_dir(hang_dir.clone())?;
        let plot_data_file = Arc::new(Mutex::new(File::create(plot_data_path)?));
        Ok(Self {
            crash_dir,
            queue_dir,
            hang_dir,
            root_dir,
            plot_data_file,
            fuzzer_stat_path,
        })
    }

    fn create_file_name_for_test_case(prefix: &str, test_case: &TestCase) -> String {
        format!(
            "{}/{}_{}",
            prefix,
            test_case.get_id(),
            test_case.get_mutator_id()
        )
    }

    fn save_test_case(file_name: String, test_case: &TestCase) -> std::io::Result<()> {
        let mut file = File::create(file_name)?;
        file.write_all(&test_case.get_buffer()[..])?;
        Ok(())
    }

    pub fn save_crash(&self, test_case: &TestCase) -> std::io::Result<()> {
        let file_name = Self::create_file_name_for_test_case(&self.crash_dir, test_case);
        Self::save_test_case(file_name, test_case)
    }

    pub fn save_hang(&self, test_case: &TestCase) -> std::io::Result<()> {
        let file_name = Self::create_file_name_for_test_case(&self.hang_dir, test_case);
        Self::save_test_case(file_name, test_case)
    }

    pub fn save_queue(&self, test_case: &TestCase) -> std::io::Result<()> {
        let file_name = Self::create_file_name_for_test_case(&self.queue_dir, test_case);
        Self::save_test_case(file_name, test_case)
    }

    pub fn write_plot_data(&self, fuzzer_stat: &FuzzerInfo) -> std::io::Result<()> {
        let data = format!(
            "{}, {}, {}, {}\n",
            fuzzer_stat.get_fuzzing_time(),
            fuzzer_stat.get_exec(),
            fuzzer_stat.get_timeout_exec(),
            fuzzer_stat.get_crash(),
        );
        self.plot_data_file
            .lock()
            .unwrap()
            .write_all(data.as_bytes())?;
        Ok(())
    }

    pub fn write_fuzzer_stat(&self, fuzzer_stat: &FuzzerInfo) -> std::io::Result<()> {
        let data = format!(
            "{}, {}, {}, {}\n",
            fuzzer_stat.get_fuzzing_time(),
            fuzzer_stat.get_exec(),
            fuzzer_stat.get_timeout_exec(),
            fuzzer_stat.get_crash(),
        );
        let mut file = fs::File::create(self.fuzzer_stat_path.clone())?;
        file.write_all(data.as_bytes())?;
        Ok(())
    }

    pub fn get_root_path(&self) -> String {
        self.root_dir.clone()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::path::Path;

    #[test]
    fn output_writer_create_dir_when_it_does_not_exist() {
        let output_dir = "./test_output1".to_string();
        let output_writer = OutputWriter::new(output_dir.clone());
        assert!(output_writer.is_ok());
        assert!(Path::new(&output_dir).is_dir());
        assert!(Path::new(&format!("{}/queue", output_dir)).is_dir());
        assert!(Path::new(&format!("{}/crash", output_dir)).is_dir());
        assert!(Path::new(&format!("{}/hang", output_dir)).is_dir());
        assert!(Path::new(&format!("{}/plot_data", output_dir)).is_file());

        drop(output_writer);
        assert!(fs::remove_dir_all(output_dir).is_ok());
    }

    #[test]
    fn output_writer_panick_when_the_dir_exist() {
        let output_dir = "./test_output2".to_string();
        let output_writer = OutputWriter::new(output_dir.clone());
        assert!(output_writer.is_ok());

        let output_writer = OutputWriter::new(output_dir.clone());
        assert!(output_writer.is_err());
        drop(output_writer);

        assert!(fs::remove_dir_all(output_dir).is_ok());
    }

    #[test]
    fn output_writer_save_file_in_disk() {
        let output_dir = "./test_output3".to_string();
        let output_writer = OutputWriter::new(output_dir.clone()).unwrap();

        let mut test_case = TestCase::new(vec![0xff; 100], 0);
        test_case.set_mutator_id(1);

        output_writer.save_crash(&test_case).unwrap();

        let file_name = OutputWriter::create_file_name_for_test_case(
            &format!("{}/crash", output_dir),
            &test_case,
        );

        assert!(Path::new(&file_name).is_file());

        let mut f = File::open(file_name).unwrap();
        let mut buffer = Vec::new();

        f.read_to_end(&mut buffer).unwrap();

        assert_eq!(buffer.len(), 100);

        assert_eq!(buffer.iter().filter(|&x| *x == 0xff).count(), 100);

        assert!(fs::remove_dir_all(output_dir).is_ok());
    }
}
