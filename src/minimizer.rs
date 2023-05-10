use crate::datatype::TestCase;
use crate::executor::shmem::ShMem;
use crate::executor::BitmapTracer;
use crate::executor::ExecutionStatus;
use crate::executor::Executor;
use crate::executor::ForkServerExecutor;
use std::collections::hash_map::DefaultHasher;
use std::hash::{Hash, Hasher};

pub struct TestCaseMinimizer {
    executor: ForkServerExecutor<BitmapTracer>,
}

impl TestCaseMinimizer {
    pub fn new(args: Vec<String>, map_size: usize, timeout: u64) -> Self {
        let bitmap_tracer = BitmapTracer::new(map_size);
        bitmap_tracer.bitmap.write_to_env("__AFL_SHM_ID").unwrap();

        let executor = ForkServerExecutor::new(bitmap_tracer, args, None, timeout, false)
            .ok()
            .unwrap();

        TestCaseMinimizer { executor }
    }

    fn get_tracer_bitmap_hash(&self) -> u64 {
        let mut s = DefaultHasher::new();
        self.executor.tracer.bitmap.map().hash(&mut s);
        s.finish()
    }

    fn next_p2(&self, val: usize) -> usize {
        let mut ret = 1;
        while val > ret {
            ret <<= 1;
        }
        ret
    }

    fn keep_same_state_and_bitmap(
        &mut self,
        testcase: &TestCase,
        save_exec_result: &ExecutionStatus,
        save_hash: u64,
    ) -> bool {
        let exec_result = self.executor.run_target(testcase);
        assert!(exec_result.is_ok());
        if exec_result.unwrap() != *save_exec_result {
            return false;
        }
        let exec_hash = self.get_tracer_bitmap_hash();
        self.executor.tracer.clear();
        exec_hash == save_hash
    }

    // Minimize the length of the test case while keeping the coverage same.
    pub fn minimize(&mut self, testcase: TestCase) -> TestCase {
        const TRIM_START_STEPS: usize = 2;
        const TRIM_MIN_BYTES: usize = 1;
        const TRIM_END_STEPS: usize = 1024;
        let save_exec_result = self.executor.run_target(&testcase);
        assert!(save_exec_result.is_ok());
        let save_exec_result = save_exec_result.unwrap();

        let save_hash = self.get_tracer_bitmap_hash();
        self.executor.tracer.clear();

        let mut result = testcase;
        let mut len_p2 = self.next_p2(result.len());
        let mut remove_len = (len_p2 / TRIM_START_STEPS).max(TRIM_MIN_BYTES);
        while remove_len >= (len_p2 / TRIM_END_STEPS).max(TRIM_MIN_BYTES) {
            let mut remove_pos: usize = remove_len;
            while remove_pos < result.len() {
                let trim_avail = remove_len.min(result.len() - remove_pos);
                let mut tmp_testcase = result.clone();
                tmp_testcase
                    .get_buffer_mut()
                    .drain(remove_pos..remove_pos + trim_avail);
                if self.keep_same_state_and_bitmap(&tmp_testcase, &save_exec_result, save_hash) {
                    result = tmp_testcase;
                    len_p2 = self.next_p2(result.len());
                } else {
                    remove_pos += remove_len;
                }
            }
            remove_len >>= 1;
        }
        result
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::util;
    use serial_test::serial;

    #[test]
    #[serial]
    fn test_case_minimizer_remove_redundant_bytes() {
        const MAP_SIZE: usize = 65536;
        let args = vec![util::get_test_bin_by_name("simple_prog").unwrap()];
        let mut minimizer = TestCaseMinimizer::new(args, MAP_SIZE, 20);

        let mut buffer = vec![b'f'; 10];
        buffer[0] = b'c';
        buffer[1] = b'a';
        let testcase = TestCase::new(buffer, 0);

        let minimized_testcase = minimizer.minimize(testcase);
        assert_eq!(minimized_testcase.len(), 10);
        //assert_eq!(*minimized_testcase.get_buffer(), vec![b'c', b'a']);

        let mut buffer = vec![b'f'; 1000];
        buffer[0] = b'c';
        buffer[1] = b'a';
        buffer[2] = b'b';
        let testcase = TestCase::new(buffer, 0);
        let minimized_testcase = minimizer.minimize(testcase);
        // Triggering EOF will make the edge different. SO we have a 'f' at the end.
        assert!(minimized_testcase.len() < 10);
        assert_eq!(minimized_testcase.get_buffer()[0..3], [b'c', b'a', b'b']);
    }
}
