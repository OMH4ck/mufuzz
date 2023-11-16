use crate::mutator::afl_mutator::DeterministicMutationPass;
use serde::{Deserialize, Serialize};
use std::sync::atomic::{AtomicU32, Ordering};

pub type TestCaseID = u32;

#[derive(Default, Clone, Debug, Serialize, Deserialize)]
pub struct TestCaseMetaInfo {
    #[allow(dead_code)]
    done_pass: DeterministicMutationPass, // All the finished determinsitic passes.
    todo_pass: DeterministicMutationPass, // The determinsitic passes that we want to do next.
}

impl TestCaseMetaInfo {
    pub fn get_todo_pass(&self) -> DeterministicMutationPass {
        self.todo_pass
    }

    pub fn set_todo_pass(&mut self, todo_pass: DeterministicMutationPass) {
        self.todo_pass = todo_pass;
    }

    pub fn get_done_pass(&self) -> DeterministicMutationPass {
        self.done_pass
    }

    pub fn set_done_pass(&mut self, done_pass: DeterministicMutationPass) {
        self.done_pass = done_pass;
    }
}

#[derive(Default, Clone, Debug, Serialize, Deserialize)]
pub struct TestCase {
    id: TestCaseID, // The id of the test_case itself or the id of parent. Since we won't use both of them at the same time, we reuse the same slot.
    mutator_id: u32,
    buffer: Vec<u8>,
    meta: Option<TestCaseMetaInfo>,
}

impl TestCase {
    pub fn new(buffer: Vec<u8>, id: u32) -> Self {
        TestCase {
            id,
            buffer,
            mutator_id: 0,
            meta: None,
        }
    }

    pub fn clone_without_meta(&self) -> Self {
        let mut result = self.clone();
        result.take_meta();
        result
    }

    pub fn set_meta(&mut self, meta: TestCaseMetaInfo) {
        self.meta = Some(meta);
    }

    pub fn borrow_meta(&self) -> Option<&TestCaseMetaInfo> {
        self.meta.as_ref()
    }

    pub fn take_meta(&mut self) -> Option<TestCaseMetaInfo> {
        self.meta.take()
    }

    pub fn borrow_meta_mut(&mut self) -> Option<&mut TestCaseMetaInfo> {
        self.meta.as_mut()
    }

    pub fn has_meta(&self) -> bool {
        self.meta.is_some()
    }

    pub fn len(&self) -> usize {
        self.buffer.len()
    }

    pub fn is_empty(&self) -> bool {
        self.buffer.is_empty()
    }

    pub fn extract_mutation_info(&self) -> MutationInfo {
        MutationInfo::new(self.id, self.mutator_id)
    }

    pub fn set_mutator_id(&mut self, mid: u32) {
        self.mutator_id = mid;
    }

    pub fn get_mutator_id(&self) -> u32 {
        self.mutator_id
    }

    pub fn get_buffer(&self) -> &Vec<u8> {
        &self.buffer
    }

    pub fn get_buffer_mut(&mut self) -> &mut Vec<u8> {
        &mut self.buffer
    }

    pub fn take_buffer(self) -> Vec<u8> {
        self.buffer
    }

    pub fn get_id(&self) -> u32 {
        self.id
    }

    pub fn set_id(&mut self, id: TestCaseID) {
        self.id = id;
    }

    pub fn get_pid(&self) -> u32 {
        self.id
    }

    pub fn create_variant(&self, buffer: Vec<u8>) -> TestCase {
        Self::new(buffer, self.id)
    }

    pub fn gen_id(&mut self) {
        self.id = get_id();
    }
}

static COUNTER: AtomicU32 = AtomicU32::new(0);
pub fn get_id() -> u32 {
    COUNTER.fetch_add(1, Ordering::Relaxed)
}

#[cfg(test)]
pub fn reset_id() {
    COUNTER.store(0, Ordering::Relaxed)
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum ExecutionStatus {
    Ok,          // Nothing special happens
    Interesting, // Find new coverage or something that worth notice
    Crash,
    Timeout,
}

#[derive(Default, Clone, Hash, PartialEq, Eq, Copy, Debug)]
pub struct MutationInfo {
    pid: u32,
    mutation_id: u32,
}

impl MutationInfo {
    pub fn new(pid: u32, mutation_id: u32) -> Self {
        Self { pid, mutation_id }
    }

    pub fn get_pid(&self) -> u32 {
        self.pid
    }

    pub fn get_mutation_id(&self) -> u32 {
        self.mutation_id
    }

    pub fn get_mutator_id2(&self) -> u32 {
        self.mutation_id >> 16
    }

    pub fn get_mutator_func_id(&self) -> u32 {
        self.mutation_id & 0xFFFF
    }
}

#[derive(Default, Clone, Hash, PartialEq, Eq, Copy, Debug)]
pub struct NewBit {
    pub index: usize,
    pub val: u8,
}

impl NewBit {
    pub fn new(index: usize, val: u8) -> Self {
        Self { index, val }
    }
}

#[derive(Clone, Debug)]
pub enum FeedbackData {
    Unknown,
    NewCoverage(Vec<NewBit>),
    Counter(u32), // Indicate how many times such feedback is repeated.
}

impl FeedbackData {
    pub fn new_coverage(new_bits: Vec<NewBit>) -> Self {
        Self::NewCoverage(new_bits)
    }

    pub fn into_new_coverage(self) -> Option<Vec<NewBit>> {
        match self {
            Self::NewCoverage(coverage) => Some(coverage),
            _ => None,
        }
    }
}

#[derive(Clone, Debug)]
pub struct Feedback {
    status: ExecutionStatus,
    test_case: Option<TestCase>, // The test case that produce this feedback
    data: Option<FeedbackData>,  // The real feedback data
    mutation_info: Option<MutationInfo>,
}

impl Feedback {
    pub fn new(status: ExecutionStatus) -> Self {
        Self {
            status,
            test_case: None,
            data: None,
            mutation_info: None,
        }
    }

    pub fn get_status(&self) -> ExecutionStatus {
        self.status
    }

    pub fn get_mutation_info(&self) -> Option<&MutationInfo> {
        self.mutation_info.as_ref()
    }

    pub fn set_mutation_info(mut self, mutation_info: MutationInfo) -> Self {
        self.mutation_info = Some(mutation_info);
        self
    }

    pub fn is_valid(&self) -> bool {
        matches!(self.status, ExecutionStatus::Ok) && self.test_case.is_none()
            || (matches!(self.status, ExecutionStatus::Interesting)
                && (self.test_case.is_some() || self.data.is_some()))
            || (matches!(self.status, ExecutionStatus::Crash) && self.test_case.is_some())
            || (matches!(self.status, ExecutionStatus::Timeout) && self.test_case.is_none())
    }

    /*
    pub fn get_test_case(&self) -> Option<&TestCase> {
        self.test_case.as_ref()
    }
    */
    pub fn take_mutation_info(&mut self) -> Option<MutationInfo> {
        self.mutation_info.take()
    }

    pub fn take_test_case(&mut self) -> Option<TestCase> {
        self.test_case.take()
    }

    pub fn borrow_test_case(&self) -> Option<&TestCase> {
        self.test_case.as_ref()
    }

    pub fn contain_test_case(&self) -> bool {
        self.test_case.is_some()
    }

    pub fn take_data(&mut self) -> Option<FeedbackData> {
        self.data.take()
    }

    pub fn borrow_data(&self) -> Option<&FeedbackData> {
        self.data.as_ref()
    }

    pub fn set_data(mut self, data: FeedbackData) -> Self {
        self.data = Some(data);
        self
    }

    pub fn set_test_case(mut self, test_case: TestCase) -> Self {
        if self.mutation_info.is_none() {
            self.mutation_info = Some(test_case.extract_mutation_info());
        }
        self.test_case = Some(test_case);
        self
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn test_construct_test_case() {
        let test_case = TestCase::new(Vec::default(), 0);
        let test_case2 = test_case.create_variant(Vec::default());
        assert_eq!(test_case.get_id(), test_case2.get_pid());
    }
}
