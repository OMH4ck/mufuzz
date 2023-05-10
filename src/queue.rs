pub mod frontend;
pub mod manager;
pub mod rpc;

use crate::datatype::TestCase;

pub trait QueueManager {
    // Check the feedback that tells which test cases are good for exploring the program.
    // The queue mananger should utilize this to fine tune their schduling algorithm.
    fn process_test_case_feedbacks(&mut self, feedbacks: &[crate::datatype::Feedback]);

    // Save the interesting test cases into the queue.
    fn receive_interesting_test_cases(&mut self, test_cases: Vec<TestCase>);

    // Select n test cases for mutation.
    fn select_interesting_inputs(&mut self, num: usize) -> Vec<TestCase>;
}
