pub mod bitmap_collector;
pub mod frontend;
mod newbit_filter;
pub mod rpc;
pub use newbit_filter::NewBitFilter;

use crate::datatype::Feedback;

pub trait FeedbackCollector {
    fn process_feedback(&mut self, feedbacks: Vec<Feedback>);

    // Get test cases that should be saved in the queue manager.
    fn get_interesting_test_cases(&mut self, num: Option<usize>) -> Vec<Feedback>;
    // Get test cases that cause crashes.
    fn get_crash_test_cases(&mut self, num: Option<usize>) -> Vec<Feedback>;
    // Get feedback about the mutation quality.
    fn get_mutation_feedbacks(&mut self, num: Option<usize>) -> Vec<Feedback>;
    // Get feedback about whether mutating a test case produce good variants.
    fn get_test_case_feedbacks(&mut self, num: Option<usize>) -> Vec<Feedback>;
}
