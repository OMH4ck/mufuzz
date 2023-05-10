use crate::datatype::TestCaseID;
use crate::datatype::{Feedback, TestCase};
use serde_json::Value;

pub mod load_balance_frontend;
pub mod simple_frontend;
mod toy_frontend;
pub mod work_stealing_frontend;

// The basic unit controlled by the Frontend.
pub trait Worker: Send {
    type Input: Clone + Send + 'static;
    type Output: Send + 'static;

    // The main function for Worker. Accept some inputs and produce some results.
    fn handle_one_input(&mut self, input: Vec<Self::Input>) -> Vec<Self::Output>;

    // Other than producing some results, a worker might also produce some extra side data that
    // we want to keep. For example, the executor might tell how many test cases it has tried.
    fn retrieve_monitor_data(&mut self) -> Vec<Value> {
        unreachable!();
    }
}

macro_rules! io_type_marcro {
        ($($type_name:tt($data_type:ty)), *) => {
        #[derive(Clone, Debug)]
        pub enum FuzzerIO {
            $($type_name($data_type)), *
        }

        #[derive(Debug, Clone, Copy, std::hash::Hash, PartialEq, Eq, std::cmp::Ord, std::cmp::PartialOrd)]
        pub enum FuzzerIOType {
            $($type_name), *
        }

        impl FuzzerIO {
            pub fn get_type(&self) -> FuzzerIOType {
                match self{
                    $(FuzzerIO::$type_name(_) => FuzzerIOType::$type_name),*
                }
            }

            pub fn len(&self) -> usize {
                match self{
                    $(FuzzerIO::$type_name(data) => data.len()),*
                }

            }

            pub fn is_empty(&self) -> bool {
                match self{
                    $(FuzzerIO::$type_name(data) => data.is_empty()),*
                }

            }

            pub fn get_default(io_type: FuzzerIOType) -> FuzzerIO {
                match io_type {
                    $(FuzzerIOType::$type_name => FuzzerIO::$type_name(Vec::default())),*
                }
            }
        }

        };
    }

// For each element we will generate a enum variant in FuzzerIO and FuzzerIOType
// respectively. For example, we will have
// FuzzerIO::TestCase(_).get_type() = FuzzerIOType::TestCase
io_type_marcro! {
    TestCase(Vec<TestCase>),
    Feedback(Vec<Feedback>),
    MutationFeedback(Vec<Feedback>),
    TestCaseFeedback(Vec<Feedback>),
    MonitorData(Vec<Value>),
    TestCaseScoreChange(Vec<(TestCaseID, i64)>),
    MutatorScoreChange(Vec<(u32, i64)>)
}
