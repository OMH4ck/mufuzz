use crate::datatype;
use crate::datatype::ExecutionStatus;
use crate::datatype::{Feedback, FeedbackData, MutationInfo, NewBit};
use crate::feedback::FeedbackCollector;
use crate::feedback::NewBitFilter;
use crate::util;
use serde_json::{json, Value};
use std::collections::HashMap;

//pub type BitmapFeedbackWithTestCase = crate::datatype::Feedback<Vec<(usize, u8)>>;

/*
#[derive(Clone)]
pub struct BitmapFeedbackWithTestCase {
    pub id: u32,
    pub new_bits: Vec<(usize, u8)>,
    pub test_case: TestCase,
}
*/

impl Feedback {
    pub fn create_coverage_feedback(
        status: datatype::ExecutionStatus,
        test_case: Option<datatype::TestCase>,
        data: Option<Vec<NewBit>>,
    ) -> Self {
        let mut result = Self::new(status);
        if let Some(test_case) = test_case {
            result = result.set_test_case(test_case);
        }
        if let Some(data) = data {
            result = result.set_data(FeedbackData::new_coverage(data));
        }
        result
    }
}

pub struct BitmapCollector {
    filter: NewBitFilter,
    interesting_test_cases: Vec<Feedback>,
    crash_test_cases: Vec<Feedback>,
    mutation_feedbacks: Vec<Feedback>,
    test_case_feedbacks: Vec<Feedback>,
    monitor_data: Vec<Value>,
}

impl BitmapCollector {
    pub fn new(n: usize) -> Self {
        BitmapCollector {
            filter: NewBitFilter::new(n), //interesing_inputs: Vec::default(),
            interesting_test_cases: Vec::default(),
            crash_test_cases: Vec::default(),
            mutation_feedbacks: Vec::default(),
            test_case_feedbacks: Vec::default(),
            monitor_data: Vec::default(),
        }
    }

    #[cfg(test)]
    fn visited_bytes_num(&self) -> u32 {
        self.filter.visited_bits_num()
    }

    pub fn retrieve_monitor_data(&mut self) -> Vec<Value> {
        self.monitor_data.split_off(0)
    }
}

impl FeedbackCollector for BitmapCollector {
    fn process_feedback(&mut self, feedbacks: Vec<Feedback>) {
        let mut uninteresting_counter: HashMap<datatype::MutationInfo, u32> = HashMap::new();
        let mut interesting_counter: HashMap<datatype::MutationInfo, u32> = HashMap::new();
        let mut timeout_counter: HashMap<datatype::MutationInfo, u32> = HashMap::new();
        let mut crash_counter: HashMap<datatype::MutationInfo, u32> = HashMap::new();
        let mut interesting_test_cases = Vec::default();
        let mut crash_test_cases = Vec::default();
        let mut crashes_vec = Vec::new();

        let mut total_timeout: u64 = 0;
        let mut total_crash: u64 = 0;

        for mut feedback in feedbacks.into_iter() {
            assert!(feedback.is_valid());
            let mutation_info = *feedback.get_mutation_info().unwrap();
            match feedback.get_status() {
                ExecutionStatus::Ok => {
                    *uninteresting_counter.entry(mutation_info).or_insert(1) += 1;
                }
                ExecutionStatus::Interesting => {
                    let coverage = feedback.take_data().unwrap().into_new_coverage().unwrap();
                    if let Some(new_bits) = self.filter.filter_old_bits_mut(coverage) {
                        assert!(feedback.contain_test_case());
                        interesting_test_cases
                            .push(feedback.set_data(FeedbackData::NewCoverage(new_bits)));

                        *interesting_counter.entry(mutation_info).or_insert(1) += 1;
                    } else {
                        // This interesting test case can be coverage by some others. So we skip it.
                        *uninteresting_counter.entry(mutation_info).or_insert(1) += 1;
                    }
                }
                ExecutionStatus::Crash => {
                    assert!(feedback.contain_test_case());
                    let test_case_str =
                        serde_json::to_string(feedback.borrow_test_case().unwrap()).unwrap();
                    crashes_vec.push(test_case_str);
                    crash_test_cases.push(feedback);
                    *crash_counter.entry(mutation_info).or_insert(1) += 1;
                    total_crash += 1;
                }
                ExecutionStatus::Timeout => {
                    *timeout_counter.entry(mutation_info).or_insert(1) += 1;
                    total_timeout += 1;
                }
            };
        }

        // generate monitor data if any
        if total_crash != 0 {
            self.monitor_data
                .push(json!({ "crash": total_crash, "testcases": crashes_vec }));
        }
        if total_timeout != 0 {
            self.monitor_data.push(json!({ "timeout": total_timeout }));
        }

        // Further separate them based on mutation info and parent id.
        for (status, counter) in [
            (ExecutionStatus::Ok, uninteresting_counter),
            (ExecutionStatus::Interesting, interesting_counter),
            (ExecutionStatus::Timeout, timeout_counter),
            (ExecutionStatus::Crash, crash_counter),
        ] {
            let mut mutation_counter = HashMap::new();
            let mut test_case_counter = HashMap::new();
            for (mutation_info, count) in counter.into_iter() {
                if count == 0 {
                    continue;
                }
                *mutation_counter
                    .entry(mutation_info.get_mutation_id())
                    .or_insert(count) += count;
                *test_case_counter
                    .entry(mutation_info.get_pid())
                    .or_insert(count) += count;
            }

            self.mutation_feedbacks
                .extend(mutation_counter.into_iter().map(|(k, v)| {
                    Feedback::new(status)
                        .set_mutation_info(MutationInfo::new(0, k))
                        .set_data(FeedbackData::Counter(v))
                }));
            self.test_case_feedbacks
                .extend(test_case_counter.into_iter().map(|(k, v)| {
                    Feedback::new(status)
                        .set_mutation_info(MutationInfo::new(k, 0))
                        .set_data(FeedbackData::Counter(v))
                }));
        }

        self.interesting_test_cases.extend(interesting_test_cases);
        self.crash_test_cases.extend(crash_test_cases);
    }

    fn get_interesting_test_cases(&mut self, num: Option<usize>) -> Vec<Feedback> {
        util::take_n_elements_from_vector(&mut self.interesting_test_cases, num)
    }

    fn get_crash_test_cases(&mut self, num: Option<usize>) -> Vec<Feedback> {
        util::take_n_elements_from_vector(&mut self.crash_test_cases, num)
    }

    fn get_mutation_feedbacks(&mut self, num: Option<usize>) -> Vec<Feedback> {
        util::take_n_elements_from_vector(&mut self.mutation_feedbacks, num)
    }

    fn get_test_case_feedbacks(&mut self, num: Option<usize>) -> Vec<Feedback> {
        util::take_n_elements_from_vector(&mut self.test_case_feedbacks, num)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::datatype::ExecutionStatus;
    use crate::datatype::TestCase;

    #[test]
    fn virgin_map_tract_new_bit() {
        let mut feedback_collector = BitmapCollector::new(0x100);
        let mut feedbacks = Vec::new();
        for i in 0..0x10usize {
            feedbacks.push(
                Feedback::create_coverage_feedback(
                    ExecutionStatus::Interesting,
                    Some(TestCase::new(vec![i as u8; 1], 0)),
                    Some(vec![NewBit::new(i, 1), NewBit::new(i + 1, 1)]),
                )
                .set_mutation_info(datatype::MutationInfo::new(0, 0)),
            );
        }

        feedback_collector.process_feedback(feedbacks);
        assert_eq!(feedback_collector.visited_bytes_num(), 0x11);
    }

    #[test]
    fn feedback_collector_categorize_different_types_feedback() {
        let mut feedback_collector = BitmapCollector::new(0x100);
        let mut feedbacks = Vec::new();

        // Add 10 interesting feedback of new coverage, this should produce 1 mutation feedback, 10 interesting test case feedback, 1 test case feedback.
        for i in 0..10usize {
            feedbacks.push(
                Feedback::create_coverage_feedback(
                    ExecutionStatus::Interesting,
                    Some(TestCase::new(vec![i as u8; 1], 0)),
                    Some(vec![NewBit::new(i, 1), NewBit::new(i + 1, 1)]),
                )
                .set_mutation_info(datatype::MutationInfo::new(0, 0)),
            );
        }

        // Add 10 feedback of uninteresting from 2 mutator, this should produce two mutation feedback, 1 test case feedback.
        for i in 0..10usize {
            feedbacks.push(
                Feedback::create_coverage_feedback(ExecutionStatus::Ok, None, None)
                    .set_mutation_info(datatype::MutationInfo::new(0, (i as u32) & 1)),
            );
        }

        // Add 10 feedback of timeout from 2 mutator, this should produce two timeout feedback, 1 test case feedback.
        for i in 0..10usize {
            feedbacks.push(
                Feedback::create_coverage_feedback(ExecutionStatus::Timeout, None, None)
                    .set_mutation_info(datatype::MutationInfo::new(0, (i as u32) & 1)),
            );
        }
        // Add 10 feedback of timeout from 2 mutator, this should produce two crash feedback, 1 test case feedback, 2 crash test case feedback.
        for i in 0..10usize {
            feedbacks.push(
                Feedback::create_coverage_feedback(
                    ExecutionStatus::Crash,
                    Some(TestCase::new(vec![i as u8; 1], 0)),
                    None,
                )
                .set_mutation_info(datatype::MutationInfo::new(0, (i as u32) & 1)),
            );
        }

        feedback_collector.process_feedback(feedbacks);
        assert_eq!(feedback_collector.get_mutation_feedbacks(None).len(), 7);
        assert_eq!(feedback_collector.get_test_case_feedbacks(None).len(), 4);
        assert_eq!(feedback_collector.get_crash_test_cases(None).len(), 10);
        assert_eq!(
            feedback_collector.get_interesting_test_cases(None).len(),
            10
        );
    }

    #[test]
    fn test_virgin_map_filter_seen_bit() {
        let mut feedback_collector = BitmapCollector::new(0x100);
        let mut feedbacks = Vec::new();
        for i in 0..0x10usize {
            feedbacks.push(
                Feedback::create_coverage_feedback(
                    ExecutionStatus::Interesting,
                    Some(TestCase::new(vec![i as u8; 1], 0)),
                    Some(vec![NewBit::new(i, 1), NewBit::new(i + 1, 1)]),
                )
                .set_mutation_info(datatype::MutationInfo::new(0, 0)),
            );
        }

        feedback_collector.process_feedback(feedbacks);
        let len1 = feedback_collector.visited_bytes_num();

        let mut all_visited_new_bits = Vec::new();
        for i in 0..0x11 {
            all_visited_new_bits.push(NewBit::new(i, 1));
        }
        feedback_collector.process_feedback(vec![Feedback::create_coverage_feedback(
            ExecutionStatus::Interesting,
            Some(TestCase::new(vec![0u8; 1], 0)),
            Some(all_visited_new_bits.clone()),
        )
        .set_mutation_info(datatype::MutationInfo::new(0, 0))]);
        assert_eq!(len1, feedback_collector.visited_bytes_num());

        let len2 = feedback_collector.visited_bytes_num();
        all_visited_new_bits.push(NewBit::new(0x11, 1));
        feedback_collector.process_feedback(vec![Feedback::create_coverage_feedback(
            ExecutionStatus::Interesting,
            Some(TestCase::new(vec![0u8; 1], 0)),
            Some(all_visited_new_bits.clone()),
        )
        .set_mutation_info(datatype::MutationInfo::new(0, 0))]);
        assert_eq!(feedback_collector.visited_bytes_num(), len2 + 1);
    }

    #[test]
    fn process_feedback_generate_monitor_data() {
        let mut feedback_collector = BitmapCollector::new(0x100);
        let mut feedbacks = Vec::new();

        // Add 10 interesting feedback of new coverage, this should produce 1 mutation feedback, 10 interesting test case feedback, 1 test case feedback.
        for i in 0..10usize {
            feedbacks.push(
                Feedback::create_coverage_feedback(
                    ExecutionStatus::Crash,
                    Some(TestCase::new(vec![i as u8; 1], 0)),
                    Some(vec![NewBit::new(i, 1), NewBit::new(i + 1, 1)]),
                )
                .set_mutation_info(datatype::MutationInfo::new(0, 0)),
            );
        }

        for i in 0..10usize {
            feedbacks.push(
                Feedback::create_coverage_feedback(ExecutionStatus::Timeout, None, None)
                    .set_mutation_info(datatype::MutationInfo::new(0, (i as u32) & 1)),
            );
        }

        feedback_collector.process_feedback(feedbacks);

        let monitor_data = feedback_collector.retrieve_monitor_data();
        assert_eq!(monitor_data.len(), 2);
        for v in ["crash", "timeout"] {
            assert_eq!(
                monitor_data
                    .iter()
                    .find(|a| a.get(v).is_some())
                    .unwrap()
                    .get(v)
                    .unwrap(),
                10
            );
        }
    }
}
