use crate::datatype::{self, ExecutionStatus, FeedbackData, TestCase, TestCaseID};
use crate::minimizer::TestCaseMinimizer;
use crate::mutator::afl_mutator::DeterministicMutationPass;
use crate::queue::QueueManager;
use itertools::sorted;
use rand::rngs::StdRng;
use rand::Rng;
use rand::SeedableRng;
use std::cmp;
use std::collections::HashMap;
use std::collections::VecDeque;

pub const DEFAULT_SCORE: i64 = 100000000;
const SCORE_INTERESTING: i64 = 50;
const SCORE_TIMEOUT: i64 = -700;
const SCORE_CRASH: i64 = -50; // We assume that it is not typical for a input to find two different bugs.
const SCORE_UNINTERESTING: i64 = -7;

const MAX_SELECT_COUNT: u64 = 0xFF;
const TRIGGER_DETERMINISTIC_THRESHOLD: u64 = 6;
const TRIGGER_DETERMINISTIC_INTERVAL: u64 = 3;

#[derive(Debug, Clone, Default)]
pub struct TestCaseWithMeta {
    test_case: TestCase,
    // Indicate the quality of the test case. The higher it is, the more likely it should be picked for mutation.
    score: i64,
    select_count: u64, // How many times is the test case selected for mutation
}

impl PartialEq for TestCaseWithMeta {
    fn eq(&self, other: &Self) -> bool {
        self.score == other.score
    }
}

impl Eq for TestCaseWithMeta {}

impl Ord for TestCaseWithMeta {
    fn cmp(&self, other: &Self) -> cmp::Ordering {
        other.score.cmp(&self.score)
    }
}

impl PartialOrd for TestCaseWithMeta {
    fn partial_cmp(&self, other: &Self) -> Option<cmp::Ordering> {
        Some(self.cmp(other))
    }
}

impl TestCaseWithMeta {
    pub fn new(test_case: TestCase, score: i64) -> Self {
        Self {
            test_case,
            score,
            select_count: 0,
        }
    }

    pub fn get_test_case(&self) -> &TestCase {
        &self.test_case
    }

    pub fn add_score(&mut self, delta: i64) {
        self.score += delta;
        assert!(self.score > 0);
    }

    pub fn set_score(&mut self, delta: i64) {
        self.score = delta;
        assert!(self.score > 0);
    }

    pub fn select(&mut self) -> TestCase {
        self.increase_select_count();

        self.select_by_count(self.select_count)
        //self.test_case.clone()
    }

    pub fn get_select_counter(&self) -> u64 {
        self.select_count
    }

    fn increase_select_count(&mut self) {
        if self.select_count < MAX_SELECT_COUNT {
            self.select_count += 1;
        }
    }

    fn select_by_count(&mut self, select_count: u64) -> TestCase {
        if select_count >= TRIGGER_DETERMINISTIC_THRESHOLD
            && select_count % TRIGGER_DETERMINISTIC_INTERVAL == 0
        {
            let mut test_case = self.test_case.clone();
            let mut meta = test_case.take_meta().unwrap_or_default();
            for flag in DeterministicMutationPass::all().iter() {
                if !meta.get_done_pass().contains(flag) {
                    meta.set_done_pass(meta.get_done_pass() | flag);
                    self.test_case.set_meta(meta.clone());
                    meta.set_todo_pass(flag);
                    test_case.set_meta(meta);
                    return test_case;
                }
            }
        }
        self.test_case.clone_without_meta()
    }
}

pub struct SimpleQueueManager {
    // Test cases that have been picked and are waiting for mutation.
    to_mutate_queue: VecDeque<TestCase>,
    // Interesting queues that we should save.
    interesting_queue: HashMap<TestCaseID, TestCaseWithMeta>,
    rng: StdRng,
    minimizer: Option<TestCaseMinimizer>,
}

impl SimpleQueueManager {
    pub fn new(test_case_minimizer: Option<TestCaseMinimizer>) -> Self {
        SimpleQueueManager {
            to_mutate_queue: VecDeque::default(),
            interesting_queue: HashMap::default(),
            rng: StdRng::seed_from_u64(crate::datatype::get_id() as u64),
            minimizer: test_case_minimizer,
        }
    }

    pub fn set_test_case_score(&mut self, id: TestCaseID, score: i64) {
        if self.interesting_queue.contains_key(&id) {
            self.interesting_queue
                .get_mut(&id)
                .unwrap()
                .set_score(score);
        }
    }

    pub fn add_score_to_test_case(&mut self, id: TestCaseID, score: i64) {
        if self.interesting_queue.contains_key(&id) {
            self.interesting_queue
                .get_mut(&id)
                .unwrap()
                .add_score(score);
        }
    }

    fn add_new_test_case(&mut self, test_case: TestCase) {
        let mut test_case = test_case;
        let new_id = test_case.get_id();
        if let Some(minimizer) = self.minimizer.as_mut() {
            test_case = minimizer.minimize(test_case);
            assert_eq!(test_case.get_id(), new_id);
        }
        assert!(!self.interesting_queue.contains_key(&new_id));
        self.interesting_queue
            .insert(new_id, TestCaseWithMeta::new(test_case, DEFAULT_SCORE));
    }
}

impl Default for SimpleQueueManager {
    fn default() -> Self {
        Self::new(None)
    }
}

impl QueueManager for SimpleQueueManager {
    fn receive_interesting_test_cases(&mut self, test_cases: Vec<TestCase>) {
        for mut test_case in test_cases.into_iter() {
            // Generate a unique id for the new test case
            test_case.gen_id();
            self.add_new_test_case(test_case);
        }
    }

    fn process_test_case_feedbacks(&mut self, feedbacks: &[datatype::Feedback]) {
        let mut score_changes: HashMap<TestCaseID, i64> = HashMap::default();
        for feedback in feedbacks.iter() {
            let mutation_info = feedback.get_mutation_info().unwrap();
            let pid = mutation_info.get_pid();

            let counter = match feedback.borrow_data() {
                Some(FeedbackData::Counter(c)) => *c as i64,
                _ => {
                    unreachable!();
                }
            };

            let score_change = counter
                * match feedback.get_status() {
                    ExecutionStatus::Ok => SCORE_UNINTERESTING,
                    ExecutionStatus::Interesting => SCORE_INTERESTING,
                    ExecutionStatus::Timeout => SCORE_TIMEOUT,

                    ExecutionStatus::Crash => SCORE_CRASH,
                };
            self.add_score_to_test_case(pid, score_change);
            score_changes
                .entry(pid)
                .and_modify(|e| *e += score_change)
                .or_insert(score_change);
        }

        for (pid, score_change) in score_changes {
            self.add_score_to_test_case(pid, score_change);
        }
    }

    fn select_interesting_inputs(&mut self, num: usize) -> Vec<TestCase> {
        if self.interesting_queue.is_empty() {
            return Vec::default();
        }

        while self.to_mutate_queue.len() < num {
            let len = self.interesting_queue.len();
            self.to_mutate_queue.extend(
                sorted(self.interesting_queue.values_mut())
                    .filter_map(|test_case_with_meta| {
                        // Make every test case have at least 50% chance of being selected
                        if self.rng.gen_range(1..MAX_SELECT_COUNT * 2)
                            > test_case_with_meta.get_select_counter()
                        {
                            Some(test_case_with_meta.select())
                        } else {
                            None
                        }
                    })
                    .take((len / 2).max(1)),
            );
        }
        self.to_mutate_queue.drain(0..num).collect()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use datatype::Feedback;
    use datatype::MutationInfo;

    #[test]
    #[serial_test::serial]
    fn simple_manager_correctly_add_new_interesting_inputs() {
        let mut simple_manager = SimpleQueueManager::default();

        // Create 0x100 executed inputs which wait for feedback.
        simple_manager.receive_interesting_test_cases(
            (0..100)
                .map(|i| TestCase::new(vec![i; 1], i as u32))
                .collect(),
        );
        (0..10).for_each(|_| {
            assert_eq!(simple_manager.select_interesting_inputs(10).len(), 10);
        })
    }

    #[test]
    #[serial_test::serial]
    fn queue_manager_select_inputs_with_highest_scores() {
        let mut simple_manager = SimpleQueueManager::default();
        for i in 0..20 {
            simple_manager.add_new_test_case(TestCase::new(Vec::default(), i));
        }

        let mut feedbacks = Vec::default();

        // Make the test case from 5 to 14 most interesting.
        for i in 5..15 {
            feedbacks.push(
                Feedback::new(ExecutionStatus::Interesting)
                    .set_mutation_info(MutationInfo::new(i, 1))
                    .set_data(FeedbackData::Counter(1)),
            );
        }

        simple_manager.process_test_case_feedbacks(&feedbacks);
        let interesing_inputs = simple_manager.select_interesting_inputs(20);
        assert_eq!(interesing_inputs.len(), 20);
        let mut ids = interesing_inputs
            .iter()
            .take(10)
            .map(|a| a.get_id())
            .collect::<Vec<_>>();

        ids.sort_unstable();
        println!("{:?}", ids);
        assert!((5..15).eq(ids.into_iter()));
    }

    #[test]
    fn every_test_case_have_chances_of_being_selected_even_with_very_low_score() {
        let mut simple_manager = SimpleQueueManager::default();
        for i in 0..10 {
            simple_manager.add_new_test_case(TestCase::new(Vec::default(), i));
        }

        // Make the test case 0 very uninteresting.
        let feedbacks = vec![Feedback::new(ExecutionStatus::Timeout)
            .set_mutation_info(MutationInfo::new(0, 1))
            .set_data(FeedbackData::Counter(100))];

        simple_manager.process_test_case_feedbacks(&feedbacks);
        loop {
            if simple_manager
                .select_interesting_inputs(20)
                .iter()
                .any(|test_case| test_case.get_id() == 0)
            {
                break;
            }
        }
    }

    #[test]
    fn test_case_will_try_to_run_deterministic_pass_if_selected_too_many_times() {
        let mut simple_manager = SimpleQueueManager::default();
        simple_manager.add_new_test_case(TestCase::new(Vec::default(), 0));
        let test_cases =
            simple_manager.select_interesting_inputs(TRIGGER_DETERMINISTIC_THRESHOLD as usize - 1);
        assert_eq!(
            test_cases
                .iter()
                .filter(|test_case| test_case.has_meta())
                .count(),
            0,
        );
        let test_case = simple_manager
            .select_interesting_inputs(1)
            .first()
            .unwrap()
            .clone();
        assert!(test_case.has_meta());
        assert!(test_case
            .borrow_meta()
            .unwrap()
            .get_todo_pass()
            .contains(DeterministicMutationPass::BITFLIP1));
        let test_case = simple_manager
            .select_interesting_inputs(TRIGGER_DETERMINISTIC_INTERVAL as usize)
            .last()
            .unwrap()
            .clone();
        assert!(test_case.has_meta());
        assert!(test_case
            .borrow_meta()
            .unwrap()
            .get_todo_pass()
            .contains(DeterministicMutationPass::BITFLIP2));
    }
}
