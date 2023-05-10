use crate::datatype::{Feedback, TestCase};
use crate::frontend::simple_frontend::{
    AsyncFrontend, AsyncFrontendChannel, AsyncReceiver, AsyncSender, BasicFrontend,
};
use crate::frontend::Worker;
use crate::frontend_default;
use crate::queue::manager::SimpleQueueManager;
use crate::queue::QueueManager;
use async_trait::async_trait;
use rayon::prelude::*;
use std::sync::{Arc, Mutex};

type SimpleQueueManagerWorkerPool = Vec<Arc<Mutex<SimpleQueueManagerWorker>>>;
type SimpleManagerT = SimpleQueueManager;

#[derive(Default)]
pub struct SimpleQueueManagerWorker {
    manager: SimpleManagerT,
}

pub struct SimpleQueueManagerFrontend {
    worker_pool: SimpleQueueManagerWorkerPool,
    sender: Option<AsyncSender<Vec<TestCase>>>,
    output_receiver: Option<AsyncReceiver<Vec<TestCase>>>,
    output_sender: Option<AsyncSender<Vec<TestCase>>>,
}

impl SimpleQueueManagerWorker {
    pub fn new() -> Self {
        Self::default()
    }

    pub fn process_feedbacks(&mut self, feedbacks: &[Feedback]) {
        self.manager.process_test_case_feedbacks(feedbacks);
    }

    pub fn select_interesting_inputs(&mut self, num: usize) -> Vec<TestCase> {
        self.manager.select_interesting_inputs(num)
    }
}

impl Worker for SimpleQueueManagerWorker {
    type Input = TestCase;
    type Output = TestCase;

    // TODO: This should be non-blocking frontend
    fn handle_one_input(&mut self, input: Vec<TestCase>) -> Vec<Self::Output> {
        self.manager.receive_interesting_test_cases(input);
        Vec::default()
    }
}

impl SimpleQueueManagerFrontend {
    #[allow(dead_code)]
    pub fn new() -> Self {
        SimpleQueueManagerFrontend {
            worker_pool: Vec::default(),
            sender: None,
            output_receiver: None,
            output_sender: None,
        }
    }

    fn collect_interesting_inputs(&self, num: usize) -> Vec<TestCase> {
        let mut result = Vec::new();
        let average =
            num / self.get_worker_pool().len() + 1.min(num % self.get_worker_pool().len());
        assert_ne!(average, 0);
        while result.len() < num {
            for worker in self.get_worker_pool() {
                let mut worker = worker.lock().unwrap();
                result.extend(
                    worker
                        .select_interesting_inputs(average.min(num - result.len()))
                        .into_iter(),
                );
            }
        }
        assert_eq!(result.len(), num);
        result
    }

    pub async fn process_feedbacks(&self, inputs: Vec<Feedback>) {
        if inputs.is_empty() {
            return;
        }

        self.get_worker_pool().par_iter().for_each(|worker| {
            let worker = worker.clone();
            let mut worker = worker.lock().unwrap();
            worker.process_feedbacks(&inputs);
        });
    }
}

impl Default for SimpleQueueManagerFrontend {
    fn default() -> Self {
        Self::new()
    }
}

impl BasicFrontend for SimpleQueueManagerFrontend {
    type Worker = SimpleQueueManagerWorker;
    type Output = TestCase;
    fn create_worker(&mut self, num: u32) {
        self.worker_pool
            .extend((0..num).map(|_| Arc::new(Mutex::new(SimpleQueueManagerWorker::new()))));
    }

    fn output_transfrom(&self, input: Vec<TestCase>) -> Vec<TestCase> {
        input
    }
    frontend_default!();
}

impl AsyncFrontendChannel for SimpleQueueManagerFrontend {
    crate::async_frontend_default!();
}

#[async_trait]
impl AsyncFrontend for SimpleQueueManagerFrontend {
    async fn get_results(&self, num: Option<u32>) -> Vec<Self::Output> {
        // TODO: FIX this pure magic number.
        let max_num = self.get_worker_pool().len() * 2;
        let num = match num {
            Some(num) => (num as usize).min(max_num),
            None => max_num,
        };
        self.collect_interesting_inputs(num)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    #[serial_test::serial]
    fn simple_queue_manager_frontend_add_delete_worker_correctly() {
        let mut simple_queue_manager_frontend = SimpleQueueManagerFrontend::new();
        simple_queue_manager_frontend.create_worker(15);
        assert!(simple_queue_manager_frontend.worker_pool.len() == 15);
        simple_queue_manager_frontend.delete_worker(5);
        assert!(simple_queue_manager_frontend.worker_pool.len() == 10);
    }
}

pub mod new_queue_frontend {
    use crate::datatype::{ExecutionStatus, FeedbackData};
    use crate::datatype::{TestCase, TestCaseID};
    use crate::frontend::load_balance_frontend::*;
    use crate::frontend::{FuzzerIO, FuzzerIOType};
    use crate::minimizer::TestCaseMinimizer;
    use crate::queue::QueueManager;
    use std::collections::HashMap;

    pub struct QueueManagerWorker {
        manager: super::SimpleQueueManager,
    }

    impl Worker for QueueManagerWorker {
        fn handle_inputs(&mut self, input: FuzzerIO) -> Option<Vec<FuzzerIO>> {
            match input {
                FuzzerIO::TestCase(data) => {
                    let len = data.len();
                    self.manager.receive_interesting_test_cases(data);
                    Some(vec![FuzzerIO::MonitorData(vec![serde_json::json!({
                        "interesting_test_case": len as u64
                    })])])
                }
                FuzzerIO::TestCaseScoreChange(data) => {
                    for (id, score_change) in data.into_iter() {
                        self.manager.set_test_case_score(id, score_change);
                    }
                    None
                }
                _ => {
                    unreachable!();
                }
            }
        }

        fn non_blocking_generate_output(&mut self, gen_type: FuzzerIOType) -> Option<FuzzerIO> {
            match gen_type {
                FuzzerIOType::TestCase => {
                    let outputs = self.manager.select_interesting_inputs(20);
                    if outputs.is_empty() {
                        None
                    } else {
                        Some(FuzzerIO::TestCase(outputs))
                    }
                }
                _ => {
                    unreachable!();
                }
            }
        }
    }

    impl QueueManagerWorker {
        pub fn new(test_case_minimizer: Option<TestCaseMinimizer>) -> Self {
            Self {
                manager: super::SimpleQueueManager::new(test_case_minimizer),
            }
        }

        pub fn new_with_seeds(
            initial_corpus: Option<Vec<TestCase>>,
            test_case_minimizer: Option<TestCaseMinimizer>,
        ) -> Self {
            let mut manager = super::SimpleQueueManager::new(test_case_minimizer);
            if let Some(initial_corpus) = initial_corpus {
                manager.receive_interesting_test_cases(initial_corpus);
            }
            Self { manager }
        }
    }

    impl Default for QueueManagerWorker {
        fn default() -> Self {
            Self::new(None)
        }
    }

    #[derive(Default, Clone)]
    pub struct QueueManagerTransformer {}

    impl Transformer for QueueManagerTransformer {}

    #[derive(Default)]
    pub struct QueueSyncSource {
        queue_scores: HashMap<TestCaseID, i64>,
        scores_to_send_out: Vec<(TestCaseID, i64)>,
    }

    impl SyncSource for QueueSyncSource {
        fn update_status(&mut self, input: FuzzerIO) {
            if let FuzzerIO::TestCaseFeedback(feedbacks) = input {
                const DEFAULT_SCORE: i64 = 100000000;
                const SCORE_INTERESTING: i64 = 20;
                const SCORE_TIMEOUT: i64 = -1000;
                const SCORE_CRASH: i64 = -50;
                const SCORE_UNINTERESTING: i64 = -1;

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
                    self.queue_scores
                        .entry(pid)
                        .and_modify(|e| *e += score_change)
                        .or_insert(score_change + DEFAULT_SCORE);
                }
            } else {
                unreachable!();
            }
        }
        fn get_status(&mut self) -> FuzzerIO {
            if self.scores_to_send_out.len() < 1000 {
                self.scores_to_send_out
                    .extend(self.queue_scores.iter().map(|(id, score)| (*id, *score)));
            }
            /*
            FuzzerIO::TestCaseScoreChange(
                self.queue_scores
                    .iter()
                    .map(|(id, score)| (*id, *score))
                    .collect(),
            )
            */
            FuzzerIO::TestCaseScoreChange(
                self.scores_to_send_out
                    .split_off(self.scores_to_send_out.len().saturating_sub(1000)),
            )
        }
    }

    pub type QueueManagerFrontend =
        Frontend<QueueManagerWorker, QueueManagerTransformer, QueueSyncSource>;

    #[cfg(test)]
    mod tests {
        use super::*;
        use crate::datatype::TestCase;

        #[tokio::test(flavor = "multi_thread", worker_threads = 20)]
        #[serial_test::serial]
        async fn queue_manager_load_balance_frontend_test() {
            let closure = move || QueueManagerWorker::new(None);
            let mut frontend = QueueManagerFrontend::new(Box::new(closure));
            let worker_num = 5;
            frontend.create_worker(worker_num);
            let (test_case_feedback_sender, test_case_feedback_rx) = frontend_channel(100);
            let (input_sender, input_rx) = frontend_channel(100);
            let (output_sender, mut output_rx) = frontend_channel(100);
            frontend.register_input_handler(Receiver::AsyncExclusive(input_rx));

            frontend.register_watch_input_handler(
                Receiver::AsyncExclusive(test_case_feedback_rx),
                QueueSyncSource::default(),
            );

            frontend.register_non_blocking_output_handler(
                FuzzerIOType::TestCase,
                Sender::AsyncExclusive(output_sender),
            );

            frontend.run();

            let mut feedbacks = Vec::default();
            for i in 5..15 {
                feedbacks.push(
                    crate::datatype::Feedback::new(crate::datatype::ExecutionStatus::Interesting)
                        .set_mutation_info(crate::datatype::MutationInfo::new(i % 2, 1))
                        .set_data(crate::datatype::FeedbackData::Counter(1)),
                );
            }

            let test_cases = (0..100)
                .map(|id| TestCase::new(String::from("select 1;").as_bytes().to_vec(), id))
                .collect::<Vec<TestCase>>();
            input_sender
                .send(FuzzerIO::TestCase(test_cases.clone()))
                .await
                .unwrap();
            for _i in 0..10 {
                test_case_feedback_sender
                    .send(FuzzerIO::TestCaseFeedback(feedbacks.clone()))
                    .await
                    .unwrap();
            }
            for _i in 0..10 {
                let output = output_rx.recv().await.unwrap();
                match output.get_type() {
                    FuzzerIOType::TestCase => {
                        //println!("Receive output {} ", _i);
                    }
                    _ => {
                        unreachable!();
                    }
                }
            }
        }
    }
}

pub mod work_stealing_queue_manager {
    use crate::datatype::TestCase;
    use crate::frontend::load_balance_frontend::SyncSource;
    use crate::frontend::work_stealing_frontend::WorkerImpl;
    use crate::frontend::{FuzzerIO, FuzzerIOType};
    use crate::minimizer::TestCaseMinimizer;
    use crate::queue::QueueManager;

    pub struct QueueManagerSyncSource {
        inner: super::new_queue_frontend::QueueSyncSource,
        counter: usize,
    }

    impl QueueManagerSyncSource {
        pub fn new() -> Self {
            Self {
                inner: super::new_queue_frontend::QueueSyncSource::default(),
                counter: 0,
            }
        }
    }

    impl Default for QueueManagerSyncSource {
        fn default() -> Self {
            Self::new()
        }
    }

    impl WorkerImpl for QueueManagerSyncSource {
        fn handle_inputs(&mut self, input: FuzzerIO) -> Option<Vec<FuzzerIO>> {
            self.counter += 1;
            self.inner.update_status(input);
            if self.counter % 20 == 0 {
                self.counter = 0;
                Some(vec![self.inner.get_status()])
            } else {
                None
            }
        }
    }
    pub struct QueueManagerWorker {
        manager: super::SimpleQueueManager,
    }

    impl WorkerImpl for QueueManagerWorker {
        fn handle_inputs(&mut self, input: FuzzerIO) -> Option<Vec<FuzzerIO>> {
            match input {
                FuzzerIO::TestCase(data) => {
                    let len = data.len();
                    self.manager.receive_interesting_test_cases(data);
                    Some(vec![FuzzerIO::MonitorData(vec![serde_json::json!({
                        "interesting_test_case": len as u64
                    })])])
                }
                FuzzerIO::TestCaseScoreChange(data) => {
                    for (id, score_change) in data.into_iter() {
                        self.manager.set_test_case_score(id, score_change);
                    }
                    None
                }
                _ => {
                    unreachable!();
                }
            }
        }

        fn generate_outputs(&mut self, gen_type: FuzzerIOType) -> Option<FuzzerIO> {
            match gen_type {
                FuzzerIOType::TestCase => {
                    let outputs = self.manager.select_interesting_inputs(20);
                    if outputs.is_empty() {
                        None
                    } else {
                        Some(FuzzerIO::TestCase(outputs))
                    }
                }
                _ => {
                    println!("Generate outputs for {:?}", gen_type);
                    unreachable!();
                }
            }
        }
    }

    impl QueueManagerWorker {
        pub fn new(
            initial_corpus: Option<Vec<TestCase>>,
            test_case_minimizer: Option<TestCaseMinimizer>,
        ) -> Self {
            let mut manager = super::SimpleQueueManager::new(test_case_minimizer);
            if let Some(initial_corpus) = initial_corpus {
                manager.receive_interesting_test_cases(initial_corpus);
            }
            Self { manager }
        }
    }

    impl Default for QueueManagerWorker {
        fn default() -> Self {
            Self::new(None, None)
        }
    }
}
