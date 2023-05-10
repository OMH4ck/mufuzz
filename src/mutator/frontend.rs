use crate::datatype::Feedback;
use crate::datatype::TestCase;
use crate::frontend::simple_frontend::{
    AsyncFrontend, AsyncFrontendChannel, AsyncReceiver, AsyncSender, BasicFrontend,
};
use crate::frontend::Worker;
use crate::mutator::afl_mutator::BitFlipMutator;
use crate::mutator::Mutator;
use async_trait::async_trait;
use dashmap::DashSet;
use rayon::prelude::*;
use std::collections::hash_map::DefaultHasher;
use std::hash::{Hash, Hasher};
use std::sync::atomic::{AtomicU64, Ordering};
use std::sync::{Arc, Mutex};

pub type BitFlipMutatorResultPool = Arc<Mutex<Vec<TestCase>>>;
pub type BitFlipMutatorWorkerPool = Vec<Arc<Mutex<BitFlipMutatorWorker>>>;

#[derive(Default)]
pub struct BitFlipMutatorWorker {
    mutator: BitFlipMutator,
    mutate_round: usize,
}

#[derive(Default)]
pub struct BitFlipMutatorFrontend {
    worker_pool: BitFlipMutatorWorkerPool,
    unique_hashes: DashSet<u64>,
    sender: Option<AsyncSender<Vec<TestCase>>>,
    output_receiver: Option<AsyncReceiver<Vec<TestCase>>>,
    output_sender: Option<AsyncSender<Vec<TestCase>>>,
    total_mutation: AtomicU64,
}

const DEFAULT_ROUND: usize = 50;
impl BitFlipMutatorWorker {
    pub fn new() -> Self {
        Self {
            mutator: BitFlipMutator::new(),
            mutate_round: DEFAULT_ROUND,
        }
    }

    pub fn process_feedbacks(&mut self, feedbacks: &[Feedback]) {
        self.mutator.process_mutation_feedback(feedbacks);
    }
}

impl Worker for BitFlipMutatorWorker {
    type Input = TestCase;
    type Output = TestCase;
    fn handle_one_input(&mut self, input: Vec<TestCase>) -> Vec<Self::Output> {
        let mut new_test_cases = Vec::with_capacity(self.mutate_round * input.len());
        for single_input in input {
            self.mutator.mutate(&single_input, self.mutate_round);
            new_test_cases.extend(self.mutator.get_mutated_test_cases(None).into_iter());
        }
        new_test_cases
    }
}

impl BitFlipMutatorFrontend {
    #[allow(dead_code)]
    pub fn new() -> Self {
        BitFlipMutatorFrontend {
            worker_pool: Vec::default(),
            unique_hashes: DashSet::new(),
            sender: None,
            output_receiver: None,
            output_sender: None,
            total_mutation: AtomicU64::new(0),
        }
    }

    // Deduplicate inputs by hashing the data buffer in TestCase.
    fn deep_filter(&self, test_cases: Vec<TestCase>) -> Vec<TestCase> {
        fn calculate_test_case_hash(t: &TestCase) -> u64 {
            let mut s = DefaultHasher::new();
            t.get_buffer().hash(&mut s);
            s.finish()
        }

        let total_mutation = self.total_mutation.load(Ordering::SeqCst) + test_cases.len() as u64;
        self.total_mutation.store(total_mutation, Ordering::SeqCst);

        let unique_hashes = &self.unique_hashes;
        test_cases
            .into_par_iter()
            .filter(|test_case| {
                let hash_sum = calculate_test_case_hash(test_case);
                unique_hashes.insert(hash_sum)
            })
            .collect()
        /*
        println!(
            "Total mutation: {}, unique mutation {}, effective rate {}",
            total_mutation,
            unique_hashes.len(),
            unique_hashes.len() as f64 / total_mutation as f64
        );
        */
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

    #[cfg(test)]
    fn get_output_sender(&self) -> &AsyncSender<Vec<TestCase>> {
        self.output_sender.as_ref().unwrap()
    }
}

impl BasicFrontend for BitFlipMutatorFrontend {
    type Worker = BitFlipMutatorWorker;
    type Output = TestCase;

    // TODO: In the future we should be able to add different mutator.
    // We need to make this more flexible.
    fn create_worker(&mut self, num: u32) {
        self.worker_pool
            .extend((0..num).map(|_| Arc::new(Mutex::new(BitFlipMutatorWorker::new()))));
    }

    crate::frontend_default!();

    fn output_transfrom(&self, input: Vec<TestCase>) -> Vec<TestCase> {
        self.deep_filter(input)
    }
}

impl AsyncFrontendChannel for BitFlipMutatorFrontend {
    crate::async_frontend_default!();
}

#[async_trait]
impl AsyncFrontend for BitFlipMutatorFrontend {}

#[cfg(test)]
mod tests {
    use super::*;

    #[tokio::test]
    async fn mutator_fronted_deduplicate_mutated_test_cases() {
        let mut mutator_frontend = BitFlipMutatorFrontend::new();
        mutator_frontend.create_worker(10);
        println!("Go check1234 ");
        mutator_frontend.run().await;
        println!("Go check123 ");
        let sender = mutator_frontend.get_output_sender();
        sender
            .send((0..10).map(|i| TestCase::new(vec![i; 1], 0)).collect())
            .await
            .unwrap();
        println!("Go check12 ");
        sender
            .send((0..10).map(|i| TestCase::new(vec![i; 1], 0)).collect())
            .await
            .unwrap();
        println!("Go check");
        let result = mutator_frontend.get_results(None).await;
        assert_eq!(result.len(), 10);
        println!("Go check 33");

        sender
            .send((0..100).map(|_i| TestCase::new(vec![0; 1], 0)).collect())
            .await
            .unwrap();

        let result = mutator_frontend.get_results(Some(100)).await;
        assert_eq!(result.len(), 0);
    }
}

pub mod new_mutator_frontend {
    use super::DashSet;
    use super::TestCase;
    use super::DEFAULT_ROUND;
    use crate::datatype::{ExecutionStatus, FeedbackData};
    use crate::frontend::load_balance_frontend::*;
    use crate::frontend::FuzzerIO;
    use crate::mutator::Mutator;
    use rayon::prelude::*;
    use std::collections::hash_map::DefaultHasher;
    use std::collections::HashMap;
    use std::hash::{Hash, Hasher};

    pub struct BitFlipMutatorWorker {
        mutator: super::BitFlipMutator,

        inputs_all_bytes: usize,
        inputs_num: usize,
        acc_counter: usize,
    }

    impl Worker for BitFlipMutatorWorker {
        fn handle_inputs(&mut self, input: FuzzerIO) -> Option<Vec<FuzzerIO>> {
            match input {
                FuzzerIO::TestCase(data) => {
                    let mut new_test_cases = Vec::with_capacity(DEFAULT_ROUND * data.len());
                    for single_input in data {
                        self.mutator.mutate(&single_input, DEFAULT_ROUND);
                        new_test_cases
                            .extend(self.mutator.get_mutated_test_cases(None).into_iter());
                    }

                    self.inputs_num += new_test_cases.len();
                    for input in new_test_cases.iter() {
                        self.inputs_all_bytes += input.get_buffer().len();
                    }

                    self.acc_counter += 1;
                    if self.acc_counter % 100 == 0 {
                        println!(
                            "Average len: {} for {} inputs.",
                            self.inputs_all_bytes / self.inputs_num,
                            self.inputs_num
                        );
                        self.inputs_all_bytes = 0;
                        self.inputs_num = 0;
                    }

                    Some(vec![FuzzerIO::TestCase(new_test_cases)])
                }

                FuzzerIO::MutatorScoreChange(data) => {
                    self.mutator.update_mutator_func_score(data);
                    None
                }
                _ => {
                    unreachable!();
                }
            }
        }
    }

    impl BitFlipMutatorWorker {
        pub fn new() -> Self {
            Self {
                mutator: super::BitFlipMutator::new(),

                inputs_all_bytes: 0,
                inputs_num: 0,
                acc_counter: 0,
            }
        }
    }

    impl Default for BitFlipMutatorWorker {
        fn default() -> Self {
            Self::new()
        }
    }

    pub struct MutatorTransformer {
        unique_hashes: DashSet<u64>,
    }

    impl MutatorTransformer {
        pub fn new() -> Self {
            Self {
                unique_hashes: DashSet::default(),
            }
        }
    }

    impl Default for MutatorTransformer {
        fn default() -> Self {
            Self::new()
        }
    }

    impl Transformer for MutatorTransformer {
        fn transform(&self, input: FuzzerIO) -> FuzzerIO {
            fn calculate_test_case_hash(t: &TestCase) -> u64 {
                let mut s = DefaultHasher::new();
                t.get_buffer().hash(&mut s);
                s.finish()
            }

            match input {
                FuzzerIO::TestCase(test_cases) => {
                    // Disable filtering
                    if !test_cases.is_empty() {
                        return FuzzerIO::TestCase(test_cases);
                    }

                    let unique_hashes = &self.unique_hashes;
                    FuzzerIO::TestCase(
                        test_cases
                            .into_par_iter()
                            .filter(|test_case| {
                                let hash_sum = calculate_test_case_hash(test_case);
                                unique_hashes.insert(hash_sum)
                            })
                            .collect(),
                    )
                }
                _ => {
                    unreachable!();
                }
            }
        }
    }

    #[derive(Default)]
    pub struct MutatorSyncSource {
        mutator_scores: HashMap<u32, i64>,
    }

    impl SyncSource for MutatorSyncSource {
        fn update_status(&mut self, input: FuzzerIO) {
            const DEFAULT_SCORE: i64 = 1000000000000;
            const SCORE_UNINTERESTING: i64 = -1;
            const SCORE_INTERESTING: i64 = 1000;
            const SCORE_TIMEOUT: i64 = -100000;
            const SCORE_CRASH: i64 = 10000;

            if let FuzzerIO::MutationFeedback(feedbacks) = input {
                for feedback in feedbacks.iter() {
                    let mutation_info = feedback.get_mutation_info().unwrap();
                    let counter = match feedback.borrow_data() {
                        Some(FeedbackData::Counter(c)) => *c as i64,
                        _ => {
                            unreachable!();
                        }
                    };

                    let score_change = counter
                        * match feedback.get_status() {
                            ExecutionStatus::Ok => SCORE_UNINTERESTING,
                            ExecutionStatus::Timeout => SCORE_TIMEOUT,
                            ExecutionStatus::Crash => SCORE_CRASH,
                            ExecutionStatus::Interesting => SCORE_INTERESTING,
                        };

                    self.mutator_scores
                        .entry(mutation_info.get_mutation_id())
                        .and_modify(|e| *e += score_change)
                        .or_insert(DEFAULT_SCORE + score_change);
                }
            } else {
                unreachable!();
            }
        }

        fn get_status(&mut self) -> FuzzerIO {
            FuzzerIO::MutatorScoreChange(
                self.mutator_scores
                    .iter()
                    .map(|(id, score)| (*id, *score))
                    .collect(),
            )
        }
    }

    pub type BitFlipMutatorFrontend =
        Frontend<BitFlipMutatorWorker, MutatorTransformer, MutatorSyncSource>;

    #[cfg(test)]
    mod tests {
        use super::*;
        use crate::frontend::FuzzerIOType;

        #[tokio::test(flavor = "multi_thread", worker_threads = 5)]
        #[serial_test::serial]
        async fn mutator_load_balance_frontend_test() {
            let closure = BitFlipMutatorWorker::new;
            let mut frontend = BitFlipMutatorFrontend::new(Box::new(closure));
            let transformer = std::sync::Arc::new(MutatorTransformer::new());
            let worker_num = 10;
            frontend.create_worker(worker_num);
            frontend.set_transformer(Some(transformer));
            let (input_sender, input_rx) = frontend_channel(100);
            let (output_sender, mut output_rx) = frontend_channel(100);
            let (mutation_feedback_sender, mutation_feedback_rx) = frontend_channel(100);
            frontend.register_input_handler(Receiver::AsyncExclusive(input_rx));
            frontend.register_watch_input_handler(
                Receiver::AsyncExclusive(mutation_feedback_rx),
                MutatorSyncSource::default(),
            );
            frontend.register_output_handler(
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

            let test_case = TestCase::new(String::from("select 1;").as_bytes().to_vec(), 0);
            let test_cases = (0..100)
                .map(|_x| test_case.clone())
                .collect::<Vec<TestCase>>();

            for _i in 0..10 {
                input_sender
                    .send(FuzzerIO::TestCase(test_cases.clone()))
                    .await
                    .unwrap();
                let output = output_rx.recv().await.unwrap();
                match output.get_type() {
                    FuzzerIOType::TestCase => {
                        //println!("Receive output");
                    }
                    _ => {
                        unreachable!();
                    }
                }
                mutation_feedback_sender
                    .send(FuzzerIO::MutationFeedback(feedbacks.clone()))
                    .await
                    .unwrap();
            }
        }
    }
}

pub mod work_stealing_mutator {
    use super::DEFAULT_ROUND;
    use crate::frontend::load_balance_frontend::SyncSource;
    use crate::frontend::work_stealing_frontend::WorkerImpl;
    use crate::frontend::{FuzzerIO, FuzzerIOType};
    use crate::mutator::Mutator;

    pub struct MutatorSyncSource {
        sync_source: super::new_mutator_frontend::MutatorSyncSource,
        counter: usize,
    }

    impl MutatorSyncSource {
        pub fn new() -> Self {
            MutatorSyncSource {
                sync_source: super::new_mutator_frontend::MutatorSyncSource::default(),
                counter: 0,
            }
        }
    }

    impl Default for MutatorSyncSource {
        fn default() -> Self {
            Self::new()
        }
    }

    impl WorkerImpl for MutatorSyncSource {
        fn handle_inputs(&mut self, input: FuzzerIO) -> Option<Vec<FuzzerIO>> {
            assert!(input.get_type() == FuzzerIOType::MutationFeedback);
            self.sync_source.update_status(input);
            self.counter += 1;
            if self.counter == 20 {
                self.counter = 0;
                Some(vec![self.sync_source.get_status()])
            } else {
                None
            }
        }
    }

    pub struct MutatorWorker {
        mutator: super::BitFlipMutator,

        inputs_all_bytes: usize,
        inputs_num: usize,
        acc_counter: usize,
    }

    impl WorkerImpl for MutatorWorker {
        fn handle_inputs(&mut self, input: FuzzerIO) -> Option<Vec<FuzzerIO>> {
            match input {
                FuzzerIO::TestCase(data) => {
                    let mut new_test_cases = Vec::with_capacity(DEFAULT_ROUND * data.len());
                    for single_input in data {
                        self.mutator.mutate(&single_input, DEFAULT_ROUND);
                        new_test_cases
                            .extend(self.mutator.get_mutated_test_cases(None).into_iter());
                    }

                    self.inputs_num += new_test_cases.len();
                    for input in new_test_cases.iter() {
                        self.inputs_all_bytes += input.get_buffer().len();
                    }

                    self.acc_counter += 1;
                    if self.acc_counter % 100 == 0 {
                        println!(
                            "Average len: {} for {} inputs.",
                            self.inputs_all_bytes / self.inputs_num,
                            self.inputs_num
                        );
                        self.inputs_all_bytes = 0;
                        self.inputs_num = 0;
                    }

                    Some(vec![FuzzerIO::TestCase(new_test_cases)])
                }
                FuzzerIO::MutatorScoreChange(data) => {
                    self.mutator.update_mutator_func_score(data);
                    None
                }
                _ => {
                    unreachable!();
                }
            }
        }
    }

    impl MutatorWorker {
        pub fn new() -> Self {
            Self {
                mutator: super::BitFlipMutator::new(),

                inputs_all_bytes: 0,
                inputs_num: 0,
                acc_counter: 0,
            }
        }
    }

    impl Default for MutatorWorker {
        fn default() -> Self {
            Self::new()
        }
    }
}
