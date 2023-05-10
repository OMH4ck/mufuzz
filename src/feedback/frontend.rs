use super::NewBitFilter;
use crate::datatype::ExecutionStatus;
use crate::datatype::Feedback;
use crate::datatype::{NewBit, TestCase};
use crate::feedback::bitmap_collector::*;
use crate::feedback::FeedbackCollector;
use crate::frontend::simple_frontend::{AsyncFrontend, AsyncFrontendChannel, BasicFrontend};
use crate::frontend::simple_frontend::{AsyncReceiver, AsyncSender};
use crate::frontend::Worker;
use async_trait::async_trait;
use rayon::prelude::*;
use serde_json::{json, Value};
use std::sync::{Arc, Mutex, RwLock};
type BitmapCollectorWorkerPool = Vec<Arc<Mutex<BitmapCollectorWorker>>>;

pub struct BitmapCollectorWorker {
    pub collector: BitmapCollector,
}

pub struct BitmapCollectorFrontend {
    worker_pool: BitmapCollectorWorkerPool,
    filter: Arc<RwLock<NewBitFilter>>,
    bitmap_size: usize,
    monitor_data: Arc<Mutex<Vec<Value>>>,
    sender: Option<AsyncSender<Vec<Feedback>>>,
    output_receiver: Option<AsyncReceiver<Vec<Feedback>>>,
    output_sender: Option<AsyncSender<Vec<Feedback>>>,
}

impl BitmapCollectorWorker {
    pub fn new(bitmap_size: usize) -> Self {
        Self {
            collector: BitmapCollector::new(bitmap_size),
        }
    }
}
impl Worker for BitmapCollectorWorker {
    type Input = Feedback;
    type Output = Feedback;
    fn handle_one_input(&mut self, input: Vec<Feedback>) -> Vec<Self::Output> {
        self.collector.process_feedback(input);
        self.collector.get_interesting_test_cases(None)
    }

    fn retrieve_monitor_data(&mut self) -> Vec<Value> {
        self.collector.retrieve_monitor_data()
    }
}

impl BitmapCollectorFrontend {
    #[allow(dead_code)]
    pub fn new(map_size: usize) -> Self {
        BitmapCollectorFrontend {
            worker_pool: Vec::default(),
            filter: Arc::new(RwLock::new(NewBitFilter::new(map_size))),
            bitmap_size: map_size,
            monitor_data: Arc::new(Mutex::new(Vec::default())),
            sender: None,
            output_receiver: None,
            output_sender: None,
        }
    }

    // Filter out duplicated interesting test cases and leave other feedbacks intact.
    fn deep_filter(&self, feedbacks: Vec<Feedback>) -> Vec<TestCase> {
        let new_test_cases: Vec<(Vec<NewBit>, TestCase)> = feedbacks
            .into_par_iter()
            .filter_map(|mut feedback| {
                assert!(feedback.is_valid());
                assert_eq!(feedback.get_status(), ExecutionStatus::Interesting);
                assert!(feedback.contain_test_case());
                if let Some(crate::datatype::FeedbackData::NewCoverage(new_coverage)) =
                    feedback.take_data()
                {
                    self.filter
                        .read()
                        .unwrap()
                        .filter_old_bits(new_coverage)
                        .map(|new_bits| (new_bits, feedback.take_test_case().unwrap()))
                } else {
                    unreachable!();
                }
            })
            .collect::<Vec<_>>();

        new_test_cases
            .into_iter()
            .filter_map(|(new_bits, test_case)| {
                let mut filter = self.filter.write().unwrap();
                if filter.try_update_bits(&new_bits) {
                    Some(test_case)
                } else {
                    None
                }
            })
            .collect::<Vec<_>>()
    }

    pub async fn retrieve_monitor_data(&self) -> Vec<Value> {
        let mut result = Vec::default();
        for worker in self.get_worker_pool() {
            let mut worker = worker.lock().unwrap();
            result.extend(worker.collector.retrieve_monitor_data().into_iter());
        }
        let mut monitor_data = self.monitor_data.lock().unwrap();
        result.extend(monitor_data.split_off(0).into_iter());
        result
    }

    pub async fn get_mutation_feedbacks(&self, num: Option<u32>) -> Vec<Feedback> {
        let mut result = Vec::default();
        for worker in self.get_worker_pool() {
            let mut worker = worker.lock().unwrap();
            result.extend(
                worker
                    .collector
                    .get_mutation_feedbacks(num.map(|a| a as usize))
                    .into_iter(),
            );
        }
        result
    }

    pub async fn get_test_case_feedbacks(&self, num: Option<u32>) -> Vec<Feedback> {
        let mut result = Vec::default();
        for worker in self.get_worker_pool() {
            let mut worker = worker.lock().unwrap();
            result.extend(
                worker
                    .collector
                    .get_test_case_feedbacks(num.map(|a| a as usize))
                    .into_iter(),
            );
        }
        result
    }
}

impl BasicFrontend for BitmapCollectorFrontend {
    type Worker = BitmapCollectorWorker;
    type Output = TestCase;
    fn create_worker(&mut self, num: u32) {
        self.worker_pool.extend(
            (0..num).map(|_| Arc::new(Mutex::new(BitmapCollectorWorker::new(self.bitmap_size)))),
        );
    }

    fn output_transfrom(
        &self,
        input: Vec<<Self::Worker as crate::frontend::Worker>::Output>,
    ) -> Vec<TestCase> {
        let interesting_test_cases = self.deep_filter(input);
        let mut monitor_data = self.monitor_data.lock().unwrap();
        monitor_data.push(json!({"interesting_test_case": interesting_test_cases.len() as u64}));
        interesting_test_cases
    }

    crate::frontend_default!();
}

impl AsyncFrontendChannel for BitmapCollectorFrontend {
    crate::async_frontend_default!();
}

#[async_trait]
impl AsyncFrontend for BitmapCollectorFrontend {}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::datatype::*;

    fn create_bitmap_feedback_with_test_case(num: usize) -> Vec<Feedback> {
        vec![Feedback::create_coverage_feedback(ExecutionStatus::Ok, None, None); num]
    }

    #[test]
    #[serial_test::serial]
    fn forkserver_executor_frontend_add_delete_worker_correctly() {
        const MAP_SIZE: usize = 65536;
        let mut bitmap_collector_frontend = BitmapCollectorFrontend::new(MAP_SIZE);
        bitmap_collector_frontend.create_worker(15);
        assert!(bitmap_collector_frontend.worker_pool.len() == 15);
        bitmap_collector_frontend.delete_worker(5);
        assert!(bitmap_collector_frontend.worker_pool.len() == 10);
    }

    #[tokio::test]
    #[serial_test::serial]
    #[should_panic]
    async fn bitmap_collector_fronted_panic_when_no_worker_is_created() {
        const MAP_SIZE: usize = 65536;
        let bitmap_collector_frontend = BitmapCollectorFrontend::new(MAP_SIZE);
        bitmap_collector_frontend
            .handle_inputs(create_bitmap_feedback_with_test_case(2))
            .await;
    }
}

pub mod new_feedback_frontend {
    use super::NewBitFilter;
    use crate::datatype::ExecutionStatus;
    use crate::datatype::{NewBit, TestCase};
    use crate::feedback::FeedbackCollector;
    use crate::frontend::load_balance_frontend::*;
    use crate::frontend::FuzzerIO;
    use rayon::prelude::*;
    use std::sync::{Arc, RwLock};

    pub struct FeedbackCollectorWorker {
        collector: super::BitmapCollector,
    }

    impl Worker for FeedbackCollectorWorker {
        fn handle_inputs(&mut self, input: FuzzerIO) -> Option<Vec<FuzzerIO>> {
            let mut result = Vec::new();
            match input {
                FuzzerIO::Feedback(data) => {
                    self.collector.process_feedback(data);
                    result.push(FuzzerIO::Feedback(
                        self.collector.get_interesting_test_cases(None),
                    ));
                    /*
                    self.collector.get_mutation_feedbacks(None);
                    self.collector.get_test_case_feedbacks(None);
                    */
                    result.push(FuzzerIO::MutationFeedback(
                        self.collector.get_mutation_feedbacks(None),
                    ));
                    result.push(FuzzerIO::TestCaseFeedback(
                        self.collector.get_test_case_feedbacks(None),
                    ));
                    result.push(FuzzerIO::MonitorData(
                        self.collector.retrieve_monitor_data(),
                    ));
                    Some(result)
                }
                _ => {
                    unreachable!();
                }
            }
        }
    }

    impl FeedbackCollectorWorker {
        pub fn new(bitmap_size: usize) -> Self {
            Self {
                collector: super::BitmapCollector::new(bitmap_size),
            }
        }
    }

    pub struct FeedbackCollectorTransformer {
        filter: Arc<RwLock<NewBitFilter>>,
    }

    impl FeedbackCollectorTransformer {
        pub fn new(bitmap_size: usize) -> Self {
            Self {
                filter: Arc::new(RwLock::new(NewBitFilter::new(bitmap_size))),
            }
        }
    }

    impl Transformer for FeedbackCollectorTransformer {
        // Filter out duplicated interesting test cases and leave other feedbacks intact.
        fn transform(&self, feedbacks: FuzzerIO) -> FuzzerIO {
            match feedbacks {
                FuzzerIO::Feedback(feedbacks) => {
                    let new_test_cases: Vec<(Vec<NewBit>, TestCase)> = feedbacks
                        .into_par_iter()
                        .filter_map(|mut feedback| {
                            assert!(feedback.is_valid());
                            assert_eq!(feedback.get_status(), ExecutionStatus::Interesting);
                            assert!(feedback.contain_test_case());
                            if let Some(crate::datatype::FeedbackData::NewCoverage(new_coverage)) =
                                feedback.take_data()
                            {
                                self.filter
                                    .read()
                                    .unwrap()
                                    .filter_old_bits(new_coverage)
                                    .map(|new_bits| (new_bits, feedback.take_test_case().unwrap()))
                            } else {
                                unreachable!();
                            }
                        })
                        .collect::<Vec<_>>();

                    let result = FuzzerIO::TestCase(
                        new_test_cases
                            .into_iter()
                            .filter_map(|(new_bits, test_case)| {
                                let mut filter = self.filter.write().unwrap();
                                if filter.try_update_bits(&new_bits) {
                                    Some(test_case)
                                } else {
                                    None
                                }
                            })
                            .collect::<Vec<_>>(),
                    );

                    if !result.is_empty() {
                        /*
                        let byte_count = { self.filter.read().unwrap().byte_count() };
                        println!(
                            "Map density: {}/65536, {:.2}",
                            byte_count,
                            byte_count as f64 / 63356f64 * 100f64
                        );
                        */
                    }

                    result
                }
                _ => feedbacks,
            }
        }
    }

    pub type FeedbackCollectorFrontend =
        Frontend<FeedbackCollectorWorker, FeedbackCollectorTransformer, DummySyncSource>;

    #[cfg(test)]
    mod tests {
        use super::*;
        use crate::datatype::Feedback;
        use crate::frontend::FuzzerIOType;

        #[tokio::test]
        async fn feedback_frontend_testme() {
            let bitmap_size: usize = 65536;
            let closure = move || FeedbackCollectorWorker::new(bitmap_size);
            let mut frontend = FeedbackCollectorFrontend::new(Box::new(closure));
            let transformer = Arc::new(FeedbackCollectorTransformer::new(bitmap_size));
            let worker_num = 5;
            frontend.create_worker(worker_num);
            frontend.set_transformer(Some(transformer));
            let (input_sender, input_rx) = frontend_channel(100);
            let (output_sender, mut output_rx) = frontend_channel(100);
            frontend.register_input_handler(Receiver::AsyncExclusive(input_rx));
            frontend.register_output_handler(
                FuzzerIOType::TestCase,
                Sender::AsyncExclusive(output_sender),
            );

            let (mutation_feedback_sender, _mutation_feedback_output_rx) = frontend_channel(100);
            let (test_case_feedback_sender, _test_case_feedback_output_rx) = frontend_channel(100);
            let (monitor_data_sender, _monitor_data_output_rx) = frontend_channel(100);
            frontend.register_output_handler(
                FuzzerIOType::MutationFeedback,
                Sender::AsyncExclusive(mutation_feedback_sender),
            );
            frontend.register_output_handler(
                FuzzerIOType::TestCaseFeedback,
                Sender::AsyncExclusive(test_case_feedback_sender),
            );
            frontend.register_output_handler(
                FuzzerIOType::MonitorData,
                Sender::AsyncExclusive(monitor_data_sender),
            );

            frontend.run();

            let mut feedbacks = Vec::default();
            for i in 0..10usize {
                feedbacks.push(
                    Feedback::create_coverage_feedback(
                        ExecutionStatus::Interesting,
                        Some(TestCase::new(vec![i as u8; 1], 0)),
                        Some(vec![NewBit::new(i, 1), NewBit::new(i + 1, 1)]),
                    )
                    .set_mutation_info(crate::datatype::MutationInfo::new(0, 0)),
                );
            }

            for i in 0..10 {
                input_sender
                    .send(FuzzerIO::Feedback(
                        feedbacks.iter().take(i + 1).cloned().collect(),
                    ))
                    .await
                    .unwrap();
                let output = output_rx.recv().await.unwrap();
                match output {
                    FuzzerIO::TestCase(v) => {
                        assert!(!v.is_empty());
                        println!("Receive output");
                    }
                    _ => {
                        unreachable!();
                    }
                }
                /*
                for _i in 0..worker_num {
                    let output = mutation_feedback_output_rx.recv().await.unwrap();
                    match output.get_type() {
                        FuzzerIOType::MutationFeedback => {
                            println!("yes {:?}", output);
                        }
                        _ => {
                            unreachable!();
                        }
                    }
                }
                */
            }
        }
    }
}

pub mod work_stealing_feedback {
    use super::NewBitFilter;
    use crate::datatype::ExecutionStatus;
    use crate::datatype::{Feedback, NewBit, TestCase};
    use crate::feedback::FeedbackCollector;
    use crate::frontend::work_stealing_frontend::WorkerImpl;
    use crate::frontend::FuzzerIO;
    use rayon::prelude::*;
    use std::sync::{Arc, RwLock};

    pub struct FeedbackCollectorWorker {
        collector: super::BitmapCollector,
        transformer: FeedbackCollectorTransformer,
    }

    impl WorkerImpl for FeedbackCollectorWorker {
        fn handle_inputs(&mut self, input: FuzzerIO) -> Option<Vec<FuzzerIO>> {
            let mut result = Vec::new();
            match input {
                FuzzerIO::Feedback(data) => {
                    self.collector.process_feedback(data);
                    if let Some(test_cases) = self
                        .transformer
                        .transform(self.collector.get_interesting_test_cases(None))
                    {
                        result.push(test_cases);
                    }
                    // self.collector.get_mutation_feedbacks(None);
                    // self.collector.get_test_case_feedbacks(None);
                    result.push(FuzzerIO::MutationFeedback(
                        self.collector.get_mutation_feedbacks(None),
                    ));
                    result.push(FuzzerIO::TestCaseFeedback(
                        self.collector.get_test_case_feedbacks(None),
                    ));
                    result.push(FuzzerIO::MonitorData(
                        self.collector.retrieve_monitor_data(),
                    ));
                    Some(result)
                }
                _ => {
                    unreachable!();
                }
            }
        }
    }

    impl FeedbackCollectorWorker {
        pub fn new(bitmap_size: usize, transformer: FeedbackCollectorTransformer) -> Self {
            Self {
                collector: super::BitmapCollector::new(bitmap_size),
                transformer,
            }
        }
    }

    pub struct FeedbackCollectorTransformer {
        filter: Arc<RwLock<NewBitFilter>>,
    }

    impl Clone for FeedbackCollectorTransformer {
        fn clone(&self) -> Self {
            Self {
                filter: self.filter.clone(),
            }
        }
    }

    impl FeedbackCollectorTransformer {
        pub fn new(bitmap_size: usize) -> Self {
            Self {
                filter: Arc::new(RwLock::new(NewBitFilter::new(bitmap_size))),
            }
        }

        // Filter out duplicated interesting test cases and leave other feedbacks intact.
        fn transform(&self, feedbacks: Vec<Feedback>) -> Option<FuzzerIO> {
            let new_test_cases: Vec<(Vec<NewBit>, TestCase)> = feedbacks
                .into_par_iter()
                .filter_map(|mut feedback| {
                    assert!(feedback.is_valid());
                    assert_eq!(feedback.get_status(), ExecutionStatus::Interesting);
                    assert!(feedback.contain_test_case());
                    if let Some(crate::datatype::FeedbackData::NewCoverage(new_coverage)) =
                        feedback.take_data()
                    {
                        self.filter
                            .read()
                            .unwrap()
                            .filter_old_bits(new_coverage)
                            .map(|new_bits| (new_bits, feedback.take_test_case().unwrap()))
                    } else {
                        unreachable!();
                    }
                })
                .collect::<Vec<_>>();

            let result = FuzzerIO::TestCase(
                new_test_cases
                    .into_iter()
                    .filter_map(|(new_bits, test_case)| {
                        let mut filter = self.filter.write().unwrap();
                        if filter.try_update_bits(&new_bits) {
                            Some(test_case)
                        } else {
                            None
                        }
                    })
                    .collect::<Vec<_>>(),
            );

            if !result.is_empty() {
                /*
                let byte_count = { self.filter.read().unwrap().byte_count() };
                println!(
                    "Map density: {}/65536, {:.2}",
                    byte_count,
                    byte_count as f64 / 63356f64 * 100f64
                );
                */
                Some(result)
            } else {
                None
            }
        }
    }
}
