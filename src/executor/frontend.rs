use crate::datatype::{Feedback, TestCase};
use crate::executor::shmem::ShMem;
use crate::executor::*;
use crate::frontend::simple_frontend::{
    AsyncFrontend, AsyncFrontendChannel, AsyncReceiver, AsyncSender, BasicFrontend,
};
use crate::frontend::Worker;
use crate::monitor::Monitor;
use async_trait::async_trait;
use std::sync::{Arc, Mutex};

pub type ForkServerExecutorResultPool = Arc<Mutex<Vec<Feedback>>>;
pub type ForkServerExecutorWorkerMutex = Arc<Mutex<ForkServerExecutorWorker>>;
pub type ForkServerExecutorWorkerPool = Vec<ForkServerExecutorWorkerMutex>;

pub struct ForkServerExecutorWorker {
    executor: ForkServerExecutor<BitmapTracer>,
}

pub struct ForkServerExecutorFrontend {
    worker_pool: ForkServerExecutorWorkerPool, // TODO(yongheng): Lock this with read/write lock
    bitmap_size: usize,
    args: Vec<String>,
    sender: Option<AsyncSender<Vec<TestCase>>>,
    output_receiver: Option<AsyncReceiver<Vec<Feedback>>>,
    output_sender: Option<AsyncSender<Vec<Feedback>>>,
    working_dir: Option<String>,
    timeout: u64,
}

impl ForkServerExecutorWorker {
    pub fn new(
        bitmap_size: usize,
        args: Vec<String>,
        timeout: u64,
        working_dir: Option<String>,
    ) -> Self {
        let bitmap_tracer = BitmapTracer::new(bitmap_size);
        bitmap_tracer.bitmap.write_to_env("__AFL_SHM_ID").unwrap();
        match ForkServerExecutor::new(bitmap_tracer, args, working_dir, timeout, true) {
            Ok(executor) => ForkServerExecutorWorker { executor },
            Err(e) => {
                panic!("ForkServerExecutorWorker::new failed: {}", e);
            }
        }
    }
}

impl Worker for ForkServerExecutorWorker {
    type Input = TestCase;
    type Output = Feedback;

    fn handle_one_input(&mut self, input: Vec<TestCase>) -> Vec<Feedback> {
        let output_len = input.len();
        let feedbacks = input
            .into_iter()
            .map(|single_input| self.executor.execute(single_input))
            .collect::<Vec<Feedback>>();
        crate::monitor::get_monitor()
            .read()
            .unwrap()
            .receive_statistics(serde_json::json!({ "exec": output_len as u64 }));
        feedbacks
    }
}

impl ForkServerExecutorFrontend {
    #[allow(dead_code)]
    pub fn new(
        bitmap_size: usize,
        args: Vec<String>,
        timeout: u64,
        working_dir: Option<String>,
    ) -> Self {
        ForkServerExecutorFrontend {
            worker_pool: Vec::default(),
            bitmap_size,
            args,
            sender: None,
            output_receiver: None,
            output_sender: None,
            working_dir,
            timeout,
        }
    }
}

impl BasicFrontend for ForkServerExecutorFrontend {
    type Worker = ForkServerExecutorWorker;
    type Output = Feedback;

    fn create_worker(&mut self, num: u32) {
        for _i in 0..num {
            self.worker_pool
                .push(Arc::new(Mutex::new(ForkServerExecutorWorker::new(
                    self.bitmap_size,
                    self.args.clone(),
                    self.timeout,
                    self.working_dir.clone(),
                ))));
        }
    }

    fn output_transfrom(&self, input: Vec<Feedback>) -> Vec<Feedback> {
        input
    }
    crate::frontend_default!();
}

impl AsyncFrontendChannel for ForkServerExecutorFrontend {
    crate::async_frontend_default!();
}

#[async_trait]
impl AsyncFrontend for ForkServerExecutorFrontend {}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::util;

    #[test]
    #[serial_test::serial]
    fn forkserver_executor_frontend_add_delete_worker_correctly() {
        const MAP_SIZE: usize = 65536;
        let args = vec![util::get_test_bin_by_name("simple_prog").unwrap()];
        let mut toy_frontend = ForkServerExecutorFrontend::new(MAP_SIZE, args, 20, None);
        toy_frontend.create_worker(15);

        assert!(toy_frontend.worker_pool.len() == 15);
        toy_frontend.delete_worker(5);
        assert!(toy_frontend.worker_pool.len() == 10);
    }

    #[tokio::test]
    #[serial_test::serial]
    #[should_panic]
    async fn forkserver_executor_fronted_panic_when_no_worker_is_created() {
        const MAP_SIZE: usize = 65536;
        let args = vec![util::get_test_bin_by_name("simple_prog").unwrap()];
        let toy_frontend = ForkServerExecutorFrontend::new(MAP_SIZE, args, 20, None);
        toy_frontend
            .handle_inputs(vec![TestCase::new(vec![0, 1], 0)])
            .await;
    }
}

pub mod new_executor_frontend {
    use crate::executor::shmem::ShMem;
    use crate::executor::*;
    use crate::frontend::load_balance_frontend::*;
    use crate::frontend::FuzzerIO;

    pub struct ExecutorWorker {
        executor: super::ForkServerExecutor<super::BitmapTracer>,
        execute_len: usize,
    }

    impl Worker for ExecutorWorker {
        fn handle_inputs(&mut self, input: FuzzerIO) -> Option<Vec<FuzzerIO>> {
            match input {
                FuzzerIO::TestCase(data) => {
                    let len = data.len();
                    let feedbacks = data
                        .into_iter()
                        .map(|test_case| self.executor.execute(test_case))
                        .collect();
                    let mut result = vec![FuzzerIO::Feedback(feedbacks)];
                    //if self.execute_len >= 1000 {
                    result.push(FuzzerIO::MonitorData(vec![serde_json::json!({
                        "exec": len as u64
                    })]));
                    self.execute_len = 0;
                    //}
                    Some(result)
                }
                _ => {
                    println!("{:?}", input.get_type());
                    unreachable!();
                }
            }
        }
    }
    impl ExecutorWorker {
        pub fn new(
            bitmap_size: usize,
            args: Vec<String>,
            timeout: u64,
            working_dir: Option<String>,
        ) -> Self {
            let bitmap_tracer = super::BitmapTracer::new(bitmap_size);
            bitmap_tracer.bitmap.write_to_env("__AFL_SHM_ID").unwrap();

            let executor =
                super::ForkServerExecutor::new(bitmap_tracer, args, working_dir, timeout, true)
                    .ok()
                    .unwrap();
            Self {
                executor,
                execute_len: 0,
            }
        }
    }

    pub struct ExecutorTransformer {}

    impl Transformer for ExecutorTransformer {}

    pub type ExecutorFrontend = Frontend<ExecutorWorker, ExecutorTransformer, DummySyncSource>;

    #[cfg(test)]
    mod tests {
        use super::*;
        use crate::datatype::TestCase;
        use crate::frontend::FuzzerIOType;
        use crate::util;

        #[tokio::test(flavor = "multi_thread", worker_threads = 5)]
        #[serial_test::serial]
        async fn do_testing() {
            let bitmap_size: usize = 65536;
            let args = vec![util::get_test_bin_by_name("simple_prog").unwrap()];
            let timeout = 20;
            let working_dir: Option<String> = None;
            let closure = move || {
                ExecutorWorker::new(bitmap_size, args.clone(), timeout, working_dir.clone())
            };
            let mut frontend: ExecutorFrontend = ExecutorFrontend::new(Box::new(closure));
            frontend.create_worker(5);
            let (input_sender, input_rx) = frontend_channel(100);
            let (output_sender, mut output_rx) = frontend_channel(100);
            let (monitor_output_sender, _monitor_output_rx) = frontend_channel(100);
            frontend.register_input_handler(Receiver::AsyncExclusive(input_rx));
            frontend.register_output_handler(
                FuzzerIOType::Feedback,
                Sender::AsyncExclusive(output_sender),
            );
            frontend.register_output_handler(
                FuzzerIOType::MonitorData,
                Sender::AsyncExclusive(monitor_output_sender),
            );

            frontend.run();

            let test_case = TestCase::new(String::from("select 1;").as_bytes().to_vec(), 0);
            let test_cases = (0..100)
                .map(|_x| test_case.clone())
                .collect::<Vec<TestCase>>();
            for _i in 0..10 {
                input_sender
                    .send(FuzzerIO::TestCase(test_cases.clone()))
                    .await
                    .unwrap();
                tokio::time::sleep(tokio::time::Duration::from_millis(5)).await;
                let output = output_rx.recv().await.unwrap();
                match output.get_type() {
                    FuzzerIOType::Feedback => {
                        println!("Receive output");
                    }
                    _ => {
                        unreachable!();
                    }
                }
            }
        }
    }
}

pub mod work_stealing_executor_frontend {
    use crate::{
        executor::{shmem::ShMem, Executor},
        frontend::{work_stealing_frontend::*, FuzzerIO},
    };

    pub struct ExecutorWorker {
        executor: super::ForkServerExecutor<super::BitmapTracer>,
        execute_len: usize,
    }

    impl WorkerImpl for ExecutorWorker {
        fn handle_inputs(&mut self, input: FuzzerIO) -> Option<Vec<FuzzerIO>> {
            match input {
                FuzzerIO::TestCase(data) => {
                    let len = data.len();
                    let feedbacks = data
                        .into_iter()
                        .map(|test_case| self.executor.execute(test_case))
                        .collect();
                    let mut result = vec![FuzzerIO::Feedback(feedbacks)];
                    //if self.execute_len >= 1000 {
                    result.push(FuzzerIO::MonitorData(vec![serde_json::json!({
                        "exec": len as u64
                    })]));
                    self.execute_len = 0;
                    //}
                    Some(result)
                }
                _ => {
                    println!("{:?}", input.get_type());
                    unreachable!();
                }
            }
        }
    }

    impl ExecutorWorker {
        pub fn new(
            bitmap_size: usize,
            args: Vec<String>,
            timeout: u64,
            working_dir: Option<String>,
        ) -> Self {
            let bitmap_tracer = super::BitmapTracer::new(bitmap_size);
            bitmap_tracer.bitmap.write_to_env("__AFL_SHM_ID").unwrap();

            let executor =
                super::ForkServerExecutor::new(bitmap_tracer, args, working_dir, timeout, true)
                    .ok()
                    .unwrap();
            Self {
                executor,
                execute_len: 0,
            }
        }
    }

    #[cfg(test)]
    mod tests {
        use super::*;
        use crate::datatype::TestCase;
        use crate::frontend::FuzzerIOType;
        use crate::util;

        #[tokio::test(flavor = "multi_thread", worker_threads = 5)]
        #[serial_test::serial]
        async fn do_testing() {
            let bitmap_size: usize = 65536;
            let args = vec![util::get_test_bin_by_name("simple_prog").unwrap()];
            let timeout = 20;
            let working_dir: Option<String> = None;
            let closure = move || {
                Worker::new(
                    Box::new(ExecutorWorker::new(
                        bitmap_size,
                        args.clone(),
                        timeout,
                        working_dir.clone(),
                    )),
                    0,
                )
            };
            let frontend: Frontend = FrontendBuilder::new()
                .worker_creator(Box::new(closure))
                .worker_num(2)
                .add_input_type(FuzzerIOType::TestCase)
                .add_output_type(FuzzerIOType::Feedback)
                .add_output_type(FuzzerIOType::MonitorData)
                .build()
                .unwrap();
            //WorkStealingFrontend::new(Box::new(closure));
            // frontend.create_worker(5);
            let (all_senders, all_receivers) = frontend.self_setup();

            assert_eq!(all_senders.len(), 1);
            assert_eq!(all_senders[0].len(), 2);
            assert_eq!(all_receivers.len(), 2);
            frontend.run();

            let test_case = TestCase::new(String::from("select 1;").as_bytes().to_vec(), 0);
            let test_cases = (0..100)
                .map(|_x| test_case.clone())
                .collect::<Vec<TestCase>>();

            let test_case_sender = all_senders[0][0].clone();
            let feedback_receiver = all_receivers[0][0].clone();
            let monitor_receiver = all_receivers[1][0].clone();
            assert!(!feedback_receiver.is_closed());
            assert!(!monitor_receiver.is_closed());
            assert!(!test_case_sender.is_closed());

            test_case_sender
                .send(FuzzerIO::TestCase(test_cases.clone()))
                .await
                .unwrap();
            assert!(!test_case_sender.is_closed());

            let feedback = feedback_receiver.recv().await.unwrap();
            let monitor_data = monitor_receiver.recv().await.unwrap();
            assert_eq!(feedback.get_type(), FuzzerIOType::Feedback);
            assert_eq!(monitor_data.get_type(), FuzzerIOType::MonitorData);
            tokio::time::sleep(tokio::time::Duration::from_millis(5)).await;
            test_case_sender
                .send(FuzzerIO::TestCase(test_cases.clone()))
                .await
                .unwrap();
            let feedback = feedback_receiver.recv().await.unwrap();
            let monitor_data = monitor_receiver.recv().await.unwrap();
            assert_eq!(feedback.get_type(), FuzzerIOType::Feedback);
            assert_eq!(monitor_data.get_type(), FuzzerIOType::MonitorData);
            tokio::time::sleep(tokio::time::Duration::from_millis(5)).await;
        }
    }
}
