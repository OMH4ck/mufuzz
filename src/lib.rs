use clap::Parser;
use executor::frontend::{ForkServerExecutorFrontend, ForkServerExecutorWorker};
use executor::rpc::{ExecutorClient, MyExecutor};
use feedback::frontend::{BitmapCollectorFrontend, BitmapCollectorWorker};
use feedback::rpc::{FeedbackCollectorClient, MyFeedbackCollector};
use feedback::FeedbackCollector;
use frontend::load_balance_frontend::{ChannelType, Sender};
use frontend::simple_frontend::*;
use frontend::work_stealing_frontend::ChannelIOType;
use frontend::*;
use load_balance_frontend::frontend_channel;
use minimizer::TestCaseMinimizer;
use monitor::stats::FuzzerInfo;
use monitor::Monitor;
use mutator::frontend::{BitFlipMutatorFrontend, BitFlipMutatorWorker};
use mutator::rpc::{MutatorClient, MyMutator};
use queue::frontend::{SimpleQueueManagerFrontend, SimpleQueueManagerWorker};
use queue::rpc::{MyQueueManager, QueueManagerClient};
use rpc::msg_type::{Empty, Number, TestCase, TestCases};
use std::sync::Arc;
use tokio::time::{sleep, Duration, Instant};

use executor::frontend::new_executor_frontend;
use executor::frontend::work_stealing_executor_frontend;
use feedback::frontend::new_feedback_frontend;
use mutator::frontend::new_mutator_frontend;
use queue::frontend::new_queue_frontend;

use std::net::SocketAddr;
use tokio::sync::mpsc::channel;

pub mod datatype;
pub mod executor;
pub mod feedback;
pub mod frontend;
pub mod minimizer;
pub mod monitor;
pub mod mutator;
pub mod queue;
pub mod rpc;
pub mod util;
use std::{env::VarError, fmt, io, num::ParseIntError};

#[derive(Debug)]
pub enum Error {
    /// Serialization error
    Serialize(String),
    /// Compression error
    //#[cfg(feature = "llmp_compression")]
    //Compression,
    /// File related error
    //#[cfg(feature = "std")]
    File(io::Error),
    /// Optional val was supposed to be set, but isn't.
    EmptyOptional(String),
    /// Key not in Map
    KeyNotFound(String),
    /// No elements in the current item
    Empty(String),
    /// End of iteration
    IteratorEnd(String),
    /// This is not supported (yet)
    NotImplemented(String),
    /// You're holding it wrong
    IllegalState(String),
    /// The argument passed to this method or function is not valid
    IllegalArgument(String),
    /// Forkserver related Error
    ForkServer(String),
    /// MOpt related Error
    MOpt(String),
    /// Shutting down, not really an error.
    ShuttingDown,
    /// Something else happened
    Unknown(String),
}

impl fmt::Display for Error {
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        match self {
            Self::Serialize(s) => {
                write!(f, "Error in Serialization: `{0}`", &s)
            }
            #[cfg(feature = "llmp_compression")]
            Self::Compression => write!(f, "Error in decompression"),
            #[cfg(feature = "std")]
            Self::File(err) => write!(f, "File IO failed: {:?}", &err),
            Self::EmptyOptional(s) => {
                write!(f, "Optional value `{0}` was not set", &s)
            }
            Self::KeyNotFound(s) => write!(f, "Key `{0}` not in Corpus", &s),
            Self::Empty(s) => write!(f, "No items in {0}", &s),
            Self::IteratorEnd(s) => {
                write!(f, "All elements have been processed in {0} iterator", &s)
            }
            Self::NotImplemented(s) => write!(f, "Not implemented: {0}", &s),
            Self::IllegalState(s) => write!(f, "Illegal state: {0}", &s),
            Self::IllegalArgument(s) => write!(f, "Illegal argument: {0}", &s),
            Self::ForkServer(s) => write!(f, "Forkserver : {0}", &s),
            Self::MOpt(s) => write!(f, "MOpt: {0}", &s),
            Self::ShuttingDown => write!(f, "Shutting down!"),
            Self::Unknown(s) => write!(f, "Unknown error: {0}", &s),
        }
    }
}

impl From<nix::Error> for Error {
    fn from(err: nix::Error) -> Self {
        Self::Unknown(format!("{:?}", err))
    }
}

/// Create an AFL Error from io Error
// #[cfg(feature = "std")]
impl From<io::Error> for Error {
    fn from(err: io::Error) -> Self {
        Self::File(err)
    }
}

impl From<ParseIntError> for Error {
    fn from(err: ParseIntError) -> Self {
        Self::Unknown(format!("Failed to parse Int: {:?}", err))
    }
}

impl From<VarError> for Error {
    fn from(err: VarError) -> Self {
        Self::Empty(format!("Could not get env var: {:?}", err))
    }
}

// Basic fuzzer config that we can parse from cmd.
#[derive(Parser, Debug, Clone)]
#[clap(author, version, about, long_about = None)]
pub struct FuzzerConfig {
    /// The command line to run the target
    #[clap(short, long)]
    pub cmd: String,

    /// If set, the fuzzer will stop when the exec num reaches max_exec
    #[clap(long, default_value_t = u64::MAX)]
    pub max_exec: u64,

    /// If set, the fuzzer will stop when the running time reaches max_fuzzing_time.
    #[clap(long, default_value_t = u64::MAX)]
    pub max_fuzzing_time: u64,

    /// Timeout in ms for each execution. Keep it small.
    #[clap(long, default_value_t = 20)]
    pub timeout: u64,

    #[clap(short, long, default_value_t = 1)]
    pub executor: u32,

    #[clap(short, long, default_value_t = 1)]
    pub queuemgr: u32,

    #[clap(short, long, default_value_t = 1)]
    pub feedback: u32,

    /// Number of cores the fuzzer can use
    #[clap(long, default_value = "")]
    pub config: String,

    /// Number of cores the fuzzer can use
    #[clap(long, default_value_t = 1)]
    pub core: u32,

    /// Path of initial corpus
    #[clap(short, long)]
    pub input: String,

    /// Path of initial corpus
    #[clap(short, long)]
    pub output: Option<String>,

    /// Size of the bitmap (if the fuzzer use code coverage as feedback)
    #[clap(long, default_value_t = 65536)]
    pub map_size: usize,

    /// 0 for local mode. 1 for rpc mode. 2 for rpc async mode.
    #[clap(long, default_value_t = 5)]
    pub mode: i32,
}

async fn connect_mutator_executor(
    mutator_addr: SocketAddr,
    executor_addr: SocketAddr,
) -> tokio::task::JoinHandle<()> {
    let mut mutator_client = MutatorClient::connect(format!("http://{}", mutator_addr))
        .await
        .unwrap();
    let mut executor_client = ExecutorClient::connect(format!("http://{}", executor_addr))
        .await
        .unwrap();

    let (tx, mut rx) = channel(5);

    tokio::spawn(async move {
        loop {
            let new_test_cases = mutator_client
                .get_mutated_test_cases(tonic::Request::new(Number { num: 10000 }))
                .await
                .unwrap()
                .into_inner();
            if tx.send(new_test_cases).await.is_err() {
                break;
            }
        }
    });
    tokio::spawn(async move {
        while let Some(new_test_cases) = rx.recv().await {
            executor_client
                .execute_test_cases(tonic::Request::new(new_test_cases))
                .await
                .unwrap();
        }
    })
}

async fn connect_queuemgr_mutator(queuemgr_addr: SocketAddr, mutator_addr: SocketAddr) {
    let mut mutator_client = MutatorClient::connect(format!("http://{}", mutator_addr))
        .await
        .unwrap();

    let mut queue_mgr_client = QueueManagerClient::connect(format!("http://{}", queuemgr_addr))
        .await
        .unwrap();

    let (tx, mut rx) = channel(5);

    tokio::spawn(async move {
        loop {
            let new_test_cases = queue_mgr_client
                .get_interesting_test_cases(tonic::Request::new(Number { num: 200 }))
                .await
                .unwrap()
                .into_inner();
            if tx.send(new_test_cases).await.is_err() {
                break;
            }
        }
    });
    tokio::spawn(async move {
        while let Some(new_test_cases) = rx.recv().await {
            mutator_client
                .mutate_test_cases(tonic::Request::new(new_test_cases))
                .await
                .unwrap();
        }
    });
}

async fn connect_executor_feedback_collector(
    executor_addr: SocketAddr,
    feedback_collector_addr: SocketAddr,
) {
    let mut executor_client = ExecutorClient::connect(format!("http://{}", executor_addr))
        .await
        .unwrap();

    let mut feedback_collector_client =
        FeedbackCollectorClient::connect(format!("http://{}", feedback_collector_addr))
            .await
            .unwrap();

    let (tx, mut rx) = channel(5);

    {
        tokio::spawn(async move {
            loop {
                let request = tonic::Request::new(Number { num: 10000 });
                let feedbacks = executor_client
                    .get_feedbacks(request)
                    .await
                    .unwrap()
                    .into_inner();
                tx.send(feedbacks).await.unwrap();
            }
        });
    }

    {
        tokio::spawn(async move {
            let monitor = monitor::get_monitor();
            while let Some(feedbacks) = rx.recv().await {
                let request = tonic::Request::new(feedbacks);
                feedback_collector_client
                    .check_feedbacks(request)
                    .await
                    .unwrap();
                let monitor_data = feedback_collector_client
                    .retrieve_monitor_data(tonic::Request::new(Empty::default()))
                    .await
                    .unwrap()
                    .into_inner()
                    .into();
                for v in monitor_data.into_iter() {
                    monitor.read().unwrap().receive_statistics(v);
                }
            }
        });
    }
}

async fn connect_feedback_collector_mutator(
    feedback_collector_addr: SocketAddr,
    mutator_addr: SocketAddr,
) {
    let mut feedback_collector_client =
        FeedbackCollectorClient::connect(format!("http://{}", feedback_collector_addr))
            .await
            .unwrap();

    let mut mutator_client = MutatorClient::connect(format!("http://{}", mutator_addr))
        .await
        .unwrap();

    let (tx, mut rx) = channel(5);

    {
        tokio::spawn(async move {
            loop {
                let request = tonic::Request::new(Number { num: 10000 });
                let feedbacks = feedback_collector_client
                    .get_mutation_feedbacks(request)
                    .await
                    .unwrap()
                    .into_inner();
                tx.send(feedbacks).await.unwrap();
            }
        });
    }

    {
        tokio::spawn(async move {
            while let Some(feedbacks) = rx.recv().await {
                let request = tonic::Request::new(feedbacks);
                mutator_client
                    .post_mutation_feedbacks(request)
                    .await
                    .unwrap();
            }
        });
    }
}

async fn connect_feedback_collector_queuemgr(
    feedback_collector_addr: SocketAddr,
    queuemgr_addr: SocketAddr,
) {
    let mut feedback_collector_client =
        FeedbackCollectorClient::connect(format!("http://{}", feedback_collector_addr))
            .await
            .unwrap();
    let mut new_feedback_collector_client = feedback_collector_client.clone();

    let mut queue_mgr_client = QueueManagerClient::connect(format!("http://{}", queuemgr_addr))
        .await
        .unwrap();
    let mut new_queue_mgr_client = queue_mgr_client.clone();

    let (tx, mut rx) = channel(5);
    let (feedback_tx, mut feedback_rx) = channel(5);

    {
        tokio::spawn(async move {
            loop {
                let request = tonic::Request::new(Number { num: 10000 });
                let feedbacks = new_feedback_collector_client
                    .get_test_case_feedbacks(request)
                    .await
                    .unwrap()
                    .into_inner();
                feedback_tx.send(feedbacks).await.unwrap();
            }
        });
    }

    {
        tokio::spawn(async move {
            while let Some(feedbacks) = feedback_rx.recv().await {
                let request = tonic::Request::new(feedbacks);
                new_queue_mgr_client
                    .post_test_case_feedbacks(request)
                    .await
                    .unwrap();
            }
        });
    }

    {
        tokio::spawn(async move {
            loop {
                let request = tonic::Request::new(Number { num: 10000 });
                let feedbacks = feedback_collector_client
                    .get_interesting_test_cases(request)
                    .await
                    .unwrap()
                    .into_inner();
                tx.send(feedbacks).await.unwrap();
            }
        });
    }

    {
        tokio::spawn(async move {
            while let Some(feedbacks) = rx.recv().await {
                let request = tonic::Request::new(feedbacks);
                queue_mgr_client
                    .post_interesting_test_cases(request)
                    .await
                    .unwrap();
            }
        });
    }
}

/// The most high-level functions for running the fuzzers.
/// All we need is a configured `FuzzerConfig`, which contains every information about the fuzzer we want to run.
pub async fn run_fuzzer_local_mode(fuzzer: FuzzerConfig) {
    monitor::stats::get_fuzzer_info()
        .lock()
        .unwrap()
        .sync_with_fuzzer_config(&fuzzer);

    let map_size: usize = fuzzer.map_size;
    let args = fuzzer
        .cmd
        .split_whitespace()
        .map(|s| s.to_string())
        .collect();
    let n_executor = fuzzer.executor.max(fuzzer.core);
    let n_feedback_collector = fuzzer.feedback.max(fuzzer.core);
    let n_queue_mgr = fuzzer.queuemgr.max(fuzzer.core);
    let n_mutator = fuzzer.queuemgr.max(fuzzer.core);

    let mut forkserver_executor_frontend =
        ForkServerExecutorFrontend::new(map_size, args, fuzzer.timeout, fuzzer.output);
    let mut bitmap_collector_frontend = BitmapCollectorFrontend::new(map_size);
    let mut simple_queue_manager_frontend = SimpleQueueManagerFrontend::new();
    let mut mutator_frontend = BitFlipMutatorFrontend::new();

    forkserver_executor_frontend.create_worker(n_executor);
    forkserver_executor_frontend.run().await;
    bitmap_collector_frontend.create_worker(n_feedback_collector);
    bitmap_collector_frontend.run().await;
    simple_queue_manager_frontend.create_worker(n_queue_mgr);
    simple_queue_manager_frontend.run().await;
    mutator_frontend.create_worker(n_mutator);
    mutator_frontend.run().await;

    let interesting_queues = util::read_corpus(fuzzer.input).unwrap();

    simple_queue_manager_frontend
        .handle_inputs(interesting_queues)
        .await;

    let mut fuzzer_stat = FuzzerInfo::new();
    fuzzer_stat.set_cmd(fuzzer.cmd).set_total_coverage(1000);
    while fuzzer_stat.get_exec() < fuzzer.max_exec {
        let to_be_mutated_inputs = simple_queue_manager_frontend.get_results(None).await;
        mutator_frontend.handle_inputs(to_be_mutated_inputs).await;

        let new_inputs = mutator_frontend.get_results(None).await;
        fuzzer_stat.add_exec(new_inputs.len() as u64);

        forkserver_executor_frontend.handle_inputs(new_inputs).await;

        let interesting_results = forkserver_executor_frontend.get_results(None).await;

        bitmap_collector_frontend
            .handle_inputs(interesting_results)
            .await;

        let new_interesting_inputs = bitmap_collector_frontend.get_results(None).await;
        fuzzer_stat.add_coverage(new_interesting_inputs.len() as u64);
        simple_queue_manager_frontend
            .handle_inputs(new_interesting_inputs)
            .await;
        let mutation_feedbacks = bitmap_collector_frontend.get_mutation_feedbacks(None).await;
        mutator_frontend.process_feedbacks(mutation_feedbacks).await;
        simple_queue_manager_frontend
            .process_feedbacks(
                bitmap_collector_frontend
                    .get_test_case_feedbacks(None)
                    .await,
            )
            .await;

        fuzzer_stat.simple_show();
    }
}

// Run a fuzzer instance of single core like traditional afl.
pub fn run_single_fuzzer_mode(fuzzer: FuzzerConfig, flag: bool) {
    monitor::get_monitor()
        .write()
        .unwrap()
        .get_fuzzer_info_mut()
        .sync_with_fuzzer_config(&fuzzer);

    let map_size: usize = fuzzer.map_size;
    let args = fuzzer
        .cmd
        .split_whitespace()
        .map(|s| s.to_string())
        .collect();
    let mut executor = ForkServerExecutorWorker::new(map_size, args, fuzzer.timeout, fuzzer.output);
    let mut queue_manager = SimpleQueueManagerWorker::new();
    let mut feedback_collector = BitmapCollectorWorker::new(map_size);
    let feedback_collector_frontend = BitmapCollectorFrontend::new(map_size);
    let mut mutator = BitFlipMutatorWorker::new();
    //let mutator_frontend = BitFlipMutatorFrontend::new();

    let init_seeds = util::read_corpus(fuzzer.input).unwrap();
    queue_manager.handle_one_input(init_seeds);

    let monitor = monitor::get_monitor();
    let monitor = monitor.read().unwrap();
    let now = Instant::now();
    let mut prev = 0;
    let mut loop_prev = 0;
    loop {
        let inputs_to_mutate = queue_manager.select_interesting_inputs(20);
        let select_time = now.elapsed().as_micros() - prev;
        let select_len = inputs_to_mutate.len();
        prev += select_time;
        /*
        let new_inputs =
            mutator_frontend.output_transfrom(
                mutator.handle_one_input(inputs_to_mutate)
            );
            */
        let mutate_len = inputs_to_mutate.len();
        let new_inputs = mutator.handle_one_input(inputs_to_mutate);
        let mutate_time = now.elapsed().as_micros() - prev;
        prev += mutate_time;

        let execute_len = new_inputs.len();
        let feedbacks = executor.handle_one_input(new_inputs);
        let execute_time = now.elapsed().as_micros() - prev;
        prev += execute_time;

        let feedback_len = feedbacks.len();
        let interesting_inputs = feedback_collector_frontend
            .output_transfrom(feedback_collector.handle_one_input(feedbacks));
        let test_case_feedbacks = feedback_collector.collector.get_test_case_feedbacks(None);
        queue_manager.process_feedbacks(&test_case_feedbacks);
        let mutation_feedbacks = feedback_collector.collector.get_mutation_feedbacks(None);
        mutator.process_feedbacks(&mutation_feedbacks);
        let feedback_time = now.elapsed().as_micros() - prev;
        prev += feedback_time;

        let accept_input_len = interesting_inputs.len();
        queue_manager.handle_one_input(interesting_inputs);
        let accept_input_time = now.elapsed().as_micros() - prev;
        prev += accept_input_time;
        for v in feedback_collector.retrieve_monitor_data() {
            monitor.receive_statistics(v);
        }
        let monitor_receive_time = now.elapsed().as_micros() - prev;
        prev += monitor_receive_time;

        if flag && now.elapsed().as_secs() - loop_prev >= 2 {
            println!("{select_time}, {select_len}, {mutate_time}, {mutate_len}, {execute_time}, {execute_len}, {feedback_time}, {feedback_len}, {accept_input_time}, {accept_input_len}, {monitor_receive_time}");
            loop_prev = now.elapsed().as_secs();
            monitor.show_statistics();
        }
    }
}

pub async fn run_fuzzer_local_async_mode(fuzzer: FuzzerConfig) {
    monitor::get_monitor()
        .write()
        .unwrap()
        .get_fuzzer_info_mut()
        .sync_with_fuzzer_config(&fuzzer);

    let map_size: usize = fuzzer.map_size;
    let args = fuzzer
        .cmd
        .split_whitespace()
        .map(|s| s.to_string())
        .collect();

    let n_executor = if fuzzer.executor != 1 {
        fuzzer.executor
    } else {
        fuzzer.executor.max(fuzzer.core)
    };
    let n_feedback_collector = fuzzer.feedback.max(fuzzer.core);
    let n_queue_mgr = fuzzer.queuemgr.max(fuzzer.core);
    let n_mutator = fuzzer.executor.max(fuzzer.core);

    let mut forkserver_executor_frontend =
        ForkServerExecutorFrontend::new(map_size, args, fuzzer.timeout, fuzzer.output);
    let mut bitmap_collector_frontend = BitmapCollectorFrontend::new(map_size);
    let mut simple_queue_manager_frontend = SimpleQueueManagerFrontend::new();
    let mut mutator_frontend = BitFlipMutatorFrontend::new();

    forkserver_executor_frontend.create_worker(n_executor);
    bitmap_collector_frontend.create_worker(n_feedback_collector);
    simple_queue_manager_frontend.create_worker(n_queue_mgr);
    mutator_frontend.create_worker(n_mutator);

    forkserver_executor_frontend.run().await;
    bitmap_collector_frontend.run().await;
    simple_queue_manager_frontend.run().await;
    mutator_frontend.run().await;

    let interesting_queues = util::read_corpus(fuzzer.input).unwrap();

    simple_queue_manager_frontend
        .handle_inputs(interesting_queues)
        .await;

    let forkserver_executor_frontend = Arc::new(forkserver_executor_frontend);
    let bitmap_collector_frontend = Arc::new(bitmap_collector_frontend);
    let simple_queue_manager_frontend = Arc::new(simple_queue_manager_frontend);
    let mutator_frontend = Arc::new(mutator_frontend);

    let forkserver_executor_frontend_output = forkserver_executor_frontend.clone();
    let bitmap_collector_frontend_input = bitmap_collector_frontend.clone();
    let mut handles = Vec::default();
    handles.push(tokio::spawn(async move {
        loop {
            let output = forkserver_executor_frontend_output.get_results(None).await;
            bitmap_collector_frontend_input.handle_inputs(output).await;
        }
    }));

    let bitmap_collector_frontend_output = bitmap_collector_frontend.clone();
    let simple_queue_manager_frontend_input = simple_queue_manager_frontend.clone();
    handles.push(tokio::spawn(async move {
        let monitor = crate::monitor::get_monitor();
        loop {
            let outputs = bitmap_collector_frontend_output.get_results(None).await;
            simple_queue_manager_frontend_input
                .handle_inputs(outputs)
                .await;
            let data = bitmap_collector_frontend_output
                .retrieve_monitor_data()
                .await;
            for v in data.into_iter() {
                monitor.read().unwrap().receive_statistics(v);
            }
        }
    }));

    let bitmap_collector_frontend_test_case_output = bitmap_collector_frontend.clone();
    let simple_queue_manager_frontend_feedback_input = simple_queue_manager_frontend.clone();
    handles.push(tokio::spawn(async move {
        loop {
            let outputs = bitmap_collector_frontend_test_case_output
                .get_test_case_feedbacks(None)
                .await;
            simple_queue_manager_frontend_feedback_input
                .process_feedbacks(outputs)
                .await;
        }
    }));

    let bitmap_collector_frontend_mutation_output = bitmap_collector_frontend.clone();
    let mutator_frontend_feedback_input = mutator_frontend.clone();
    handles.push(tokio::spawn(async move {
        loop {
            let outputs = bitmap_collector_frontend_mutation_output
                .get_mutation_feedbacks(None)
                .await;
            mutator_frontend_feedback_input
                .process_feedbacks(outputs)
                .await;
        }
    }));

    let mutator_frontend_input = mutator_frontend.clone();
    let simple_queue_manager_frontend_output = simple_queue_manager_frontend.clone();

    handles.push(tokio::spawn(async move {
        loop {
            let outputs = simple_queue_manager_frontend_output
                .get_results(Some(200))
                .await;
            mutator_frontend_input.handle_inputs(outputs).await;
        }
    }));

    let mutator_frontend_output = mutator_frontend.clone();
    let forkserver_executor_frontend_input = forkserver_executor_frontend.clone();

    handles.push(tokio::spawn(async move {
        loop {
            let outputs = mutator_frontend_output.get_results(Some(10000)).await;
            forkserver_executor_frontend_input
                .handle_inputs(outputs)
                .await;
        }
    }));

    handles.push(tokio::spawn(async move {
        let monitor = monitor::get_monitor();
        loop {
            sleep(Duration::from_millis(2000)).await;
            monitor.read().unwrap().show_statistics();
        }
    }));
    futures::future::join_all(handles).await;
}

pub async fn run_fuzzer_rpc_async_mode(fuzzer: FuzzerConfig) {
    monitor::get_monitor()
        .write()
        .unwrap()
        .get_fuzzer_info_mut()
        .sync_with_fuzzer_config(&fuzzer);

    let map_size: usize = fuzzer.map_size;
    let args = fuzzer
        .cmd
        .split_whitespace()
        .map(|s| s.to_string())
        .collect();
    let n_executor = if fuzzer.executor != 1 {
        fuzzer.executor
    } else {
        fuzzer.executor.max(fuzzer.core)
    };
    let n_feedback_collector = fuzzer.feedback.max(fuzzer.core);
    let n_queue_mgr = fuzzer.queuemgr.max(fuzzer.core);
    let n_mutator = fuzzer.executor.max(fuzzer.core);

    let executor_addr = MyExecutor::run_service(
        None,
        n_executor,
        args,
        map_size,
        fuzzer.timeout,
        fuzzer.output,
    )
    .await;
    let mutator_addr = MyMutator::run_service(None, n_mutator).await;
    let feedback_collector_addr =
        MyFeedbackCollector::run_service(None, n_feedback_collector, map_size).await;
    let queue_mgr_addr = MyQueueManager::run_service(None, n_queue_mgr).await;
    {
        let mut queue_mgr_client =
            QueueManagerClient::connect(format!("http://{}", queue_mgr_addr))
                .await
                .unwrap();

        // Create initial corpus MAP_SIZE
        let corpus = util::read_corpus(fuzzer.input).unwrap();

        println!("We have {} corpus", corpus.len());

        let request = tonic::Request::new(TestCases {
            test_cases: corpus.into_iter().map(TestCase::from).collect(),
        });

        let response = queue_mgr_client.post_interesting_test_cases(request).await;
        assert!(response.is_ok());
    }
    connect_executor_feedback_collector(executor_addr, feedback_collector_addr).await;
    connect_feedback_collector_mutator(feedback_collector_addr, mutator_addr).await;
    connect_feedback_collector_queuemgr(feedback_collector_addr, queue_mgr_addr).await;
    connect_queuemgr_mutator(queue_mgr_addr, mutator_addr).await;
    connect_mutator_executor(mutator_addr, executor_addr).await;

    tokio::spawn(async move {
        let monitor = monitor::get_monitor();
        while monitor.read().unwrap().get_fuzzer_info().get_exec()
            < monitor.read().unwrap().get_fuzzer_info().get_max_exec()
        {
            sleep(Duration::from_millis(2000)).await;
            monitor.read().unwrap().show_statistics();
        }
    })
    .await
    .unwrap();
}

pub async fn run_fuzzer_rpc_mode(fuzzer: FuzzerConfig) {
    monitor::stats::get_fuzzer_info()
        .lock()
        .unwrap()
        .sync_with_fuzzer_config(&fuzzer);

    let map_size: usize = fuzzer.map_size;
    let args = fuzzer
        .cmd
        .split_whitespace()
        .map(|s| s.to_string())
        .collect();
    let n_executor = fuzzer.executor.max(fuzzer.core);
    let n_feedback_collector = fuzzer.feedback.max(fuzzer.core);
    let n_queue_mgr = fuzzer.queuemgr.max(fuzzer.core);
    let n_mutator = fuzzer.executor.max(fuzzer.core);

    let mut executor_client = ExecutorClient::connect(format!(
        "http://{}",
        MyExecutor::run_service(
            None,
            n_executor,
            args,
            map_size,
            fuzzer.timeout,
            fuzzer.output
        )
        .await
    ))
    .await
    .unwrap();

    let mut feedback_collector_client = FeedbackCollectorClient::connect(format!(
        "http://{}",
        MyFeedbackCollector::run_service(None, n_feedback_collector, map_size).await
    ))
    .await
    .unwrap();

    let mut queue_mgr_client = QueueManagerClient::connect(format!(
        "http://{}",
        MyQueueManager::run_service(None, n_queue_mgr).await
    ))
    .await
    .unwrap();

    let mut mutator_client = MutatorClient::connect(format!(
        "http://{}",
        MyMutator::run_service(None, n_mutator).await
    ))
    .await
    .unwrap();

    // Create initial corpus MAP_SIZE
    let corpus = util::read_corpus(fuzzer.input).unwrap();

    let request = tonic::Request::new(TestCases {
        test_cases: corpus.into_iter().map(TestCase::from).collect(),
    });

    let response = queue_mgr_client.post_interesting_test_cases(request).await;
    assert!(response.is_ok());

    let mut fuzzer_stat = FuzzerInfo::new();
    fuzzer_stat.set_cmd(fuzzer.cmd).set_total_coverage(1000);
    while fuzzer_stat.get_exec() < fuzzer.max_exec {
        //while interesting_input_num < 50 {
        // Get new test_cases for execution.
        let request = tonic::Request::new(Number { num: 200 });
        let to_be_mutated_test_cases = queue_mgr_client
            .get_interesting_test_cases(request)
            .await
            .unwrap()
            .into_inner();
        let request = tonic::Request::new(to_be_mutated_test_cases);
        let response = mutator_client.mutate_test_cases(request).await;
        assert!(response.is_ok());
        let request = tonic::Request::new(Number { num: 10000 });
        let new_test_cases = mutator_client
            .get_mutated_test_cases(request)
            .await
            .unwrap()
            .into_inner();

        fuzzer_stat.add_exec(new_test_cases.test_cases.len() as u64);

        // Execute inputs.
        let request = tonic::Request::new(new_test_cases);
        let response = executor_client.execute_test_cases(request).await;
        assert!(response.is_ok());

        // Get feedbacks from executors.
        let request = tonic::Request::new(Number { num: 10000 });
        let feedbacks = executor_client
            .get_feedbacks(request)
            .await
            .unwrap()
            .into_inner();

        // Handle feedback
        let request = tonic::Request::new(feedbacks);
        let response = feedback_collector_client.check_feedbacks(request).await;
        assert!(response.is_ok());

        // Get interesting inputs
        let request = tonic::Request::new(Number { num: 10000 });
        let feedbacks = feedback_collector_client
            .get_interesting_test_cases(request)
            .await
            .unwrap()
            .into_inner();
        // fuzzer_stat.add_coverage(test_cases.test_cases.len() as u64);

        // Post them to queue manager
        let request = tonic::Request::new(feedbacks);
        let response = queue_mgr_client.post_interesting_test_cases(request).await;
        assert!(response.is_ok());

        let request = tonic::Request::new(Number { num: 10000 });
        let feedbacks = feedback_collector_client
            .get_mutation_feedbacks(request)
            .await
            .unwrap()
            .into_inner();
        let request = tonic::Request::new(feedbacks);
        let response = mutator_client.post_mutation_feedbacks(request).await;
        assert!(response.is_ok());

        let request = tonic::Request::new(Number { num: 10000 });
        let feedbacks = feedback_collector_client
            .get_test_case_feedbacks(request)
            .await
            .unwrap()
            .into_inner();
        let request = tonic::Request::new(feedbacks);
        let response = queue_mgr_client.post_test_case_feedbacks(request).await;
        assert!(response.is_ok());

        fuzzer_stat.simple_show();
    }
}

pub async fn run_fuzzer_local_async_lock_free_mode(fuzzer: FuzzerConfig) {
    monitor::get_monitor()
        .write()
        .unwrap()
        .get_fuzzer_info_mut()
        .sync_with_fuzzer_config(&fuzzer);

    let map_size: usize = fuzzer.map_size;
    let args = fuzzer
        .cmd
        .split_whitespace()
        .map(|s| s.to_string())
        .collect::<Vec<String>>();

    let n_executor = if fuzzer.executor != 1 {
        fuzzer.executor
    } else {
        fuzzer.executor.max(fuzzer.core)
    };
    let n_feedback_collector = fuzzer.feedback.max(fuzzer.core);
    let n_queue_manager = fuzzer.queuemgr.max(fuzzer.core);
    let n_mutator = fuzzer.executor.max(fuzzer.core);

    let channel_cap = 500;

    // let copy_args = args.clone();
    let closure = move || {
        new_executor_frontend::ExecutorWorker::new(
            map_size,
            args.clone(),
            fuzzer.timeout,
            fuzzer.output.clone(),
        )
    };
    let mut executor_frontend = new_executor_frontend::ExecutorFrontend::new(Box::new(closure));
    executor_frontend.create_worker(n_executor);

    // TODO(yongheng): Fix this to avoid duplicate interesting inputs.
    let init_seeds = util::read_corpus(fuzzer.input).unwrap();

    let closure = move || {
        // TODO: Fix the testcase minimizer
        new_queue_frontend::QueueManagerWorker::new_with_seeds(Some(init_seeds.clone()), None)
    };
    let mut queue_manager_frontend =
        new_queue_frontend::QueueManagerFrontend::new(Box::new(closure));
    queue_manager_frontend.create_worker(n_queue_manager);

    let closure = new_mutator_frontend::BitFlipMutatorWorker::new;
    let mut mutator_frontend = new_mutator_frontend::BitFlipMutatorFrontend::new(Box::new(closure));
    let transformer = Arc::new(new_mutator_frontend::MutatorTransformer::new());
    mutator_frontend.set_transformer(Some(transformer));
    mutator_frontend.create_worker(n_mutator);

    let closure = move || new_feedback_frontend::FeedbackCollectorWorker::new(map_size);
    let mut feedback_frontend =
        new_feedback_frontend::FeedbackCollectorFrontend::new(Box::new(closure));
    let transformer = Arc::new(new_feedback_frontend::FeedbackCollectorTransformer::new(
        map_size,
    ));
    feedback_frontend.set_transformer(Some(transformer));
    feedback_frontend.create_worker(n_feedback_collector);

    load_balance_frontend::connect_component(
        &mut feedback_frontend,
        &mut mutator_frontend,
        FuzzerIOType::MutationFeedback,
        ChannelType::NormalOutput,
        Some(new_mutator_frontend::MutatorSyncSource::default()),
    );

    load_balance_frontend::connect_component(
        &mut feedback_frontend,
        &mut queue_manager_frontend,
        FuzzerIOType::TestCaseFeedback,
        ChannelType::NormalOutput,
        Some(new_queue_frontend::QueueSyncSource::default()),
    );

    load_balance_frontend::connect_component(
        &mut mutator_frontend,
        &mut executor_frontend,
        FuzzerIOType::TestCase,
        ChannelType::LessContentionOutput,
        None,
    );

    load_balance_frontend::connect_component(
        &mut executor_frontend,
        &mut feedback_frontend,
        FuzzerIOType::Feedback,
        ChannelType::NormalOutput,
        None,
    );

    load_balance_frontend::connect_component(
        &mut queue_manager_frontend,
        &mut mutator_frontend,
        FuzzerIOType::TestCase,
        ChannelType::NonBlockingOutput,
        None,
    );

    let (monitor_data_sender, mut monitor_data_output_rx) = frontend_channel(channel_cap);

    executor_frontend.register_output_handler(
        FuzzerIOType::MonitorData,
        Sender::AsyncExclusive(monitor_data_sender.clone()),
    );

    load_balance_frontend::connect_component(
        &mut feedback_frontend,
        &mut queue_manager_frontend,
        FuzzerIOType::TestCase,
        ChannelType::NormalOutput,
        None,
    );

    /*
    for seed in init_seeds {
        sender.send(seed).await.unwrap();
    }
    */

    feedback_frontend.register_output_handler(
        FuzzerIOType::MonitorData,
        Sender::AsyncExclusive(monitor_data_sender.clone()),
    );

    queue_manager_frontend.register_output_handler(
        FuzzerIOType::MonitorData,
        Sender::AsyncExclusive(monitor_data_sender),
    );

    queue_manager_frontend.set_name(String::from("queue manager"));
    mutator_frontend.set_name(String::from("mutator"));
    executor_frontend.set_name(String::from("executor"));
    feedback_frontend.set_name(String::from("feedback"));

    queue_manager_frontend.run();
    mutator_frontend.run();
    executor_frontend.run();
    feedback_frontend.run();

    tokio::spawn(async move {
        let monitor = monitor::get_monitor();
        while let Some(value) = monitor_data_output_rx.recv().await {
            //println!("Receive monitor data");
            match value {
                FuzzerIO::MonitorData(vals) => {
                    for val in vals {
                        monitor.read().unwrap().receive_statistics(val);
                    }
                }
                _ => {
                    unreachable!();
                }
            }
        }

        unreachable!();
    });

    tokio::spawn(async move {
        let monitor = monitor::get_monitor();
        monitor.write().unwrap().reset_start_time();
        loop {
            sleep(Duration::from_millis(2000)).await;
            monitor.read().unwrap().show_statistics();
        }
    })
    .await
    .unwrap();
}

pub async fn run_fuzzer_local_work_stealing_mode(fuzzer: FuzzerConfig) {
    monitor::get_monitor()
        .write()
        .unwrap()
        .get_fuzzer_info_mut()
        .sync_with_fuzzer_config(&fuzzer);

    let map_size: usize = fuzzer.map_size;
    let args = fuzzer
        .cmd
        .split_whitespace()
        .map(|s| s.to_string())
        .collect::<Vec<String>>();

    let n_executor = if fuzzer.executor != 1 {
        fuzzer.executor
    } else {
        fuzzer.executor.max(fuzzer.core)
    };
    let n_feedback_collector = fuzzer.feedback.max(fuzzer.core);
    let n_queue_manager = fuzzer.queuemgr.max(fuzzer.core);
    let n_mutator = fuzzer.executor.max(fuzzer.core);

    let copy_args = args.clone();
    let closure = move || {
        work_stealing_frontend::Worker::new(
            Box::new(work_stealing_executor_frontend::ExecutorWorker::new(
                map_size,
                copy_args.clone(),
                fuzzer.timeout,
                fuzzer.output.clone(),
            )),
            0,
        )
    };

    let executor_frontend = work_stealing_frontend::FrontendBuilder::new()
        .worker_num(n_executor.try_into().unwrap())
        .worker_creator(Box::new(closure))
        .add_input_type(FuzzerIOType::TestCase)
        .add_output_type(FuzzerIOType::Feedback)
        .add_output_type(FuzzerIOType::MonitorData)
        .name("executor")
        .build()
        .unwrap();

    let closure = move || {
        work_stealing_frontend::Worker::new(
            Box::new(crate::mutator::frontend::work_stealing_mutator::MutatorWorker::new()),
            0,
        )
    };
    let mutator_frontend = work_stealing_frontend::FrontendBuilder::new()
        .worker_num(n_mutator.try_into().unwrap())
        .worker_creator(Box::new(closure))
        .add_input_type(FuzzerIOType::TestCase)
        .add_input_type(FuzzerIOType::MutatorScoreChange)
        .add_output_type(FuzzerIOType::TestCase)
        .name("mutator")
        .build()
        .unwrap();

    let closure = move || {
        work_stealing_frontend::Worker::new(
            Box::new(crate::mutator::frontend::work_stealing_mutator::MutatorSyncSource::new()),
            0,
        )
    };

    let mutator_syncsource = work_stealing_frontend::FrontendBuilder::new()
        .worker_num(1)
        .worker_creator(Box::new(closure))
        .add_input_type(FuzzerIOType::MutationFeedback)
        .add_output_type(FuzzerIOType::MutatorScoreChange)
        .name("mutator_syncsource")
        .build()
        .unwrap();

    let init_seeds = util::read_corpus(fuzzer.input).unwrap();

    let copy_args = args.clone();
    let closure = move || {
        let test_case_minimizer =
            TestCaseMinimizer::new(copy_args.clone(), map_size, fuzzer.timeout);
        work_stealing_frontend::Worker::new(
            Box::new(
                crate::queue::frontend::work_stealing_queue_manager::QueueManagerWorker::new(
                    Some(init_seeds.clone()),
                    Some(test_case_minimizer),
                ),
            ),
            0,
        )
    };

    let queue_manager_frontend = work_stealing_frontend::FrontendBuilder::new()
        .worker_num(n_queue_manager.try_into().unwrap())
        .worker_creator(Box::new(closure))
        .add_input_type(FuzzerIOType::TestCase)
        .add_input_type(FuzzerIOType::TestCaseScoreChange)
        .add_non_blocking_output_type(FuzzerIOType::TestCase)
        .add_output_type(FuzzerIOType::MonitorData)
        .name("queue manager")
        .build()
        .unwrap();
    let closure = move || {
        work_stealing_frontend::Worker::new(
            Box::new(
                crate::queue::frontend::work_stealing_queue_manager::QueueManagerSyncSource::new(),
            ),
            0,
        )
    };

    let queue_manager_syncsource = work_stealing_frontend::FrontendBuilder::new()
        .worker_num(1)
        .worker_creator(Box::new(closure))
        .add_input_type(FuzzerIOType::TestCaseFeedback)
        .add_output_type(FuzzerIOType::TestCaseScoreChange)
        .name("queue_manager_syncsource")
        .build()
        .unwrap();

    let feedback_transformer =
        crate::feedback::frontend::work_stealing_feedback::FeedbackCollectorTransformer::new(
            map_size,
        );
    let closure = move || {
        work_stealing_frontend::Worker::new(
            Box::new(
                crate::feedback::frontend::work_stealing_feedback::FeedbackCollectorWorker::new(
                    map_size,
                    feedback_transformer.clone(),
                ),
            ),
            0,
        )
    };

    let feedback_collector_frontend = work_stealing_frontend::FrontendBuilder::new()
        .worker_num(n_feedback_collector.try_into().unwrap())
        .worker_creator(Box::new(closure))
        .add_input_type(FuzzerIOType::Feedback)
        .add_output_type(FuzzerIOType::MonitorData)
        .add_output_type(FuzzerIOType::TestCase)
        .add_output_type(FuzzerIOType::MutationFeedback)
        .add_output_type(FuzzerIOType::TestCaseFeedback)
        .name("feedback_collector")
        .build()
        .unwrap();

    let monitor_creator = move || {
        work_stealing_frontend::Worker::new(
            Box::<work_stealing_frontend::MonitorWorker>::default(),
            0,
        )
    };
    let monitor_frontend = work_stealing_frontend::FrontendBuilder::new()
        .worker_num(1)
        .worker_creator(Box::new(monitor_creator))
        .add_input_type(FuzzerIOType::MonitorData)
        .name("monitor")
        .build()
        .unwrap();

    let mut fuzzer = work_stealing_frontend::FuzzerBuilder::new()
        .add_frontend("queue_manager", queue_manager_frontend)
        .add_frontend("mutator", mutator_frontend)
        .add_frontend("mutator_syncsource", mutator_syncsource)
        .add_frontend("queue_manager_syncsource", queue_manager_syncsource)
        .add_frontend("executor", executor_frontend)
        .add_frontend("feedback_collector", feedback_collector_frontend)
        .add_frontend("monitor", monitor_frontend)
        .add_connection(
            "queue_manager",
            "mutator",
            vec![ChannelIOType::NonBlocking(FuzzerIOType::TestCase)],
        )
        .add_connection(
            "mutator",
            "executor",
            vec![ChannelIOType::Normal(FuzzerIOType::TestCase)],
        )
        .add_connection(
            "executor",
            "feedback_collector",
            vec![ChannelIOType::Normal(FuzzerIOType::Feedback)],
        )
        .add_connection(
            "executor",
            "monitor",
            vec![ChannelIOType::Normal(FuzzerIOType::MonitorData)],
        )
        .add_connection(
            "queue_manager",
            "monitor",
            vec![ChannelIOType::Normal(FuzzerIOType::MonitorData)],
        )
        .add_connection(
            "feedback_collector",
            "monitor",
            vec![ChannelIOType::Normal(FuzzerIOType::MonitorData)],
        )
        .add_connection(
            "feedback_collector",
            "queue_manager",
            vec![ChannelIOType::Normal(FuzzerIOType::TestCase)],
        )
        .add_connection(
            "feedback_collector",
            "mutator_syncsource",
            vec![ChannelIOType::Normal(FuzzerIOType::MutationFeedback)],
        )
        .add_connection(
            "mutator_syncsource",
            "mutator",
            vec![ChannelIOType::Watcher(FuzzerIOType::MutatorScoreChange)],
        )
        .add_connection(
            "feedback_collector",
            "queue_manager_syncsource",
            vec![ChannelIOType::Normal(FuzzerIOType::TestCaseFeedback)],
        )
        .add_connection(
            "queue_manager_syncsource",
            "queue_manager",
            vec![ChannelIOType::Watcher(FuzzerIOType::TestCaseScoreChange)],
        )
        .build()
        .unwrap();

    fuzzer.run();

    tokio::spawn(async move {
        let monitor = monitor::get_monitor();
        monitor.write().unwrap().reset_start_time();
        loop {
            sleep(Duration::from_millis(2000)).await;
            monitor.read().unwrap().show_statistics();
        }
    })
    .await
    .unwrap();
}
