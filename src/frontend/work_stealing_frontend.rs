use std::{
    collections::HashMap,
    hash::Hash,
    sync::{Arc, Mutex},
    time::Duration,
};

use crate::frontend::{FuzzerIO, FuzzerIOType};
use crate::monitor::Monitor;
use async_channel;
use itertools::Itertools;
use rand::prelude::StdRng;
use rand::{
    distributions::{Bernoulli, Distribution},
    SeedableRng,
};

pub trait WorkerImpl {
    fn handle_inputs(&mut self, input: FuzzerIO) -> Option<Vec<FuzzerIO>>;
    // When defined, the worker can generate outputs without processing any inputs.
    // One example is the queue manager. As long as the queue is not empty, the worker can return
    // some random test cases from the queue.
    fn generate_outputs(&mut self, _io_type: FuzzerIOType) -> Option<FuzzerIO> {
        unimplemented!()
    }
}

#[derive(Clone, Debug, PartialEq, Eq, Hash)]
pub enum IOResult {
    OK,
    SendError(String),
    RecvError(String),
}

#[derive(Debug)]
enum InnerReceiver {
    NormalReceiver(async_channel::Receiver<FuzzerIO>),
    WatchReceiver(Arc<Mutex<tokio::sync::watch::Receiver<FuzzerIO>>>),
}

impl Clone for InnerReceiver {
    fn clone(&self) -> Self {
        match self {
            Self::NormalReceiver(receiver) => Self::NormalReceiver(receiver.clone()),
            Self::WatchReceiver(receiver) => Self::new_watcher(receiver.lock().unwrap().clone()),
        }
    }
}

impl InnerReceiver {
    pub fn new_watcher(receiver: tokio::sync::watch::Receiver<FuzzerIO>) -> Self {
        Self::WatchReceiver(Arc::new(Mutex::new(receiver)))
    }

    pub async fn recv(&self) -> Result<FuzzerIO, IOResult> {
        match self {
            Self::NormalReceiver(receiver) => receiver
                .recv()
                .await
                .map_err(|s| IOResult::RecvError(s.to_string())),
            Self::WatchReceiver(receiver) => {
                let mut counter = 1;
                loop {
                    let result = receiver.lock().unwrap().has_changed();
                    match result {
                        Ok(result) => {
                            if !result {
                                tokio::time::sleep(Duration::from_millis(counter * 10)).await;
                                counter += 1;
                            } else {
                                return Ok(receiver.lock().unwrap().borrow_and_update().clone());
                            }
                        }
                        Err(err) => {
                            return Err(IOResult::RecvError(err.to_string()));
                        }
                    }
                }
            }
        }
    }

    pub fn is_closed(&self) -> bool {
        match self {
            Self::NormalReceiver(receiver) => receiver.is_closed(),
            Self::WatchReceiver(_) => false,
        }
    }

    pub fn try_recv(&self) -> Result<FuzzerIO, IOResult> {
        match self {
            Self::NormalReceiver(receiver) => receiver
                .try_recv()
                .map_err(|x| IOResult::RecvError(x.to_string())),
            Self::WatchReceiver(_) => unimplemented!(),
        }
    }
}

#[derive(Clone, Debug)]
enum InnerSender {
    NormalSender(async_channel::Sender<FuzzerIO>),
    WatchSender(Arc<Mutex<tokio::sync::watch::Sender<FuzzerIO>>>),
}

impl InnerSender {
    pub async fn send(&self, content: FuzzerIO) -> Result<IOResult, IOResult> {
        match self {
            Self::NormalSender(sender) => sender
                .send(content)
                .await
                .map(|_| IOResult::OK)
                .map_err(|s| IOResult::SendError(s.to_string())),
            Self::WatchSender(sender) => sender
                .lock()
                .unwrap()
                .send(content)
                .map(|_| IOResult::OK)
                .map_err(|err| IOResult::SendError(err.to_string())),
        }
    }

    pub fn is_closed(&self) -> bool {
        match self {
            Self::NormalSender(sender) => sender.is_closed(),
            Self::WatchSender(sender) => sender.lock().unwrap().is_closed(),
        }
    }

    pub fn is_full(&self) -> bool {
        match self {
            Self::NormalSender(sender) => sender.is_full(),
            Self::WatchSender(_) => false,
        }
    }
}

#[derive(Clone, Debug)]
pub struct Receiver {
    io_type: FuzzerIOType,
    global_receiver: InnerReceiver,
    local_receiver: Option<InnerReceiver>,
}

impl Receiver {
    fn new(
        io_type: FuzzerIOType,
        global_receiver: InnerReceiver,
        local_receiver: Option<InnerReceiver>,
    ) -> Self {
        Self {
            io_type,
            global_receiver,
            local_receiver,
        }
    }

    pub fn is_closed(&self) -> bool {
        self.global_receiver.is_closed()
            && self
                .local_receiver
                .as_ref()
                .map(|r| r.is_closed())
                .unwrap_or(true)
    }

    pub fn get_type(&self) -> FuzzerIOType {
        self.io_type
    }

    pub async fn recv(&self) -> Result<FuzzerIO, IOResult> {
        if let Some(local_receiver) = self.local_receiver.as_ref() {
            if let Ok(input) = local_receiver.try_recv() {
                return Ok(input);
            }
        } else {
            return self.global_receiver.recv().await;
        }

        if let Ok(input) = self.global_receiver.try_recv() {
            return Ok(input);
        }

        if let Some(local_receiver) = self.local_receiver.as_ref() {
            local_receiver.recv().await
        } else {
            Err(IOResult::RecvError("Recv error".to_string()))
        }
    }
}

#[derive(Clone, Debug)]
pub struct Sender {
    io_type: FuzzerIOType,
    global_sender: InnerSender,
    local_sender: Option<InnerSender>,
}

impl Sender {
    fn new(
        io_type: FuzzerIOType,
        global_sender: InnerSender,
        local_sender: Option<InnerSender>,
    ) -> Self {
        Self {
            io_type,
            global_sender,
            local_sender,
        }
    }

    pub fn is_closed(&self) -> bool {
        self.global_sender.is_closed()
            && self
                .local_sender
                .as_ref()
                .map(|s| s.is_closed())
                .unwrap_or(true)
    }

    pub fn is_full(&self) -> bool {
        self.global_sender.is_full()
            || self
                .local_sender
                .as_ref()
                .map(|s| s.is_full())
                .unwrap_or(false)
    }

    pub fn get_type(&self) -> FuzzerIOType {
        self.io_type
    }

    pub async fn send(&self, input: FuzzerIO) -> Result<IOResult, IOResult> {
        let has_local = self.local_sender.is_some();
        if let Some(local_sender) = self.local_sender.as_ref() {
            if !local_sender.is_full() {
                return local_sender.send(input).await;
            }
        }
        if !self.global_sender.is_full() || !has_local {
            self.global_sender.send(input).await
        } else {
            self.local_sender.as_ref().unwrap().send(input).await
        }
    }
}

#[derive(Eq, Debug, Clone, Copy, PartialOrd, Ord)]
pub enum ChannelIOType {
    Normal(FuzzerIOType),
    NonBlocking(FuzzerIOType),
    Watcher(FuzzerIOType),
}

impl PartialEq for ChannelIOType {
    fn eq(&self, other: &Self) -> bool {
        self.get_io_type() == other.get_io_type()
    }
}

impl ChannelIOType {
    pub fn is_normal(&self) -> bool {
        matches!(self, Self::Normal(_))
    }

    pub fn is_non_blocking(&self) -> bool {
        matches!(self, Self::NonBlocking(_))
    }

    pub fn is_watcher(&self) -> bool {
        matches!(self, Self::Watcher(_))
    }

    pub fn get_io_type(&self) -> FuzzerIOType {
        match self {
            Self::Normal(io_type) | Self::NonBlocking(io_type) | Self::Watcher(io_type) => *io_type,
        }
    }
}

pub(crate) struct Channel {}

impl Channel {
    pub fn watcher(io_type: FuzzerIOType, num: usize) -> (Vec<Sender>, Vec<Receiver>) {
        let (sender, receiver) = tokio::sync::watch::channel(FuzzerIO::get_default(io_type));
        let sender = Sender::new(
            io_type,
            InnerSender::WatchSender(Arc::new(Mutex::new(sender))),
            None,
        );

        (
            vec![sender],
            vec![Receiver::new(io_type, InnerReceiver::new_watcher(receiver), None,); num],
        )
    }

    pub fn bounded(
        io_type: FuzzerIOType,
        num: usize,
        channle_size: usize,
        work_stealing: bool,
    ) -> (Vec<Sender>, Vec<Receiver>) {
        let (global_sender, global_receiver) = async_channel::bounded(channle_size);
        let global_sender = InnerSender::NormalSender(global_sender);
        if !work_stealing {
            return (
                vec![Sender::new(io_type, global_sender, None); num],
                vec![
                    Receiver::new(
                        io_type,
                        InnerReceiver::NormalReceiver(global_receiver),
                        None
                    );
                    num
                ],
            );
        }

        let mut senders = Vec::new();
        let mut receivers = Vec::new();

        for _ in 0..num {
            let (local_sender, local_receiver) = async_channel::bounded(channle_size);
            senders.push(Sender::new(
                io_type,
                global_sender.clone(),
                Some(InnerSender::NormalSender(local_sender)),
            ));
            receivers.push(Receiver::new(
                io_type,
                InnerReceiver::NormalReceiver(global_receiver.clone()),
                Some(InnerReceiver::NormalReceiver(local_receiver)),
            ));
        }
        (senders, receivers)
    }
}
pub struct Worker {
    inner: Box<dyn WorkerImpl + Send + 'static>,
    #[allow(dead_code)]
    id: usize,
    receivers: Vec<Receiver>,
    senders: HashMap<FuzzerIOType, Vec<Sender>>,
    non_blocking: Vec<FuzzerIOType>,
    frontend_name: String,
}

impl Worker {
    pub fn new(inner: Box<dyn WorkerImpl + Send + 'static>, id: usize) -> Self {
        Self {
            inner,
            id,
            senders: HashMap::new(),
            receivers: Vec::new(),
            non_blocking: Vec::new(),
            frontend_name: "".to_string(),
        }
    }

    pub fn get_senders(&self, io_type: FuzzerIOType) -> Vec<Sender> {
        self.senders.get(&io_type).unwrap().clone()
    }

    pub fn get_receivers(&self, io_type: FuzzerIOType) -> Vec<Receiver> {
        self.receivers
            .iter()
            .filter_map(|receiver| {
                if receiver.io_type == io_type {
                    Some(receiver.clone())
                } else {
                    None
                }
            })
            .collect()
    }

    pub fn set_frontend_name(&mut self, name: String) {
        self.frontend_name = name;
    }

    pub fn add_input_receiver(&mut self, receiver: Receiver) {
        assert!(!receiver.is_closed());
        self.receivers.push(receiver);
    }

    pub fn add_output_sender(&mut self, sender: Sender, non_blocking: bool) {
        assert!(!sender.is_closed());
        if non_blocking {
            self.non_blocking.push(sender.get_type());
        }
        self.senders
            .entry(sender.get_type())
            .or_insert(Vec::new())
            .push(sender);
    }

    pub fn handle_inputs(&mut self, input: FuzzerIO) -> Option<Vec<FuzzerIO>> {
        self.inner.handle_inputs(input)
    }

    async fn select_all(
        receivers: &[Receiver],
        start_idx: usize,
    ) -> (Result<FuzzerIO, IOResult>, usize) {
        let (inputs, idx, _) = futures::future::select_all(
            receivers
                //let all_futures = receivers
                .iter()
                .cycle()
                .skip(start_idx)
                .take(receivers.len())
                .map(|receiver| Box::pin(receiver.recv())),
        )
        .await;
        (inputs, (idx + 1) % receivers.len())
    }

    fn can_send(senders: &HashMap<FuzzerIOType, Vec<Sender>>, outputs: &[FuzzerIO]) -> bool {
        !outputs
            .iter()
            .filter(|output| !output.is_empty())
            .any(|output| {
                let output_type = output.get_type();
                senders
                    .get(&output_type)
                    .unwrap()
                    .iter()
                    .any(|sender| sender.is_full())
            })
    }

    async fn send_outputs(
        senders: &HashMap<FuzzerIOType, Vec<Sender>>,
        outputs: Vec<FuzzerIO>,
        frontend_name: String,
    ) -> Result<IOResult, IOResult> {
        let mut counter = 1;
        while !Self::can_send(senders, &outputs) {
            //println!("Cannot send yet");
            tokio::time::sleep(Duration::from_millis(counter * 10)).await;
            counter += 1;
        }
        //println!("Can send");
        let mut result = Ok(IOResult::OK);
        for output in outputs.into_iter().filter(|output| !output.is_empty()) {
            let outputs_type = output.get_type();
            assert!(
                senders.contains_key(&outputs_type),
                "Every type of output ({:?}) in {} should be handled!",
                outputs_type,
                frontend_name,
            );
            let senders_wanting_same_inputs = senders.get(&outputs_type).unwrap();
            let output_copies = vec![output; senders_wanting_same_inputs.len()];
            for (sender, output) in senders_wanting_same_inputs
                .iter()
                .zip(output_copies.into_iter())
            {
                if let Err(s) = sender.send(output).await {
                    result = Err(s);
                    break;
                }
            }
        }
        result
    }

    pub fn generate_outputs(&mut self, io_type: FuzzerIOType) -> FuzzerIO {
        self.inner.generate_outputs(io_type).unwrap()
    }

    fn run_non_blocking_thread(
        worker: Arc<Mutex<Self>>,
        sender_map: HashMap<FuzzerIOType, Vec<Sender>>,
    ) {
        let non_blocking_types = {
            let worker = worker.lock().unwrap();
            worker.non_blocking.clone()
        };
        let sender_map = sender_map
            .into_iter()
            .filter(|(io_type, _)| non_blocking_types.contains(io_type))
            .collect::<HashMap<_, _>>();

        tokio::spawn(async move {
            loop {
                let mut should_wait = true;
                for (non_blocking_type, senders) in sender_map.iter() {
                    for sender in senders.iter().filter(|sender| !sender.is_full()) {
                        let input = {
                            let mut worker = worker.lock().unwrap();
                            worker.generate_outputs(*non_blocking_type)
                        };
                        sender.send(input).await.unwrap();
                        should_wait = false;
                    }
                }
                // All the queues are full, wait for a bit.
                if should_wait {
                    tokio::time::sleep(Duration::from_millis(5)).await;
                }
            }
        });
    }

    pub fn run_in_background(self) -> Arc<Mutex<Self>> {
        let worker = Arc::new(Mutex::new(self));
        Self::run(worker.clone());
        worker
    }

    // Run the worker asyncronously.
    pub fn run(worker: Arc<Mutex<Self>>) {
        let mut rng = StdRng::from_entropy();
        let d = Bernoulli::new(0.6).unwrap();
        tokio::spawn(async move {
            let receivers = worker.lock().unwrap().receivers.clone();
            let non_blocking = worker.lock().unwrap().non_blocking.clone();
            for receiver in receivers.iter() {
                assert!(!receiver.is_closed());
            }
            let senders = worker.lock().unwrap().senders.clone();
            if !non_blocking.is_empty() {
                Self::run_non_blocking_thread(worker.clone(), senders.clone());
            }
            let frontend_name = worker.lock().unwrap().frontend_name.clone();
            assert!(
                !receivers.is_empty(),
                "{} has no input receivers!",
                &frontend_name
            );
            let mut start_idx = 0;
            loop {
                let (inputs, next_index) = Self::select_all(&receivers, start_idx).await;
                start_idx = next_index;
                if inputs.is_err() {
                    panic!("{} input error: {:?}", frontend_name, inputs.err().unwrap());
                }

                let inputs = inputs.unwrap();
                if inputs.is_empty() {
                    continue;
                }

                let outputs = worker.lock().unwrap().handle_inputs(inputs);

                if let Some(outputs) = outputs {
                    Self::send_outputs(&senders, outputs, frontend_name.clone())
                        .await
                        .unwrap();
                }

                if d.sample(&mut rng) {
                    tokio::task::yield_now().await;
                }
            }
        });
    }
}

pub struct Frontend {
    workers: Vec<Arc<Mutex<Worker>>>,
    input_types: Vec<ChannelIOType>,
    output_types: Vec<ChannelIOType>,
    name: String,
}

pub struct FrontendBuilder {
    worker_num: usize,
    input_types: Vec<ChannelIOType>,
    output_types: Vec<ChannelIOType>,
    worker_create_fn: Option<Box<dyn Fn() -> Worker>>,
    name: String,
}

impl Default for FrontendBuilder {
    fn default() -> Self {
        Self::new()
    }
}

impl FrontendBuilder {
    pub fn new() -> Self {
        Self {
            worker_num: 1,
            input_types: Vec::new(),
            output_types: Vec::new(),
            worker_create_fn: None,
            name: "".to_string(),
        }
    }

    pub fn worker_num(mut self, worker_num: usize) -> Self {
        self.worker_num = worker_num;
        self
    }

    pub fn name(mut self, name: &str) -> Self {
        self.name = name.to_string();
        self
    }

    pub fn worker_creator(mut self, worker_create_fn: Box<dyn Fn() -> Worker>) -> Self {
        self.worker_create_fn = Some(worker_create_fn);
        self
    }

    pub fn add_input_type(mut self, input_type: FuzzerIOType) -> Self {
        self.input_types.push(ChannelIOType::Normal(input_type));
        self
    }

    pub fn add_output_type(mut self, output_type: FuzzerIOType) -> Self {
        self.output_types.push(ChannelIOType::Normal(output_type));
        self
    }

    pub fn add_non_blocking_output_type(mut self, output_type: FuzzerIOType) -> Self {
        self.output_types
            .push(ChannelIOType::NonBlocking(output_type));
        self
    }

    pub fn build(self) -> Option<Frontend> {
        if self.worker_num == 0 || self.input_types.is_empty() || self.worker_create_fn.is_none() {
            return None;
        }
        let worker_create_fn = self.worker_create_fn.unwrap();
        let workers = (0..self.worker_num)
            .map(|_| {
                let mut worker = worker_create_fn();
                worker.set_frontend_name(self.name.clone());
                Arc::new(Mutex::new(worker))
            })
            .collect();

        Some(Frontend {
            workers,
            input_types: self.input_types,
            output_types: self.output_types,
            name: self.name.clone(),
        })
    }
}

impl Frontend {
    pub fn builder() -> FrontendBuilder {
        FrontendBuilder::new()
    }

    pub fn get_input_types(&self) -> Vec<ChannelIOType> {
        self.input_types.clone()
    }

    pub fn get_name(&self) -> String {
        self.name.clone()
    }

    pub fn get_output_types(&self) -> Vec<ChannelIOType> {
        self.output_types.clone()
    }

    pub fn is_output_type_blocking(&self, output_type: FuzzerIOType) -> bool {
        self.output_types
            .iter()
            .find(|channel_io_type| channel_io_type.get_io_type() == output_type)
            .unwrap()
            .is_non_blocking()
    }

    pub fn setup_inputs(&self, input_io_types: Vec<ChannelIOType>) -> Vec<Vec<Sender>> {
        for input_io_type in input_io_types.iter() {
            if !self.input_types.contains(input_io_type) {
                panic!("Input type not found!");
            }
        }

        let mut all_senders = Vec::new();
        for input_io_type in input_io_types.into_iter() {
            let (senders, receivers) =
                Channel::bounded(input_io_type.get_io_type(), self.worker_num(), 50, true);

            for (worker, receiver) in self.workers.iter().zip(receivers.into_iter()) {
                let mut worker_guard = worker.lock().unwrap();
                worker_guard.add_input_receiver(receiver);
            }
            all_senders.push(senders);
        }
        all_senders
    }

    pub fn setup_outputs(&self, output_io_types: Vec<ChannelIOType>) -> Vec<Vec<Receiver>> {
        for output_io_type in output_io_types.iter() {
            if !self.output_types.contains(output_io_type) {
                panic!("Output type not found!");
            }
        }

        let mut all_receivers = Vec::new();
        for output_io_type in output_io_types.into_iter() {
            let (senders, receivers) =
                Channel::bounded(output_io_type.get_io_type(), self.worker_num(), 50, true);

            for (worker, sender) in self.workers.iter().zip(senders.into_iter()) {
                let mut worker_guard = worker.lock().unwrap();
                worker_guard.add_output_sender(sender, output_io_type.is_non_blocking());
            }
            all_receivers.push(receivers);
        }
        all_receivers
    }

    pub fn self_setup(&self) -> (Vec<Vec<Sender>>, Vec<Vec<Receiver>>) {
        (
            self.setup_inputs(self.input_types.clone()),
            self.setup_outputs(self.output_types.clone()),
        )
    }

    pub fn connect(&self, to_frontend: &Frontend, io_types: &Vec<ChannelIOType>) -> bool {
        let output_types = self.get_output_types();
        let input_types = to_frontend.get_input_types();
        for io_type in io_types.iter() {
            if !input_types.contains(io_type) || !output_types.contains(io_type) {
                return false;
            }
        }

        let work_stealing = self.worker_num() == to_frontend.worker_num();
        if !work_stealing {
            assert!(
                to_frontend.worker_num() == 1 || self.worker_num() == 1,
                "Not supported yet."
            );
        }
        let channel_num = self.worker_num().max(to_frontend.worker_num());
        let channel_size = if work_stealing {
            10
        } else {
            10 * self.worker_num()
        };

        for io_type in io_types {
            let (senders, receivers) = match *io_type {
                ChannelIOType::NonBlocking(inner_io) | ChannelIOType::Normal(inner_io) => {
                    Channel::bounded(inner_io, channel_num, channel_size, work_stealing)
                }
                ChannelIOType::Watcher(inner_io) => Channel::watcher(inner_io, channel_num),
            };
            for (worker, sender) in self.workers.iter().zip(senders.into_iter()) {
                println!("Adding outputs from {}'s workers", self.get_name());
                let mut worker_guard = worker.lock().unwrap();
                assert!(!sender.is_closed());
                worker_guard.add_output_sender(sender, io_type.is_non_blocking());
            }

            for (worker, receiver) in to_frontend.workers.iter().zip(receivers.into_iter()) {
                println!("Adding inputs to {}'s workers", to_frontend.get_name());
                let mut worker_guard = worker.lock().unwrap();
                assert!(!receiver.is_closed());
                worker_guard.add_input_receiver(receiver);
            }
        }
        true
    }

    pub fn worker_num(&self) -> usize {
        self.workers.len()
    }

    pub fn run(&self) {
        for worker in &self.workers {
            Worker::run(worker.clone());
        }
    }

    pub fn workers(&self) -> &Vec<Arc<Mutex<Worker>>> {
        &self.workers
    }

    pub fn get_senders(&self, io_type: FuzzerIOType) -> Vec<Sender> {
        self.workers
            .iter()
            .flat_map(|worker| {
                let worker_guard = worker.lock().unwrap();
                worker_guard.get_senders(io_type)
            })
            .collect()
    }
}

pub struct Fuzzer {
    frontend: HashMap<String, Frontend>,
}

impl Fuzzer {
    pub fn run(&mut self) {
        for (name, frontend) in self.frontend.iter() {
            println!("Running frontend {}", name);
            frontend.run();
        }
    }
}

pub struct FuzzerBuilder {
    frontend: HashMap<String, Frontend>,
    connections: HashMap<String, Vec<(String, Vec<ChannelIOType>)>>,
}

#[derive(Debug)]
pub enum FuzzerBuildError {
    FrontendIsEmpty,
    MonitorNotSet,
    ConnectionsInComplete(String),
}

impl Default for FuzzerBuilder {
    fn default() -> Self {
        Self::new()
    }
}

impl FuzzerBuilder {
    pub fn new() -> Self {
        Self {
            frontend: HashMap::new(),
            connections: HashMap::new(),
        }
    }

    pub fn add_frontend(mut self, name: &str, frontend: Frontend) -> Self {
        assert!(!self.frontend.contains_key(name));
        self.frontend.insert(name.to_string(), frontend);
        self
    }

    pub fn add_connection(
        mut self,
        from_frontend: &str,
        to_frontend: &str,
        io_types: Vec<ChannelIOType>,
    ) -> Self {
        self.connections
            .entry(from_frontend.to_string())
            .or_insert(Vec::new())
            .push((to_frontend.to_string(), io_types));
        self
    }

    fn find_all_input_types(&self, frontend: &String) -> Vec<ChannelIOType> {
        let mut input_types = Vec::new();
        for (_, connections) in self.connections.iter() {
            for (to_frontend, io_types) in connections {
                if *to_frontend == *frontend {
                    input_types.extend(io_types.iter().cloned());
                }
            }
        }
        input_types
    }

    fn find_all_output_types(&self, frontend: &String) -> Vec<ChannelIOType> {
        let mut output_types = Vec::new();
        if !self.connections.contains_key(frontend) {
            println!("{} has no output!", frontend);
            return output_types;
        }
        for (_, connections) in self.connections[frontend].iter() {
            output_types.extend(connections.iter().cloned());
        }
        output_types
    }

    pub fn verify(&self) -> Result<(), FuzzerBuildError> {
        for (frontend_name, frontend) in self.frontend.iter() {
            // Check whether all input types are connected.
            let expected_input_types = frontend.input_types.clone();
            let actual_input_types = self.find_all_input_types(frontend_name);
            if !expected_input_types
                .iter()
                .map(|x| x.get_io_type())
                .unique()
                .sorted()
                .eq(actual_input_types
                    .iter()
                    .map(|x| x.get_io_type())
                    .unique()
                    .sorted())
            {
                println!(
                    "expected input types {:?}, actual {:?}",
                    expected_input_types, actual_input_types
                );
                return Err(FuzzerBuildError::ConnectionsInComplete(format!(
                    "Input types for {} mismatch",
                    frontend_name
                )));
            }

            // Check whether all output types are connected.

            let actual_output_types = self.find_all_output_types(frontend_name);
            if !frontend
                .output_types
                .iter()
                .cloned()
                .map(|x| x.get_io_type())
                .sorted()
                .eq(actual_output_types
                    .into_iter()
                    .map(|x| x.get_io_type())
                    .sorted())
            {
                return Err(FuzzerBuildError::ConnectionsInComplete(format!(
                    "Output types for {} mismatch",
                    frontend_name
                )));
            }
        }
        Ok(())
    }

    pub fn build(self) -> Result<Fuzzer, FuzzerBuildError> {
        if self.frontend.is_empty() {
            return Err(FuzzerBuildError::FrontendIsEmpty);
        }
        if let Err(e) = self.verify() {
            println!("Verification error: {:?}", e);
            return Err(e);
        }

        // Do the actual connenction.
        for (frontend_name, connections) in self.connections.iter() {
            let frontend = self.frontend.get(frontend_name).unwrap();
            for (to_frontend, io_types) in connections {
                println!("Connecting {} to {}", frontend_name, to_frontend);
                frontend.connect(&self.frontend[to_frontend], io_types);
            }
        }

        // Return a fuzzer.
        Ok(Fuzzer {
            frontend: self.frontend,
        })
    }
}

#[derive(Debug, Default)]
pub struct MonitorWorker {}

impl WorkerImpl for MonitorWorker {
    fn handle_inputs(&mut self, input: FuzzerIO) -> Option<Vec<FuzzerIO>> {
        match input {
            FuzzerIO::MonitorData(data) => {
                let monitor = crate::monitor::get_monitor();
                for val in data {
                    monitor.read().unwrap().receive_statistics(val);
                }
                None
            }
            _ => {
                unreachable!()
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::Channel;
    use super::*;
    use super::{FuzzerIO, FuzzerIOType};
    use crate::datatype::{ExecutionStatus, Feedback, TestCase};

    #[derive(Default)]
    struct ToyWorker {}

    impl WorkerImpl for ToyWorker {
        fn handle_inputs(&mut self, input: FuzzerIO) -> Option<Vec<FuzzerIO>> {
            // Make it a loop so that it can connectted to itself. Only for testing.
            // Given test cases, it generates feedbacks.
            // Given feedbacks, it generates monitor data.
            // Given monitor data, it generates test cases.
            let test_case: TestCase = TestCase::default();
            let feedback: Feedback = Feedback::new(ExecutionStatus::Ok);
            let monitor_data: serde_json::Value = serde_json::Value::Null;
            if input.is_empty() {
                return None;
            }
            match input {
                FuzzerIO::TestCase(_) => Some(vec![FuzzerIO::Feedback(vec![feedback])]),
                FuzzerIO::Feedback(_) => Some(vec![FuzzerIO::MonitorData(vec![monitor_data])]),
                FuzzerIO::MutationFeedback(_) => todo!(),
                FuzzerIO::TestCaseFeedback(_) => todo!(),
                FuzzerIO::MonitorData(_) => Some(vec![FuzzerIO::TestCase(vec![test_case])]),
                FuzzerIO::TestCaseScoreChange(_) => todo!(),
                FuzzerIO::MutatorScoreChange(_) => todo!(),
            }
        }

        fn generate_outputs(&mut self, io_type: FuzzerIOType) -> Option<FuzzerIO> {
            match io_type {
                FuzzerIOType::TestCase => todo!(),
                FuzzerIOType::Feedback => todo!(),
                FuzzerIOType::MutationFeedback => todo!(),
                FuzzerIOType::TestCaseFeedback => todo!(),
                FuzzerIOType::MonitorData => Some(FuzzerIO::MonitorData(vec![serde_json::json!({
                    "exec": 10_u64
                })])),
                FuzzerIOType::TestCaseScoreChange => todo!(),
                FuzzerIOType::MutatorScoreChange => todo!(),
            }
        }
    }

    fn toy_worker_creator() -> Box<dyn Fn() -> Worker> {
        Box::new(move || Worker::new(Box::<ToyWorker>::default(), 0))
    }

    fn create_toy_worker(
        input_types: Vec<FuzzerIOType>,
        output_types: Vec<(FuzzerIOType, bool)>,
    ) -> (Worker, Vec<Sender>, Vec<Receiver>) {
        let mut worker = (toy_worker_creator())();
        let mut senders = Vec::new();
        let mut receivers = Vec::new();
        for io_type in input_types {
            let (mut input_senders, mut input_receivers) = Channel::bounded(io_type, 1, 50, true);
            senders.push(input_senders.pop().unwrap());
            worker.add_input_receiver(input_receivers.pop().unwrap());
        }
        for (io_type, non_blocking) in output_types {
            let (mut output_senders, mut output_receivers) = Channel::bounded(io_type, 1, 50, true);
            worker.add_output_sender(output_senders.pop().unwrap(), non_blocking);
            receivers.push(output_receivers.pop().unwrap());
        }
        (worker, senders, receivers)
    }

    fn create_toy_frontend(
        worker_num: usize,
        input_types: Vec<FuzzerIOType>,
        output_types: Vec<FuzzerIOType>,
    ) -> Frontend {
        let builder = FrontendBuilder::new();

        let mut builder = builder
            .worker_num(worker_num)
            .worker_creator(toy_worker_creator());

        for io_type in input_types {
            builder = builder.add_input_type(io_type);
        }

        for io_type in output_types {
            builder = builder.add_output_type(io_type);
        }
        builder.build().unwrap()
    }

    #[tokio::test(flavor = "multi_thread", worker_threads = 2)]
    async fn worker_setup_input_and_output_correctly() {
        let (worker, mut senders, mut receivers) = create_toy_worker(
            vec![FuzzerIOType::TestCase],
            vec![(FuzzerIOType::Feedback, false)],
        );

        let test_case = FuzzerIO::TestCase(vec![TestCase::default()]);
        let sender = senders.pop().unwrap();
        let receiver = receivers.pop().unwrap();
        worker.run_in_background();

        for _ in 0..100 {
            println!("Sending test case");
            assert!(sender.send(test_case.clone()).await.is_ok());
            println!("Sent test case");
            println!("Receiving feedback");
            let output = receiver.recv().await;
            assert!(output.is_ok());
            println!("Received feedback");
            let output = output.unwrap();
            assert_eq!(output.get_type(), FuzzerIOType::Feedback);
        }
    }

    #[tokio::test(flavor = "multi_thread", worker_threads = 2)]
    async fn frontend_bridge_workers_correctly() {
        let frontend = create_toy_frontend(
            5,
            vec![FuzzerIOType::TestCase],
            vec![FuzzerIOType::Feedback],
        );
        let (all_senders, all_receivers) = frontend.self_setup();

        frontend.run();
        // Send test cases to every worker.
        for senders in all_senders.iter() {
            for sender in senders.iter() {
                println!("Sending test case");
                sender
                    .send(FuzzerIO::TestCase(vec![TestCase::default()]))
                    .await
                    .unwrap();
            }
        }

        // Receive feedback from every worker.
        for receivers in all_receivers.iter() {
            for receiver in receivers.iter() {
                println!("Receiving feedback");
                let output = receiver.recv().await.unwrap();
                assert_eq!(output.len(), 1);
                assert_eq!(output.get_type(), receiver.get_type());
            }
        }
    }

    #[tokio::test(flavor = "multi_thread", worker_threads = 2)]
    async fn frontend_bridge_workers_correctly_with_non_blocking_output() {
        let (worker, _all_senders, all_receivers) = create_toy_worker(
            vec![FuzzerIOType::TestCase],
            vec![(FuzzerIOType::MonitorData, true)],
        );

        worker.run_in_background();

        // Monitor data is non-blocking, so we can read from it without feeding any inputs.
        for receiver in all_receivers.iter() {
            for _ in 0..100 {
                let output = receiver.recv().await.unwrap();
                assert_eq!(output.len(), 1);
                assert_eq!(output.get_type(), receiver.get_type());
            }
        }
    }

    #[test]
    fn frontend_connect_with_matched_io_types_succeeds() {
        let frontend1 = create_toy_frontend(
            5,
            vec![FuzzerIOType::TestCase],
            vec![FuzzerIOType::Feedback],
        );

        let frontend2 = create_toy_frontend(
            5,
            vec![FuzzerIOType::Feedback],
            vec![FuzzerIOType::MonitorData],
        );

        let io_types = vec![ChannelIOType::Normal(FuzzerIOType::Feedback)];
        assert!(frontend1.connect(&frontend2, &io_types));
    }

    #[test]
    fn frontend_connect_with_mismatched_io_types_fails() {
        let frontend1 = create_toy_frontend(
            5,
            vec![FuzzerIOType::TestCase],
            vec![FuzzerIOType::Feedback],
        );

        let frontend2 = create_toy_frontend(
            5,
            vec![FuzzerIOType::Feedback],
            vec![FuzzerIOType::MonitorData],
        );

        let io_types = vec![ChannelIOType::Normal(FuzzerIOType::TestCase)];
        assert!(!frontend1.connect(&frontend2, &io_types));
    }

    #[tokio::test(flavor = "multi_thread", worker_threads = 5)]
    async fn connected_frontend_bridge_correctly() {
        let frontend1 = create_toy_frontend(
            5,
            vec![FuzzerIOType::TestCase],
            vec![FuzzerIOType::Feedback],
        );

        let frontend2 = create_toy_frontend(
            5,
            vec![FuzzerIOType::Feedback],
            vec![FuzzerIOType::MonitorData],
        );

        let io_types = vec![ChannelIOType::Normal(FuzzerIOType::Feedback)];
        frontend1.connect(&frontend2, &io_types);
        let test_case_senders =
            frontend1.setup_inputs(vec![ChannelIOType::Normal(FuzzerIOType::TestCase)]);
        let monitor_data_receivers =
            frontend2.setup_outputs(vec![ChannelIOType::Normal(FuzzerIOType::MonitorData)]);
        frontend1.run();
        frontend2.run();

        // Send test cases to every worker.
        for senders in test_case_senders.iter() {
            for sender in senders.iter() {
                println!("Sending test case");
                sender
                    .send(FuzzerIO::TestCase(vec![TestCase::default()]))
                    .await
                    .unwrap();
            }
        }

        // Receive feedback from every worker.
        for receivers in monitor_data_receivers.iter() {
            for receiver in receivers.iter() {
                println!("Receiving monitor data");
                let output = receiver.recv().await.unwrap();
                assert_eq!(output.len(), 1);
                assert_eq!(output.get_type(), receiver.get_type());
            }
        }
    }

    // TODO(yongheng): Add a test for nonblocking.

    #[tokio::test(flavor = "multi_thread", worker_threads = 5)]
    async fn watcher_channel_let_all_receivers_get_inputs() {
        let frontend1 = create_toy_frontend(
            1,
            vec![FuzzerIOType::TestCase],
            vec![FuzzerIOType::Feedback],
        );

        let frontend2 = create_toy_frontend(
            5,
            vec![FuzzerIOType::Feedback],
            vec![FuzzerIOType::MonitorData],
        );

        let io_types = vec![ChannelIOType::Watcher(FuzzerIOType::Feedback)];
        assert!(frontend1.connect(&frontend2, &io_types));
        let test_case_senders =
            frontend1.setup_inputs(vec![ChannelIOType::Normal(FuzzerIOType::TestCase)]);
        let monitor_data_receivers =
            frontend2.setup_outputs(vec![ChannelIOType::Normal(FuzzerIOType::MonitorData)]);
        frontend1.run();
        frontend2.run();

        // Send test cases to every worker.
        for senders in test_case_senders.iter() {
            for sender in senders.iter() {
                println!("Sending test case");
                sender
                    .send(FuzzerIO::TestCase(vec![TestCase::default()]))
                    .await
                    .unwrap();
            }
        }

        // Receive feedback from every worker.
        for receivers in monitor_data_receivers.iter() {
            for receiver in receivers.iter() {
                println!("Receiving monitor data");
                let output = receiver.recv().await.unwrap();
                assert_eq!(output.len(), 1);
                assert_eq!(output.get_type(), receiver.get_type());
            }
        }
    }

    #[test]
    fn fuzzer_builder_ensures_all_io_of_the_components_are_consumed() {
        let frontend1 = create_toy_frontend(
            5,
            vec![FuzzerIOType::TestCase],
            vec![FuzzerIOType::Feedback],
        );

        let frontend2 = create_toy_frontend(
            5,
            vec![FuzzerIOType::Feedback],
            vec![FuzzerIOType::MonitorData],
        );

        let fuzzer = FuzzerBuilder::new()
            .add_frontend("frontend1", frontend1)
            .add_frontend("frontend2", frontend2)
            .build();

        assert!(matches!(
            fuzzer,
            Err(FuzzerBuildError::ConnectionsInComplete(_))
        ));
    }

    #[test]
    fn fuzzer_builder_build_fuzzer_correctly() {
        let frontend1 = create_toy_frontend(
            5,
            vec![FuzzerIOType::TestCase],
            vec![FuzzerIOType::Feedback],
        );

        let frontend2 = create_toy_frontend(
            5,
            vec![FuzzerIOType::Feedback],
            vec![FuzzerIOType::MonitorData],
        );

        let frontend3 = create_toy_frontend(
            5,
            vec![FuzzerIOType::MonitorData],
            vec![FuzzerIOType::TestCase],
        );

        let fuzzer = FuzzerBuilder::new()
            .add_frontend("frontend1", frontend1)
            .add_frontend("frontend2", frontend2)
            .add_frontend("frontend3", frontend3)
            .add_connection(
                "frontend1",
                "frontend2",
                vec![ChannelIOType::Normal(FuzzerIOType::Feedback)],
            )
            .add_connection(
                "frontend2",
                "frontend3",
                vec![ChannelIOType::Normal(FuzzerIOType::MonitorData)],
            )
            .add_connection(
                "frontend3",
                "frontend1",
                vec![ChannelIOType::Normal(FuzzerIOType::TestCase)],
            )
            .build();

        assert!(fuzzer.is_ok());
    }
}
