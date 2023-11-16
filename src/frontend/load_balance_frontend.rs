use super::{FuzzerIO, FuzzerIOType};
use itertools::izip;
//use rand::rngs::StdRng;
//use rand::Rng;
//use rand::distributions::{Bernoulli, Distribution};
//use rand::SeedableRng;
//use serde_json::Value;
use std::collections::HashMap;
use std::sync::{Arc, Mutex};

pub use channel_mod::channel as frontend_channel;
use tokio::sync::mpsc as channel_mod;
pub type AsyncExclusiveReceiver<T> = channel_mod::Receiver<T>;
pub type AsyncExclusiveSender<T> = channel_mod::Sender<T>;
pub type AsyncWatchReceiver<T> = tokio::sync::watch::Receiver<T>;

#[derive(Debug)]
pub struct LessContentionMPSCChannel<T> {
    senders: Vec<AsyncExclusiveSender<T>>,
    receivers: Vec<AsyncExclusiveReceiver<T>>,
    //channel_size: usize,
    round_idx: usize,
}

impl<T> LessContentionMPSCChannel<T> {
    pub fn new(sender_num: usize, channel_size: usize) -> Self {
        let size_per_channel = (channel_size + sender_num - 1) / sender_num;
        let mut senders = Vec::default();
        let mut receivers = Vec::default();
        assert!(sender_num % 5 == 0 || sender_num < 5);
        let step_size = if sender_num > 5 { 5 } else { 1 };
        for _i in (0..sender_num).step_by(step_size) {
            let (sx, rx) = frontend_channel::<T>(size_per_channel);
            for _j in 0..step_size {
                senders.push(sx.clone());
            }
            receivers.push(rx);
        }
        Self {
            //channel_size,
            senders,
            receivers,
            round_idx: 0,
        }
    }

    pub fn take_senders(&mut self) -> Vec<AsyncExclusiveSender<T>> {
        self.senders.clone()
    }

    pub async fn recv(&mut self) -> Option<T> {
        let idx_save = self.round_idx;
        while self.round_idx != idx_save {
            if let Ok(res) = self.receivers[self.round_idx].try_recv() {
                self.round_idx += 1;
                self.round_idx %= self.receivers.len();
                return Some(res);
            }
            self.round_idx += 1;
            self.round_idx %= self.receivers.len();
        }

        let mut all_futures = Vec::with_capacity(self.receivers.len());
        for receiver in self.receivers.iter_mut() {
            all_futures.push(Box::pin(receiver.recv()));
        }
        let (inputs, idx, _) = futures::future::select_all(all_futures).await;
        self.round_idx = idx + 1;
        self.round_idx %= self.receivers.len();
        assert!(inputs.is_some());
        inputs
    }
}

pub trait Worker {
    // One input can produce multiple outputs.
    fn handle_inputs(&mut self, input: FuzzerIO) -> Option<Vec<FuzzerIO>>;
    fn non_blocking_generate_output(&mut self, _gen_type: FuzzerIOType) -> Option<FuzzerIO> {
        None
    }
}

#[derive(Clone)]
pub enum Sender {
    AsyncExclusive(AsyncExclusiveSender<FuzzerIO>),
    //AsyncBroadCast(AsyncBroadCastSender<FuzzerIO>),
}

impl Sender {
    pub async fn send(
        &self,
        item: FuzzerIO,
    ) -> Result<(), channel_mod::error::SendError<FuzzerIO>> {
        match self {
            Self::AsyncExclusive(sender) => {
                assert!(!sender.is_closed());
                sender.send(item).await
            }
        }
    }
}

pub enum Receiver {
    AsyncExclusive(AsyncExclusiveReceiver<FuzzerIO>),
    AsyncLessContention(LessContentionMPSCChannel<FuzzerIO>),
    //AsyncBroadCast(AsyncBroadCastReceiver<FuzzerIO>),
    AsyncWatch(AsyncWatchReceiver<FuzzerIO>),
}

impl Receiver {
    pub async fn recv(&mut self) -> Option<FuzzerIO> {
        match self {
            Self::AsyncExclusive(receiver) => receiver.recv().await,
            Self::AsyncLessContention(receiver) => receiver.recv().await,
            //Self::AsyncBroadCast(receiver) => receiver.recv().await,
            Self::AsyncWatch(receiver) => {
                if receiver.changed().await.is_ok() {
                    Some(receiver.borrow().clone())
                } else {
                    None
                }
            }
        }
    }
}

pub trait Transformer {
    fn transform(&self, input: FuzzerIO) -> FuzzerIO {
        input
    }
}

pub trait SyncSource {
    fn update_status(&mut self, input: FuzzerIO);
    fn get_status(&mut self) -> FuzzerIO;
}

pub struct DummySyncSource {
    dummy_status: FuzzerIO,
}

impl SyncSource for DummySyncSource {
    fn update_status(&mut self, _input: FuzzerIO) {}
    fn get_status(&mut self) -> FuzzerIO {
        self.dummy_status.clone()
    }
}

#[derive(Clone, Copy)]
pub enum ChannelType {
    WatchInput,
    NormalInput,
    LessContentionOutput,
    NonBlockingOutput,
    NormalOutput,
}

pub struct Frontend<
    WorkerT: Worker + Send + 'static,
    TransformerT: Transformer + Send + Sync + 'static,
    SyncSourcer: SyncSource + Send + Sync + 'static,
> {
    worker_pool: Vec<Arc<Mutex<WorkerT>>>,
    input_receivers: Vec<Receiver>,
    watch_receivers: Vec<(Receiver, SyncSourcer)>,
    // watch_inner_receivers: Vec<Vec<Receiver>>,
    // inner_receivers: Vec<Vec<Receiver>>,
    less_contention_sender: Vec<Vec<(Sender, FuzzerIOType)>>,
    output_senders: HashMap<FuzzerIOType, Vec<Sender>>,
    non_blocking_output_senders: HashMap<FuzzerIOType, Vec<Sender>>,
    transformer: Option<Arc<TransformerT>>,
    worker_creator: Box<dyn Fn() -> WorkerT>,
    name: String,
}

impl<
        WorkerT: Worker + Send + 'static,
        TransformerT: Transformer + Send + Sync + 'static,
        SyncSourcer: SyncSource + Send + Sync + 'static,
    > Frontend<WorkerT, TransformerT, SyncSourcer>
{
    pub fn new(f: Box<dyn Fn() -> WorkerT>) -> Self {
        Self {
            worker_pool: Vec::default(),
            input_receivers: Vec::default(),
            output_senders: HashMap::default(),
            watch_receivers: Vec::default(),
            less_contention_sender: Vec::default(),
            non_blocking_output_senders: HashMap::default(),
            transformer: None,
            worker_creator: f,
            name: String::from("Default frontend"),
        }
    }

    pub fn set_name(&mut self, name: String) {
        self.name = name;
    }

    pub fn get_name(&self) -> String {
        self.name.clone()
    }

    pub fn create_worker(&mut self, n: u32) {
        for _i in 0..n {
            self.worker_pool
                .push(Arc::new(Mutex::new((self.worker_creator)())));
        }
    }

    pub fn set_transformer(&mut self, transformer: Option<Arc<TransformerT>>) {
        self.transformer = transformer;
    }

    // Publisher/Subscriber channel. This is currently used for distributing mutation/
    // testcases score change. A sub-component called SyncSourcer keeps some status that
    // all the workers of some other component want. Instead of broadcasting it to all
    // workers (which will not scale when the number of workers increase), we asks the
    // worker to subscribe to the sync source. The sync source can publish new status at
    // any monent, the workers just need to ocassionally sync the status.
    pub fn register_watch_input_handler(&mut self, input: Receiver, sync_sourcer: SyncSourcer) {
        match input {
            Receiver::AsyncExclusive(_) => {
                self.watch_receivers.push((input, sync_sourcer));
            }
            Receiver::AsyncWatch(_) | Receiver::AsyncLessContention(_) => {
                unreachable!();
            }
        }
    }

    fn bridge_watch_handlers(&mut self) -> Vec<Vec<Receiver>> {
        if self.watch_receivers.is_empty() {
            return (0..self.worker_pool.len())
                .map(|_i| Vec::default())
                .collect();
        }
        let mut inner_receivers = Vec::default();
        for (receiver, mut sync_source) in self.watch_receivers.drain(..) {
            let (sx, rx) =
                tokio::sync::watch::channel(FuzzerIO::TestCaseScoreChange(Vec::default()));
            inner_receivers.push(rx);
            let mut receiver = match receiver {
                Receiver::AsyncExclusive(receiver) => receiver,
                _ => {
                    unreachable!();
                }
            };
            tokio::spawn(async move {
                let mut counter = 0;
                while let Some(v) = receiver.recv().await {
                    sync_source.update_status(v);
                    if counter % 10 == 0 {
                        sx.send(sync_source.get_status()).unwrap();
                        counter = 1;
                    } else {
                        counter += 1;
                    }
                }
            });
        }
        (0..self.worker_pool.len())
            .map(|_i| {
                inner_receivers
                    .iter()
                    .map(|x| Receiver::AsyncWatch(x.clone()))
                    .collect()
            })
            .collect()
    }

    pub fn register_input_handler(&mut self, input: Receiver) {
        match input {
            Receiver::AsyncExclusive(_) | Receiver::AsyncLessContention(_) => {
                self.input_receivers.push(input);
            }
            Receiver::AsyncWatch(_) => {
                unreachable!();
            }
        }
    }

    pub fn register_non_blocking_output_handler(&mut self, key_type: FuzzerIOType, output: Sender) {
        self.non_blocking_output_senders
            .entry(key_type)
            .or_default()
            .push(output);
    }

    pub fn register_output_handler(&mut self, key_type: FuzzerIOType, output: Sender) {
        self.output_senders
            .entry(key_type)
            .or_default()
            .push(output);
    }

    pub fn register_less_contention_sender(
        &mut self,
        senders: Vec<Sender>,
        fuzzer_io_type: FuzzerIOType,
    ) {
        if self.less_contention_sender.is_empty() {
            self.less_contention_sender = vec![Vec::default(); self.worker_pool.len()];
        }
        if senders.len() != self.worker_pool.len() {
            println!("Here, {:#?}", fuzzer_io_type);
        }
        assert_eq!(senders.len(), self.worker_pool.len());
        for (idx, sender) in senders.into_iter().enumerate() {
            self.less_contention_sender[idx].push((sender, fuzzer_io_type));
        }
    }

    pub fn get_worker_num(&self) -> usize {
        self.worker_pool.len()
    }

    // Currently only used for queue manager. Since the components might produce multiple
    // types of outputs. When you check the communications bewteen components, you will
    // find some loops. Before we find a better solution, we need to ensure that each loop
    // at least has one communication edge is not blocking (That is, what it output is
    // not driven by the inputs it accept).
    // TODO(yongheng): Unify the special handling.
    fn run_non_blocking_output_senders(&self) {
        if self.non_blocking_output_senders.is_empty() {
            return;
        }
        let senders = self.non_blocking_output_senders.clone();
        let workers = self.worker_pool.clone();
        tokio::spawn(async move {
            let mut has_generated_once = false;
            loop {
                for (output_type, output_senders) in senders.iter() {
                    for output_sender in output_senders.iter() {
                        let mut output_generated = false;
                        for worker in workers.iter() {
                            let output = {
                                let mut worker = worker.lock().unwrap();
                                worker.non_blocking_generate_output(*output_type)
                            };
                            if let Some(output) = output {
                                output_generated = true;
                                assert_eq!(output.get_type(), *output_type);
                                output_sender.send(output).await.unwrap();
                            }
                        }
                        if !output_generated {
                            tokio::time::sleep(tokio::time::Duration::from_millis(5)).await;
                            if has_generated_once {
                                unreachable!();
                            }
                        } else {
                            has_generated_once = true;
                        }
                    }
                }
            }
        });
    }

    // For normal inputs (which should be handled by only one worker), we use round robin
    // to dispatch them to different workers.
    // TODO(yongheng): Maybe we can make it work stealing.
    fn run_dispactcher_thread(&mut self) -> Vec<Vec<Receiver>> {
        let mut inner_receivers = Vec::default();
        for _i in 0..self.worker_pool.len() {
            inner_receivers.push(Vec::default());
        }
        for mut receiver in self.input_receivers.drain(..) {
            let mut senders = Vec::default();

            // For each receiver, we bridge a channel between this receiver and the worker.
            for receivers in inner_receivers.iter_mut() {
                let (sx, rx) = frontend_channel(2);
                senders.push(sx);
                receivers.push(Receiver::AsyncExclusive(rx));
            }
            let name = self.name.clone();
            tokio::spawn(async move {
                let mut robin_idx = 0;
                while let Some(inputs) = receiver.recv().await {
                    let old_idx = robin_idx;
                    loop {
                        if let Ok(permit) = senders[robin_idx].try_reserve() {
                            permit.send(inputs);
                            break;
                        }
                        robin_idx = (robin_idx + 1) % senders.len();

                        // If we cannot find a non-full channel, we just wait.
                        if robin_idx == old_idx {
                            if let Err(r) = senders[robin_idx].send(inputs).await {
                                println!("Err in {}, {}", name, r);
                                unreachable!();
                            }
                            break;
                        }
                    }
                    robin_idx = (robin_idx + 1) % senders.len();
                }
            });
        }
        inner_receivers
    }

    async fn run_a_worker_thread(
        senders: HashMap<FuzzerIOType, Vec<Sender>>,
        receivers: Vec<Receiver>,
        worker: Arc<Mutex<WorkerT>>,
        transformer: Option<Arc<TransformerT>>,
        _name: String,
        _id: i32,
    ) {
        let mut receivers = receivers;
        //let mut rng = StdRng::from_entropy();
        //let d = Bernoulli::new(0.9).unwrap();
        loop {
            let mut all_futures = Vec::with_capacity(receivers.len());
            for receiver in receivers.iter_mut() {
                all_futures.push(Box::pin(receiver.recv()));
            }
            let (inputs, _, _) = futures::future::select_all(all_futures).await;
            //println!("{} get inputs!", name);

            if inputs.is_none() {
                println!("Bye");
                break;
            }

            let inputs = inputs.unwrap();
            if inputs.is_empty() {
                continue;
            }
            /*
            println!(
                "{} worker {} working, getting type {:?}",
                _name,
                id,
                inputs.get_type()
            );
            */

            let outputs = {
                let mut worker_inner = worker.lock().unwrap();
                worker_inner.handle_inputs(inputs)
            };

            //assert!(has_non_blocking_output_sender || outputs.is_some());
            if let Some(outputs) = outputs {
                for output in outputs.into_iter() {
                    let output = match transformer.as_ref() {
                        Some(transformer) => transformer.transform(output),
                        None => output,
                    };
                    if output.is_empty() {
                        continue;
                    }
                    let outputs_type = output.get_type();
                    assert!(
                        senders.contains_key(&outputs_type),
                        "Every type of output should be handled!"
                    );
                    let specific_senders = senders.get(&outputs_type).unwrap();
                    let output_copies = vec![output; specific_senders.len()];
                    for (sender, output) in specific_senders.iter().zip(output_copies.into_iter()) {
                        sender.send(output).await.unwrap();
                    }
                }
            }
            tokio::task::yield_now().await;
        }
    }
    /// Run the frontend as a service. Afterwards, we should not modify the frontend.
    pub fn run(&mut self) {
        // Run the non blocking senders if any.
        if !self.non_blocking_output_senders.is_empty() {
            self.run_non_blocking_output_senders();
        }

        if self.less_contention_sender.is_empty() {
            self.less_contention_sender = vec![Vec::default(); self.worker_pool.len()];
        }

        // Bridge the receivers.
        let all_bridge_watch_handlers = self.bridge_watch_handlers();
        let all_inner_receivers = self.run_dispactcher_thread();
        for (idx, (worker, inner_receivers, bridge_watch_handlers, less_contention_senders)) in
            izip!(
                self.worker_pool.clone().into_iter(),
                all_inner_receivers.into_iter(),
                all_bridge_watch_handlers.into_iter(),
                self.less_contention_sender.split_off(0).into_iter()
            )
            .enumerate()
        {
            let mut all_senders = self.output_senders.clone();
            for (sender, fuzzer_io_type) in less_contention_senders.into_iter() {
                all_senders
                    .entry(fuzzer_io_type)
                    .or_default()
                    .push(sender);
            }
            let all_receivers = inner_receivers
                .into_iter()
                .chain(bridge_watch_handlers.into_iter())
                .collect::<Vec<_>>();
            assert!(!all_receivers.is_empty());
            let transformer = self.transformer.clone();
            //let name = self.get_name();
            tokio::spawn(Self::run_a_worker_thread(
                all_senders,
                all_receivers,
                worker,
                transformer,
                self.get_name(),
                idx.try_into().unwrap(),
            ));
        }
    }
}

/// Connect two components for communication.
/// `from` produces output, which becomes the input of `to`. `output_channel_type` and
/// `input_channel_type` controls the behaviors of the communication. `io_type` is the
/// type produced by the `from` and accepted by `to`.
/// This function returns a sender for injecting some initial inputs in the channel.
pub fn connect_component<
    WorkerT: Worker + Send + 'static,
    TransformerT: Transformer + Send + Sync + 'static,
    SyncSourcer: SyncSource + Send + Sync + 'static,
    WorkerT2: Worker + Send + 'static,
    TransformerT2: Transformer + Send + Sync + 'static,
    SyncSourcer2: SyncSource + Send + Sync + 'static,
>(
    from: &mut Frontend<WorkerT, TransformerT, SyncSourcer>,
    to: &mut Frontend<WorkerT2, TransformerT2, SyncSourcer2>,
    io_type: FuzzerIOType,
    output_channel_type: ChannelType,
    // input_channel_type: ChannelType,
    sync_source: Option<SyncSourcer2>,
) -> Sender {
    let channel_cap: usize = 500;
    match output_channel_type {
        ChannelType::NormalOutput | ChannelType::NonBlockingOutput => {
            let (sender, receiver) = frontend_channel(channel_cap);
            if let Some(sync_source) = sync_source {
                to.register_watch_input_handler(Receiver::AsyncExclusive(receiver), sync_source);
            } else {
                to.register_input_handler(Receiver::AsyncExclusive(receiver));
            }
            /*
            if let Some(injected_inputs) = injected_inputs {
                for input in injected_inputs {
                    sender.send(input).await.unwrap();
                }
            }
            */

            let sender = Sender::AsyncExclusive(sender);
            if matches!(output_channel_type, ChannelType::NormalOutput) {
                from.register_output_handler(io_type, sender.clone());
            } else {
                from.register_non_blocking_output_handler(io_type, sender.clone());
            }
            sender
        }
        ChannelType::LessContentionOutput => {
            // assert!(injected_inputs.is_none(), "Unhandled yet");
            let from_worker_num = from.get_worker_num();
            let mut less_contention_channel =
                LessContentionMPSCChannel::new(from_worker_num, 50 * from_worker_num);
            let all_senders = less_contention_channel
                .take_senders()
                .into_iter()
                .map(Sender::AsyncExclusive)
                .collect::<Vec<_>>();
            let sender = all_senders[0].clone();
            from.register_less_contention_sender(all_senders, io_type);
            assert!(sync_source.is_none());
            to.register_input_handler(Receiver::AsyncLessContention(less_contention_channel));
            sender
        }
        _ => {
            unreachable!();
        }
    }
}
