pub use async_channel::bounded as frontend_channel;
use async_trait::async_trait;
use std::sync::{Arc, Mutex};
pub type AsyncReceiver<T> = async_channel::Receiver<T>;
pub type AsyncSender<T> = async_channel::Sender<T>;
use crate::frontend::Worker;
use serde_json::Value;

// This macro generates the trait impl that are the same in all frontend.
// TODO(yongheng): I don't think this is a good practice and it might make the code difficult to debug.
#[macro_export]
macro_rules! async_frontend_default {
    () => {
        fn get_async_output_receiver(
            &self,
        ) -> &AsyncReceiver<Vec<<Self::Worker as $crate::frontend::Worker>::Output>> {
            self.output_receiver.as_ref().unwrap()
        }

        fn set_async_output_receiver(
            &mut self,
            receiver: AsyncReceiver<Vec<<Self::Worker as $crate::frontend::Worker>::Output>>,
        ) {
            self.output_receiver = Some(receiver);
        }

        fn set_async_output_sender(
            &mut self,
            sender: AsyncSender<Vec<<Self::Worker as $crate::frontend::Worker>::Output>>,
        ) {
            self.output_sender = Some(sender);
        }

        fn set_async_input_sender(
            &mut self,
            input: AsyncSender<Vec<<Self::Worker as $crate::frontend::Worker>::Input>>,
        ) {
            self.sender = Some(input);
        }

        fn get_async_input_sender(
            &self,
        ) -> AsyncSender<Vec<<Self::Worker as $crate::frontend::Worker>::Input>> {
            self.sender.as_ref().unwrap().clone()
        }
    };
}

#[macro_export]
macro_rules! frontend_default {
    () => {
        fn get_worker_pool(&self) -> &Vec<Arc<Mutex<Self::Worker>>> {
            &self.worker_pool
        }

        fn get_worker_pool_mut(&mut self) -> &mut Vec<Arc<Mutex<Self::Worker>>> {
            &mut self.worker_pool
        }
    };
}

pub trait BasicFrontend {
    type Worker: crate::frontend::Worker + 'static;
    type Output;

    // We don't provide default implementation because some worker requires
    // extra arguments for creation.
    fn create_worker(&mut self, num: u32);

    fn delete_worker(&mut self, num: u32) {
        let worker = self.get_worker_pool_mut();
        let final_len = worker.len().saturating_sub(num as usize);
        worker.truncate(final_len);
    }

    fn get_worker_by_idx(&self, idx: usize) -> Arc<Mutex<Self::Worker>> {
        self.get_worker_pool()[idx].clone()
    }

    fn get_worker_pool_mut(&mut self) -> &mut Vec<Arc<Mutex<Self::Worker>>>;
    fn get_worker_pool(&self) -> &Vec<Arc<Mutex<Self::Worker>>>;

    // Transform the output of the worker to that of frontend. In most case it does nothing.
    fn output_transfrom(
        &self,
        input: Vec<<Self::Worker as crate::frontend::Worker>::Output>,
    ) -> Vec<Self::Output>;
}

pub trait AsyncFrontendChannel: BasicFrontend {
    fn set_async_input_sender(
        &mut self,
        input: AsyncSender<Vec<<Self::Worker as crate::frontend::Worker>::Input>>,
    );

    fn get_async_input_sender(
        &self,
    ) -> AsyncSender<Vec<<Self::Worker as crate::frontend::Worker>::Input>>;

    fn set_async_output_receiver(
        &mut self,
        receiver: AsyncReceiver<Vec<<Self::Worker as crate::frontend::Worker>::Output>>,
    );

    fn set_async_output_sender(
        &mut self,
        sender: AsyncSender<Vec<<Self::Worker as crate::frontend::Worker>::Output>>,
    );

    fn get_async_output_receiver(
        &self,
    ) -> &AsyncReceiver<Vec<<Self::Worker as crate::frontend::Worker>::Output>>;
}

#[async_trait]
pub trait AsyncFrontend: AsyncFrontendChannel {
    async fn run(&mut self) {
        let worker_pool = self.get_worker_pool().clone();
        let (input_sx, input_rx) = frontend_channel(1000);
        let (output_sender, output_receiver) = frontend_channel(100);
        self.set_async_output_receiver(output_receiver);
        self.set_async_output_sender(output_sender.clone());
        self.set_async_input_sender(input_sx);
        tokio::spawn(async move {
            let (worker_sx, worker_rx) = frontend_channel(worker_pool.len());
            for worker in worker_pool.into_iter() {
                assert!(worker_sx.send(worker).await.is_ok());
            }

            while let Ok(inputs) = input_rx.recv().await {
                if let Ok(worker) = worker_rx.recv().await {
                    let save_worker_sx = worker_sx.clone();
                    let output_sender_tmp = output_sender.clone();
                    tokio::spawn(async move {
                        let send_back = worker.clone();
                        let worker_output = {
                            let mut worker = worker.lock().unwrap();
                            worker.handle_one_input(inputs)
                        };
                        if !worker_output.is_empty() {
                            assert!(output_sender_tmp.send(worker_output).await.is_ok());
                        }
                        assert!(save_worker_sx.send(send_back).await.is_ok());
                    });
                }
            }
        });
    }

    // Handle the inputs and save the result into the pool.
    // The inputs are splitted evenly and distributed to every workers.
    async fn handle_inputs(&self, inputs: Vec<<Self::Worker as crate::frontend::Worker>::Input>) {
        if inputs.is_empty() {
            return;
        }

        let mut inputs = inputs;
        let worker_len = self.get_worker_pool().len();
        assert_ne!(worker_len, 0);
        let mut split_workload = inputs.len() / worker_len;
        if inputs.len() % worker_len != 0 {
            split_workload += 1;
        }
        while !inputs.is_empty() {
            let sub_input = inputs.split_off(inputs.len().saturating_sub(split_workload));
            assert!(self.get_async_input_sender().send(sub_input).await.is_ok());
        }
    }

    // Fetch some results from the pool with best effort. It doesn't promise to
    // fetch as many result as you want. To avoid returning nothing, we try to wait for
    // at least one result to be produced.
    async fn get_results(&self, num: Option<u32>) -> Vec<Self::Output> {
        let num = num.unwrap_or(2000) as usize;
        let receiver = self.get_async_output_receiver();
        //let num = num.unwrap_or(receiver.len() as u32) as usize;

        let mut result = Vec::with_capacity(num);
        while result.len() < num {
            // Result avaible?
            if let Ok(some_result) = receiver.try_recv() {
                result.extend(some_result.into_iter());
            }
            // If not and we have nothing now, we wait for some.
            else if result.is_empty() {
                if let Ok(some_result) = receiver.recv().await {
                    result.extend(some_result.into_iter());
                }
            } else {
                break;
            }
        }
        self.output_transfrom(result)
    }

    async fn retrieve_monitor_data(&self) -> Vec<Value> {
        unreachable!();
    }
}
