use crate::frontend::*;
use crate::simple_frontend::*;
use async_trait::async_trait;
use std::sync::{Arc, Mutex};

#[derive(Default)]
struct ToyWorker {
    pub handled_workload: usize,
}

struct ToyFrontend {
    worker_pool: Vec<Arc<Mutex<ToyWorker>>>,
    sender: Option<AsyncSender<Vec<u32>>>,
    output_receiver: Option<AsyncReceiver<Vec<u32>>>,
    output_sender: Option<AsyncSender<Vec<u32>>>,
}

impl Worker for ToyWorker {
    type Input = u32;
    type Output = u32;
    fn handle_one_input(&mut self, input: Vec<u32>) -> Vec<u32> {
        self.handled_workload += input.len();
        input.into_iter().map(|x| x * x).collect()
    }
}

impl ToyFrontend {
    #[allow(dead_code)]
    fn new() -> Self {
        ToyFrontend {
            worker_pool: Vec::default(),
            sender: None,
            output_receiver: None,
            output_sender: None,
        }
    }
}

impl BasicFrontend for ToyFrontend {
    type Worker = ToyWorker;
    type Output = u32;

    fn create_worker(&mut self, num: u32) {
        for _i in 0..num {
            self.worker_pool
                .push(Arc::new(Mutex::new(ToyWorker::default())));
        }
    }

    fn output_transfrom(&self, input: Vec<u32>) -> Vec<u32> {
        input
    }

    crate::frontend_default!();
}

impl AsyncFrontendChannel for ToyFrontend {
    crate::async_frontend_default!();
}

#[async_trait]
impl AsyncFrontend for ToyFrontend {}

#[cfg(test)]
mod tests {
    use super::*;
    use tokio::time::{sleep, Duration};

    #[tokio::test]
    async fn frontend_evenly_distribute_workload_to_workers() {
        let mut toy_frontend = ToyFrontend::new();
        toy_frontend.create_worker(10);
        toy_frontend.run().await;
        let mut join_handle = Vec::new();
        for _j in 0..10 {
            join_handle.push(async {
                let mut inputs = Vec::new();
                for i in 0..10 {
                    inputs.push(i);
                }
                toy_frontend.handle_inputs(inputs).await;
            });
        }
        futures::future::join_all(join_handle).await;
        sleep(Duration::from_millis(10)).await;

        for worker in toy_frontend.get_worker_pool() {
            let worker = worker.lock().unwrap();
            assert_eq!(worker.handled_workload, 10);
        }
    }

    #[tokio::test]
    async fn frontend_gather_output_from_all_workers() {
        let mut toy_frontend = ToyFrontend::new();
        toy_frontend.create_worker(10);
        toy_frontend.run().await;
        let mut join_handle = Vec::new();
        for _j in 0..10 {
            join_handle.push(async {
                let mut inputs = Vec::new();
                for i in 0..10 {
                    inputs.push(i);
                }
                toy_frontend.handle_inputs(inputs).await;
            });
        }
        futures::future::join_all(join_handle).await;
        sleep(Duration::from_millis(10)).await;

        let result = toy_frontend.get_results(None).await;
        assert_eq!(result.len(), 100);
    }

    #[test]
    fn forkserver_executor_frontend_add_delete_worker_correctly() {
        let mut toy_frontend = ToyFrontend::new();
        toy_frontend.create_worker(15);

        assert!(toy_frontend.worker_pool.len() == 15);
        toy_frontend.delete_worker(5);
        assert!(toy_frontend.worker_pool.len() == 10);
    }

    #[tokio::test]
    #[should_panic]
    async fn fronted_panic_at_handling_inputs_when_no_worker_is_created() {
        let toy_frontend = ToyFrontend::new();
        toy_frontend.handle_inputs(vec![0, 1]).await;
    }

    #[tokio::test]
    #[should_panic]
    async fn fronted_panic_at_handling_inputs_when_it_is_not_run() {
        let mut toy_frontend = ToyFrontend::new();
        toy_frontend.create_worker(10);
        toy_frontend.handle_inputs(vec![0, 1]).await;
    }
}
