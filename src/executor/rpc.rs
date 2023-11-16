use crate::executor::frontend::ForkServerExecutorFrontend;
use crate::frontend::simple_frontend::{AsyncFrontend, BasicFrontend};
use crate::rpc::execution::executor_server::{Executor, ExecutorServer};
use crate::rpc::msg_type::{Empty, Feedback, Feedbacks, Number, TestCases};
use std::net::SocketAddr;
use std::sync::Arc;
use tokio::sync::RwLock;
use tonic::{Request, Response, Status};

pub use crate::rpc::execution::executor_client::ExecutorClient;

pub struct MyExecutor {
    frontend: Arc<RwLock<ForkServerExecutorFrontend>>,
    //frontend: Arc<ForkServerExecutorFrontend>,
    //frontend: ForkServerExecutorFrontend,
}

impl MyExecutor {
    async fn new(
        worker_num: u32,
        bitmap_size: usize,
        args: Vec<String>,
        timeout: u64,
        working_dir: Option<String>,
    ) -> Self {
        let mut frontend = ForkServerExecutorFrontend::new(bitmap_size, args, timeout, working_dir);
        frontend.create_worker(worker_num);
        frontend.run().await;
        Self {
            //frontend,
            //frontend: Arc::new(frontend),
            frontend: Arc::new(RwLock::new(frontend)),
        }
    }
}

#[tonic::async_trait]
impl Executor for MyExecutor {
    async fn get_feedbacks(&self, request: Request<Number>) -> Result<Response<Feedbacks>, Status> {
        let request = request.into_inner();
        let num = request.num;
        let frontend = self.frontend.clone();
        let feedbacks = tokio::task::spawn(async move {
            let feedbacks = frontend.read().await.get_results(Some(num)).await;
            //let feedbacks = self.frontend.read().unwrap().get_results(Some(num)).await;
            feedbacks
                .into_iter()
                .map(Feedback::from)
                .collect::<Vec<Feedback>>()
        })
        .await
        .unwrap();
        Ok(Response::new(Feedbacks { feedbacks }))
        //Ok(Response::new(Feedbacks::default()))
    }

    async fn execute_test_cases(
        &self,
        request: Request<TestCases>,
    ) -> Result<Response<Empty>, Status> {
        let request = request.into_inner();
        let inputs = request
            .test_cases
            .into_iter()
            .map(|test_case| test_case.into())
            .collect::<Vec<crate::datatype::TestCase>>();
        let frontend = self.frontend.clone();
        frontend.read().await.handle_inputs(inputs).await;
        Ok(Response::new(Empty::default()))
    }
}

impl MyExecutor {
    pub async fn create_service(
        worker_num: u32,
        bitmap_size: usize,
        args: Vec<String>,
        timeout: u64,
        working_dir: Option<String>,
    ) -> ExecutorServer<MyExecutor> {
        ExecutorServer::new(
            MyExecutor::new(worker_num, bitmap_size, args, timeout, working_dir).await,
        )
    }

    pub async fn run_service(
        server_addr: Option<String>,
        worker_num: u32,
        args: Vec<String>,
        map_size: usize,
        timeout: u64,
        working_dir: Option<String>,
    ) -> SocketAddr {
        crate::rpc::run_service_in_background(
            Self::create_service(worker_num, map_size, args, timeout, working_dir).await,
            server_addr,
        )
        .await
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::rpc::execution::executor_client::ExecutorClient;
    use crate::rpc::msg_type::{Number, TestCase, TestCases};
    use crate::util;

    #[tokio::test]
    #[serial_test::serial]
    async fn executor_rpc_test() {
        const MAP_SIZE: usize = 65536;
        let args = vec![util::get_test_bin_by_name("simple_prog").unwrap()];
        let addr = MyExecutor::run_service(None, 1, args, MAP_SIZE, 20, None).await;

        let mut client = ExecutorClient::connect(format!("http://{}", addr))
            .await
            .unwrap();

        let test_cases = ["select 1;", "create table v2;"]
            .iter()
            .map(|str| TestCase {
                content: str.as_bytes().to_vec(),
                pid: 0,
            })
            .collect();

        let request = tonic::Request::new(TestCases { test_cases });
        let response = client.execute_test_cases(request).await;
        assert!(response.is_ok());

        let request = tonic::Request::new(Number { num: 2 });
        let feedback = client.get_feedbacks(request).await.unwrap().into_inner();
        assert_eq!(feedback.feedbacks.len(), 2);
        println!("{:#?}", feedback.feedbacks);
    }
}
