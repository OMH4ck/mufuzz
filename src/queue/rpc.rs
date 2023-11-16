use crate::frontend::simple_frontend::{AsyncFrontend, BasicFrontend};
use crate::queue::frontend::SimpleQueueManagerFrontend;
use crate::rpc::msg_type::{Empty, Feedbacks, Number, TestCase, TestCases};
use crate::rpc::queue_management::queue_manager_server::{QueueManager, QueueManagerServer};
use std::net::SocketAddr;
use tonic::{Request, Response, Status};

pub use crate::rpc::queue_management::queue_manager_client::QueueManagerClient;

pub struct MyQueueManager {
    frontend: SimpleQueueManagerFrontend,
}

impl MyQueueManager {
    async fn new(worker_num: u32) -> Self {
        let mut frontend = SimpleQueueManagerFrontend::default();
        frontend.create_worker(worker_num);
        frontend.run().await;
        Self { frontend }
    }
}

#[tonic::async_trait]
impl QueueManager for MyQueueManager {
    async fn get_interesting_test_cases(
        &self,
        request: Request<Number>,
    ) -> Result<Response<TestCases>, Status> {
        let request = request.into_inner();
        let test_cases = self.frontend.get_results(Some(request.num)).await;
        Ok(Response::new(TestCases {
            test_cases: test_cases.into_iter().map(TestCase::from).collect(),
        }))
    }

    async fn post_interesting_test_cases(
        &self,
        request: Request<TestCases>,
    ) -> Result<Response<Empty>, Status> {
        let test_cases = request
            .into_inner()
            .test_cases
            .into_iter()
            .map(|test_case| test_case.into())
            .collect();
        self.frontend.handle_inputs(test_cases).await;
        Ok(Response::new(Empty::default()))
    }

    async fn post_test_case_feedbacks(
        &self,
        request: Request<Feedbacks>,
    ) -> Result<Response<Empty>, Status> {
        let feedbacks = request
            .into_inner()
            .feedbacks
            .into_iter()
            .map(|feedback| feedback.into())
            .collect();
        self.frontend.process_feedbacks(feedbacks).await;
        Ok(Response::new(Empty::default()))
    }
}

impl MyQueueManager {
    pub async fn create_service(worker_num: u32) -> QueueManagerServer<MyQueueManager> {
        QueueManagerServer::new(MyQueueManager::new(worker_num).await)
    }

    pub async fn run_service(server_addr: Option<String>, worker_num: u32) -> SocketAddr {
        crate::rpc::run_service_in_background(Self::create_service(worker_num).await, server_addr)
            .await
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::datatype;
    use crate::rpc::msg_type::{self, Number};
    use crate::rpc::queue_management::queue_manager_client::QueueManagerClient;

    #[tokio::test]
    async fn queue_manager_rpc_test() {
        let addr = MyQueueManager::run_service(None, 2).await;

        let mut client = QueueManagerClient::connect(format!("http://{}", addr))
            .await
            .unwrap();

        let test_cases_str = [
            "select 1;",
            "create table v(c); select c from v;",
            "select printf(a, b);",
        ];

        let request = tonic::Request::new(TestCases {
            test_cases: test_cases_str
                .iter()
                .map(|s| {
                    msg_type::TestCase::from(datatype::TestCase::new(s.as_bytes().to_vec(), 0))
                })
                .collect(),
        });

        let response = client.post_interesting_test_cases(request).await;
        assert!(response.is_ok());

        let request = tonic::Request::new(Number { num: 3 });
        let response = client
            .get_interesting_test_cases(request)
            .await
            .unwrap()
            .into_inner();
        assert_eq!(response.test_cases.len(), 3);

        let request = tonic::Request::new(Number { num: 4 });
        let response = client
            .get_interesting_test_cases(request)
            .await
            .unwrap()
            .into_inner();
        assert_eq!(response.test_cases.len(), 4);
    }
}
