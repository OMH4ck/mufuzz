use crate::frontend::simple_frontend::{AsyncFrontend, BasicFrontend};
use crate::mutator::frontend::BitFlipMutatorFrontend;
use crate::rpc::msg_type::{Empty, Feedbacks, Number, TestCase, TestCases};
use crate::rpc::mutation::mutator_server::{Mutator, MutatorServer};
use std::net::SocketAddr;
use tonic::{Request, Response, Status};

pub use crate::rpc::mutation::mutator_client::MutatorClient;
pub struct MyMutator {
    frontend: BitFlipMutatorFrontend,
}

impl MyMutator {
    async fn new(worker_num: u32) -> Self {
        let mut frontend = BitFlipMutatorFrontend::default();
        frontend.create_worker(worker_num);
        frontend.run().await;
        MyMutator { frontend }
    }
}

#[tonic::async_trait]
impl Mutator for MyMutator {
    async fn mutate_test_cases(
        &self,
        request: Request<TestCases>,
    ) -> Result<Response<Number>, Status> {
        let request = request.into_inner();

        let num = request.test_cases.len();

        self.frontend
            .handle_inputs(request.test_cases.into_iter().map(TestCase::into).collect())
            .await;

        Ok(Response::new(Number { num: num as u32 }))
    }

    async fn get_mutated_test_cases(
        &self,
        request: Request<Number>,
    ) -> Result<Response<TestCases>, Status> {
        let num = request.into_inner().num;
        let test_cases = self.frontend.get_results(Some(num)).await;
        let test_cases = TestCases {
            test_cases: test_cases.into_iter().map(TestCase::from).collect(),
        };

        Ok(Response::new(test_cases))
    }

    async fn post_mutation_feedbacks(
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

impl MyMutator {
    pub async fn create_service(worker_num: u32) -> MutatorServer<MyMutator> {
        MutatorServer::new(MyMutator::new(worker_num).await)
    }

    pub async fn run_service(server_addr: Option<String>, worker_num: u32) -> SocketAddr {
        crate::rpc::run_service_in_background(Self::create_service(worker_num).await, server_addr)
            .await
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[tokio::test]
    async fn mutation_rpc_test() {
        let addr = MyMutator::run_service(None, 2).await;

        let mut client = MutatorClient::connect(format!("http://{}", addr))
            .await
            .unwrap();

        let mut test_cases = TestCases::default();
        test_cases.test_cases.push(TestCase {
            content: (0..255).collect::<Vec<u8>>(),
            pid: 0,
        });
        test_cases.test_cases.push(TestCase {
            content: (0..255).rev().collect::<Vec<u8>>(),
            pid: 0,
        });
        let request = tonic::Request::new(test_cases);

        let response = client
            .mutate_test_cases(request)
            .await
            .unwrap()
            .into_inner();
        assert!(response.num == 2);

        let test_case_num = 500;
        let request = tonic::Request::new(Number { num: test_case_num });

        let response = client
            .get_mutated_test_cases(request)
            .await
            .unwrap()
            .into_inner();
        println!("{}", response.test_cases.len());
        assert!(response.test_cases.len() > 90);
    }
}
