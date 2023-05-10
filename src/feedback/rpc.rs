use crate::feedback::frontend::BitmapCollectorFrontend;
use crate::frontend::simple_frontend::{AsyncFrontend, BasicFrontend};
use crate::rpc::feedback::feedback_collector_server::{FeedbackCollector, FeedbackCollectorServer};
use crate::rpc::msg_type::{self, Empty, Feedbacks, MonitorData, Number, TestCases};
use std::net::SocketAddr;
use tonic::{Request, Response, Status};

pub use crate::rpc::feedback::feedback_collector_client::FeedbackCollectorClient;

pub struct MyFeedbackCollector {
    frontend: BitmapCollectorFrontend,
}

impl MyFeedbackCollector {
    pub async fn new(worker_num: u32, bitmap_size: usize) -> Self {
        let mut frontend = BitmapCollectorFrontend::new(bitmap_size);
        frontend.create_worker(worker_num);
        frontend.run().await;
        Self { frontend }
    }
}

#[tonic::async_trait]
impl FeedbackCollector for MyFeedbackCollector {
    async fn retrieve_monitor_data(
        &self,
        _request: Request<Empty>,
    ) -> Result<Response<MonitorData>, Status> {
        let monitor_data = self.frontend.retrieve_monitor_data().await;
        Ok(Response::new(MonitorData::from(monitor_data)))
    }

    async fn check_feedbacks(
        &self,
        request: Request<Feedbacks>,
    ) -> Result<Response<Empty>, Status> {
        let feedbacks = request
            .into_inner()
            .feedbacks
            .into_iter()
            .map(|feedback| feedback.into())
            .collect();
        self.frontend.handle_inputs(feedbacks).await;
        Ok(Response::new(Empty::default()))
    }

    async fn get_test_case_feedbacks(
        &self,
        request: Request<Number>,
    ) -> Result<Response<Feedbacks>, Status> {
        let number = request.into_inner().num;
        let results = self.frontend.get_test_case_feedbacks(Some(number)).await;
        Ok(Response::new(msg_type::Feedbacks {
            feedbacks: results.into_iter().map(msg_type::Feedback::from).collect(),
        }))
    }

    async fn get_mutation_feedbacks(
        &self,
        request: Request<Number>,
    ) -> Result<Response<Feedbacks>, Status> {
        let number = request.into_inner().num;
        let results = self.frontend.get_mutation_feedbacks(Some(number)).await;
        Ok(Response::new(msg_type::Feedbacks {
            feedbacks: results.into_iter().map(msg_type::Feedback::from).collect(),
        }))
    }

    async fn get_interesting_test_cases(
        &self,
        request: Request<Number>,
    ) -> Result<Response<TestCases>, Status> {
        let number = request.into_inner().num;
        let results = self.frontend.get_results(Some(number)).await;
        Ok(Response::new(msg_type::TestCases {
            test_cases: results.into_iter().map(msg_type::TestCase::from).collect(),
        }))
    }
}

impl MyFeedbackCollector {
    pub async fn create_service(
        worker_num: u32,
        bitmap_size: usize,
    ) -> FeedbackCollectorServer<MyFeedbackCollector> {
        FeedbackCollectorServer::new(MyFeedbackCollector::new(worker_num, bitmap_size).await)
    }

    pub async fn run_service(
        server_addr: Option<String>,
        worker_num: u32,
        map_size: usize,
    ) -> SocketAddr {
        crate::rpc::run_service_in_background(
            Self::create_service(worker_num, map_size).await,
            server_addr,
        )
        .await
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::datatype;
    use crate::datatype::ExecutionStatus;
    use crate::rpc::feedback::feedback_collector_client::FeedbackCollectorClient;
    use crate::rpc::msg_type::{Feedback, Feedbacks, Number, TestCase};

    #[tokio::test]
    async fn feedback_rpc_test() {
        let addr = MyFeedbackCollector::run_service(None, 2, 65536).await;

        let mut client = FeedbackCollectorClient::connect(format!("http://{}", addr))
            .await
            .unwrap();

        let test_case = TestCase::default();
        let mut feedbacks = Feedbacks::default();
        let feedback = datatype::Feedback::create_coverage_feedback(
            ExecutionStatus::Interesting,
            Some(test_case.clone().into()),
            Some(vec![datatype::NewBit::new(0, 1)]),
        )
        .set_mutation_info(datatype::MutationInfo::new(0, 0));

        feedbacks.feedbacks.push(Feedback::from(feedback.clone()));
        feedbacks.feedbacks.push(Feedback::from(feedback));
        let feedback = datatype::Feedback::create_coverage_feedback(
            ExecutionStatus::Interesting,
            Some(test_case.clone().into()),
            Some(vec![datatype::NewBit::new(1, 1)]),
        )
        .set_mutation_info(datatype::MutationInfo::new(0, 0));

        feedbacks.feedbacks.push(Feedback::from(feedback));
        let request = tonic::Request::new(feedbacks);
        let response = client.check_feedbacks(request).await;
        assert!(response.is_ok());
        println!("go check");

        let request = tonic::Request::new(Number { num: 2 });
        let response: TestCases = client
            .get_interesting_test_cases(request)
            .await
            .unwrap()
            .into_inner();

        println!("{:?}", response);
        let interesing_test_case_counter = response.test_cases.len();
        assert_eq!(interesing_test_case_counter, 2);
    }
}
