use crate::datatype;
use crate::monitor::Service;
use msg_type::ServiceType;
use serde_json::Value;
use std::net::SocketAddr;
use std::str::FromStr;

pub mod msg_type {
    #![allow(clippy::derive_partial_eq_without_eq)]
    tonic::include_proto!("microfuzz.msg_type");
}

pub mod execution {
    #![allow(clippy::derive_partial_eq_without_eq)]
    tonic::include_proto!("microfuzz.execution");
}

pub mod feedback {
    #![allow(clippy::derive_partial_eq_without_eq)]
    tonic::include_proto!("microfuzz.feedback");
}

pub mod mutation {
    #![allow(clippy::derive_partial_eq_without_eq)]
    tonic::include_proto!("microfuzz.mutation");
}

pub mod queue_management {
    #![allow(clippy::derive_partial_eq_without_eq)]
    tonic::include_proto!("microfuzz.queue_management");
}

pub mod monitor {
    #![allow(clippy::derive_partial_eq_without_eq)]
    tonic::include_proto!("microfuzz.monitor");
}

impl msg_type::MonitorData {
    pub fn from(data: Vec<Value>) -> Self {
        Self {
            data: data.iter().map(serde_json::Value::to_string).collect(),
        }
    }

    pub fn into(self) -> Vec<Value> {
        self.data
            .into_iter()
            .filter_map(|s| {
                if let Ok(s) = serde_json::from_str(s.as_str()) {
                    Some(s)
                } else {
                    None
                }
            })
            .collect::<Vec<Value>>()
    }
}

impl msg_type::ServiceInfo {
    pub fn from(service: Service) -> Self {
        match service {
            Service::FeedbackCollector(addr) => Self {
                service_type: ServiceType::Feedbackcollector as i32,
                socket_addr: addr.to_string(),
            },
            Service::QueueManager(addr) => Self {
                service_type: ServiceType::Queuemanager as i32,
                socket_addr: addr.to_string(),
            },
            Service::Mutator(addr) => Self {
                service_type: ServiceType::Mutator as i32,
                socket_addr: addr.to_string(),
            },
            Service::Executor(addr) => Self {
                service_type: ServiceType::Executor as i32,
                socket_addr: addr.to_string(),
            },
        }
    }

    pub fn into(self) -> Option<Service> {
        let addr = SocketAddr::from_str(&self.socket_addr).unwrap();
        let service = match ServiceType::from_i32(self.service_type) {
            Some(ServiceType::Executor) => Service::Executor(addr),
            Some(ServiceType::Mutator) => Service::Mutator(addr),
            Some(ServiceType::Feedbackcollector) => Service::FeedbackCollector(addr),
            Some(ServiceType::Queuemanager) => Service::QueueManager(addr),
            None => return None,
        };
        Some(service)
    }
}

impl msg_type::TestCase {
    pub fn from(test_case: datatype::TestCase) -> Self {
        Self {
            pid: test_case.get_pid(),
            content: test_case.take_buffer(),
        }
    }

    pub fn into(self) -> datatype::TestCase {
        crate::datatype::TestCase::new(self.content, self.pid)
    }
}

impl msg_type::MutationInfo {
    pub fn from(mutation_info: datatype::MutationInfo) -> Self {
        Self {
            pid: mutation_info.get_pid(),
            mutator_id: mutation_info.get_mutation_id(),
        }
    }

    pub fn into(self) -> datatype::MutationInfo {
        crate::datatype::MutationInfo::new(self.pid, self.mutator_id)
    }
}

impl msg_type::ExecutionStatus {
    pub fn from(status: datatype::ExecutionStatus) -> Self {
        match status {
            datatype::ExecutionStatus::Ok => msg_type::ExecutionStatus::Ok,
            datatype::ExecutionStatus::Timeout => msg_type::ExecutionStatus::Timeout,
            datatype::ExecutionStatus::Crash => msg_type::ExecutionStatus::Crash,
            datatype::ExecutionStatus::Interesting => msg_type::ExecutionStatus::Interesting,
        }
    }

    pub fn into(self) -> datatype::ExecutionStatus {
        match self {
            msg_type::ExecutionStatus::Ok => datatype::ExecutionStatus::Ok,
            msg_type::ExecutionStatus::Timeout => datatype::ExecutionStatus::Timeout,
            msg_type::ExecutionStatus::Crash => datatype::ExecutionStatus::Crash,
            msg_type::ExecutionStatus::Interesting => datatype::ExecutionStatus::Interesting,
        }
    }
}

impl msg_type::NewBit {
    pub fn from(new_bit: datatype::NewBit) -> Self {
        Self {
            index: new_bit.index as u32,
            value: new_bit.val as u32,
        }
    }

    pub fn into(self) -> datatype::NewBit {
        datatype::NewBit::new(self.index as usize, self.value as u8)
    }
}

impl msg_type::feedback_data::FeedbackData {
    pub fn from(data: datatype::FeedbackData) -> Self {
        match data {
            datatype::FeedbackData::NewCoverage(coverage) => {
                msg_type::feedback_data::FeedbackData::NewCoverage(msg_type::NewCoverage {
                    new_bits: coverage.into_iter().map(msg_type::NewBit::from).collect(),
                })
            }
            datatype::FeedbackData::Counter(counter) => {
                msg_type::feedback_data::FeedbackData::Counter(msg_type::Counter { counter })
            }
            datatype::FeedbackData::Unknown => msg_type::feedback_data::FeedbackData::FakeFeedback(
                msg_type::FakeFeedback::default(),
            ),
        }
    }

    pub fn into(self) -> datatype::FeedbackData {
        match self {
            msg_type::feedback_data::FeedbackData::NewCoverage(coverage) => {
                datatype::FeedbackData::NewCoverage(
                    coverage
                        .new_bits
                        .into_iter()
                        .map(|new_bit| new_bit.into())
                        .collect(),
                )
            }
            msg_type::feedback_data::FeedbackData::Counter(counter) => {
                datatype::FeedbackData::Counter(counter.counter)
            }
            msg_type::feedback_data::FeedbackData::FakeFeedback(_) => {
                datatype::FeedbackData::Unknown
            }
        }
    }
}

impl msg_type::Feedback {
    pub fn from(mut feedback: crate::datatype::Feedback) -> Self {
        let test_case = feedback.take_test_case().map(msg_type::TestCase::from);
        let mutation_info = feedback
            .take_mutation_info()
            .map(msg_type::MutationInfo::from);
        let status = msg_type::ExecutionStatus::from(feedback.get_status()) as i32;

        let data = feedback
            .take_data()
            .map(msg_type::feedback_data::FeedbackData::from);
        Self {
            test_case,
            status,
            mutation_info,
            data: Some(msg_type::FeedbackData {
                feedback_data: data,
            }),
        }
    }

    pub fn into(self) -> datatype::Feedback {
        let status = msg_type::ExecutionStatus::from_i32(self.status)
            .unwrap()
            .into();
        let mut result = datatype::Feedback::new(status);

        if let Some(mutation_info) = self.mutation_info {
            result = result.set_mutation_info(mutation_info.into());
        }

        if let Some(test_case) = self.test_case.map(msg_type::TestCase::into) {
            result = result.set_test_case(test_case);
        }

        if let Some(data) = self.data {
            if let Some(feedback_data) = data.feedback_data {
                result = result.set_data(feedback_data.into());
            }
        }

        result
    }
}

pub async fn run_service_in_background<S>(svc: S, addr: Option<String>) -> SocketAddr
where
    S: tower::Service<
            http::Request<tonic::transport::Body>,
            Response = http::Response<tonic::body::BoxBody>,
            Error = std::convert::Infallible,
        > + tonic::transport::NamedService
        + Clone
        + Send
        + 'static,
    S::Future: Send + 'static,
    S::Error: Into<Box<dyn std::error::Error + Send + Sync>> + Send,
{
    let mut server_addr_str = String::from("127.0.0.1:0");
    if let Some(server_addr) = addr {
        server_addr_str = server_addr;
    }
    let listener = tokio::net::TcpListener::bind(server_addr_str)
        .await
        .unwrap();
    let addr = listener.local_addr().unwrap();
    tokio::spawn(async move {
        tonic::transport::Server::builder()
            .add_service(svc)
            .serve_with_incoming(tokio_stream::wrappers::TcpListenerStream::new(listener))
            .await
            .unwrap();
    });
    addr
}

#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn test_case_conversion_keep_all_info() {
        let test_case_inner = crate::datatype::TestCase::new(vec![1, 2, 3], 123);
        let test_case_rpc = msg_type::TestCase::from(test_case_inner.clone());
        assert_eq!(test_case_inner.get_pid(), test_case_rpc.pid);
        assert_eq!(test_case_inner.get_buffer().clone(), test_case_rpc.content);

        let test_case_rpc_inner = test_case_rpc.into();
        assert_eq!(test_case_inner.get_pid(), test_case_rpc_inner.get_pid());
        assert_eq!(
            test_case_inner.get_buffer(),
            test_case_rpc_inner.get_buffer()
        );
    }

    #[test]
    fn feedback_conversion_keep_all_info() {
        // TODO
    }
}
