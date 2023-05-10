use crate::rpc::monitor::monitor_server::{Monitor, MonitorServer};
use crate::rpc::msg_type::{Empty, RegistrationStatus, ServiceInfo, ServiceList};
use std::net::SocketAddr;
use tonic::{Request, Response, Status};

#[derive(Debug, Default)]
pub struct MySimpleMonitor {}

#[tonic::async_trait]
impl Monitor for MySimpleMonitor {
    async fn register_service(
        &self,
        request: Request<ServiceInfo>,
    ) -> Result<Response<RegistrationStatus>, Status> {
        println!("Got a request: {:?}", request);
        let service_info = request.into_inner();
        let success = match service_info.into() {
            Some(service_info) => crate::monitor::SimpleMonitor::register_service(service_info),
            None => false,
        };
        Ok(Response::new(RegistrationStatus { success }))
    }

    async fn get_services(&self, request: Request<Empty>) -> Result<Response<ServiceList>, Status> {
        println!("Got a request: {:?}", request);

        let service_list = crate::monitor::SimpleMonitor::get_services();
        Ok(Response::new(ServiceList {
            services: service_list.into_iter().map(ServiceInfo::from).collect(),
        }))
    }
}

impl MySimpleMonitor {
    pub fn create_service() -> MonitorServer<MySimpleMonitor> {
        MonitorServer::new(MySimpleMonitor::default())
    }

    pub async fn run_service(server_addr: Option<String>) -> SocketAddr {
        crate::rpc::run_service_in_background(Self::create_service(), server_addr).await
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::rpc::monitor::monitor_client::MonitorClient;
    use crate::rpc::msg_type::{Empty, ServiceInfo, ServiceType};

    #[tokio::test]
    async fn monitor_rpc_test() {
        let addr = MySimpleMonitor::run_service(None).await;

        let mut client = MonitorClient::connect(format!("http://{}", addr))
            .await
            .unwrap();

        let request = tonic::Request::new(ServiceInfo {
            service_type: ServiceType::Feedbackcollector as i32,
            socket_addr: "127.0.0.1:22233".to_string(),
        });
        let response = client.register_service(request).await.unwrap().into_inner();
        assert!(response.success);

        let request = tonic::Request::new(Empty::default());
        let response = client.get_services(request).await.unwrap().into_inner();
        println!("{:#?}", response);
        assert_eq!(response.services.len(), 1);

        let request = tonic::Request::new(ServiceInfo {
            service_type: ServiceType::Mutator as i32,
            socket_addr: "127.0.0.1:22234".to_string(),
        });
        let response = client.register_service(request).await.unwrap().into_inner();
        assert!(response.success);

        let request = tonic::Request::new(ServiceInfo {
            service_type: ServiceType::Queuemanager as i32,
            socket_addr: "127.0.0.1:22235".to_string(),
        });
        let response = client.register_service(request).await.unwrap().into_inner();
        assert!(response.success);

        let request = tonic::Request::new(ServiceInfo {
            service_type: ServiceType::Executor as i32,
            socket_addr: "127.0.0.1:22236".to_string(),
        });
        let response = client.register_service(request).await.unwrap().into_inner();
        assert!(response.success);

        // A service with duplicated socket address should not be registered.
        let request = tonic::Request::new(ServiceInfo {
            service_type: ServiceType::Executor as i32,
            socket_addr: "127.0.0.1:22236".to_string(),
        });
        let response = client.register_service(request).await.unwrap().into_inner();
        assert!(!response.success);

        let request = tonic::Request::new(Empty::default());
        let response = client.get_services(request).await.unwrap().into_inner();
        println!("{:#?}", response);
        assert_eq!(response.services.len(), 4);
    }
}
