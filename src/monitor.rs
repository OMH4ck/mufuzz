use lazy_static::lazy_static;
use stats::FuzzerInfo;
use std::collections::HashSet;
use std::net::SocketAddr;
use std::net::{IpAddr, Ipv4Addr};
use std::sync::{Arc, Mutex, RwLock};

use crate::datatype::TestCase;
use crate::executor::ExecutionStatus;

use self::output_writer::OutputWriter;

pub mod output_writer;
pub mod rpc;
pub mod stats;

#[derive(Clone, Copy)]
pub enum Service {
    Mutator(SocketAddr),
    FeedbackCollector(SocketAddr),
    Executor(SocketAddr),
    QueueManager(SocketAddr),
}

pub trait Monitor {
    fn get_services(&self) -> Vec<Service>;
    fn register_service(&self, service: Service) -> bool;
    // Print the fuzzer status.
    fn show_statistics(&self);
    fn reset_start_time(&mut self);

    // Receive statistics from other components. To make this general we use json type.
    fn receive_statistics(&self, v: serde_json::Value);
}

pub struct SimpleMonitor {
    services: Arc<Mutex<Vec<Service>>>,
    #[allow(dead_code)]
    addr: SocketAddr,
    service_addr_hashes: Arc<Mutex<HashSet<SocketAddr>>>,
    fuzzer_info: FuzzerInfo,
}

lazy_static! {
    pub static ref SIMPLE_MONITOR: Arc<RwLock<SimpleMonitor>> =
        Arc::new(RwLock::new(SimpleMonitor::default()));
}

static SIMPLE_MONITOR_PORT: u16 = 12345;
static WORKER_NUM: u32 = 1;

pub fn get_worker_num() -> u32 {
    WORKER_NUM
}

fn dump_json_testcases(
    output_writer: &OutputWriter,
    stats: &serde_json::Value,
    testcase_type: ExecutionStatus,
) {
    let test_case_vec = stats.get("testcases").unwrap().as_array().unwrap();
    for test_case_val in test_case_vec {
        let test_case_str = test_case_val.as_str().unwrap();
        let mut test_case: TestCase = serde_json::from_str(test_case_str).unwrap();
        match testcase_type {
            ExecutionStatus::Crash => {
                test_case.gen_id();
                match output_writer.save_crash(&test_case) {
                    Ok(_v) => {}
                    Err(_e) => println!("Cannot write crash file"),
                }
            }
            ExecutionStatus::Interesting => {
                output_writer.save_queue(&test_case).unwrap();
            }
            ExecutionStatus::Timeout => {
                test_case.gen_id();
                output_writer.save_hang(&test_case).unwrap();
            }
            _ => {}
        }
    }
}

impl SimpleMonitor {
    fn default() -> Self {
        SimpleMonitor {
            services: Arc::new(Mutex::new(Vec::default())),
            service_addr_hashes: Arc::new(Mutex::new(HashSet::default())),
            addr: SocketAddr::new(IpAddr::V4(Ipv4Addr::new(127, 0, 0, 1)), SIMPLE_MONITOR_PORT),
            fuzzer_info: FuzzerInfo::default(),
        }
    }

    #[allow(dead_code)]
    fn get_monitor_addr() -> SocketAddr {
        SIMPLE_MONITOR.read().unwrap().addr
    }

    fn get_services() -> Vec<Service> {
        SIMPLE_MONITOR.read().unwrap().get_services()
    }

    fn register_service(service: Service) -> bool {
        SIMPLE_MONITOR.write().unwrap().register_service(service)
    }

    pub fn get_fuzzer_info(&self) -> &FuzzerInfo {
        &self.fuzzer_info
    }

    pub fn get_fuzzer_info_mut(&mut self) -> &mut FuzzerInfo {
        &mut self.fuzzer_info
    }
}

pub fn get_monitor() -> Arc<RwLock<SimpleMonitor>> {
    SIMPLE_MONITOR.clone()
}

impl Monitor for SimpleMonitor {
    fn reset_start_time(&mut self) {
        self.fuzzer_info.reset_start_time();
    }
    fn get_services(&self) -> Vec<Service> {
        let services = self.services.lock().unwrap();
        services.clone()
    }

    fn register_service(&self, service: Service) -> bool {
        let mut services = self.services.lock().unwrap();
        let mut addr_hash = self.service_addr_hashes.lock().unwrap();
        let addr = match service {
            Service::Mutator(addr) => addr,
            Service::FeedbackCollector(addr) => addr,
            Service::Executor(addr) => addr,
            Service::QueueManager(addr) => addr,
        };
        if addr_hash.contains(&addr) {
            return false;
        }
        addr_hash.insert(addr);
        services.push(service);
        true
    }

    // TODO: Receive crash/hang/interesting test cases for saving in disk.
    fn receive_statistics(&self, stats: serde_json::Value) {
        let mut testcase_type = ExecutionStatus::Ok;
        if let Some(v) = stats.get("exec") {
            self.fuzzer_info.add_exec(v.as_u64().unwrap());
        } else if let Some(v) = stats.get("crash") {
            self.fuzzer_info.add_crash(v.as_u64().unwrap());
            testcase_type = ExecutionStatus::Crash;
        } else if let Some(v) = stats.get("timeout") {
            self.fuzzer_info.add_timeout_exec(v.as_u64().unwrap());
        } else if let Some(v) = stats.get("interesting_test_case") {
            self.fuzzer_info.add_coverage(v.as_u64().unwrap());
        } else {
            unreachable!();
        }
        if self.fuzzer_info.get_output_writer().is_some() && stats.get("testcases").is_some() {
            dump_json_testcases(
                self.fuzzer_info.get_output_writer().unwrap(),
                &stats,
                testcase_type,
            )
        }
    }

    fn show_statistics(&self) {
        self.fuzzer_info.simple_show();
        self.fuzzer_info.stop_if_done();
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::str::FromStr;

    #[test]
    fn simple_monitor_return_its_addr() {
        let monitor_addr = SimpleMonitor::get_monitor_addr();
        assert_eq!(
            SocketAddr::from_str("127.0.0.1:12345").unwrap(),
            monitor_addr
        );
    }
}
