use crate::util::pretty_format_num;
use lazy_static::lazy_static;
use psutil::cpu::CpuPercentCollector;
use serde_json::Value;
use std::fs;
use std::net::SocketAddr;
use std::sync::atomic::{AtomicU64, Ordering};
use std::sync::{Arc, Mutex};
use std::time::SystemTime;

use super::output_writer::OutputWriter;

// All the high level information we want to know about the fuzzer.
// TODO: This should be merged with the monitor.
#[allow(dead_code)]
pub struct FuzzerInfo {
    // Dynamic info
    start_time: SystemTime,
    last_print_time: AtomicU64,
    last_num_exec: AtomicU64,
    num_exec: AtomicU64,
    cur_exec_speed: AtomicU64,
    num_timeout_exec: AtomicU64,
    num_crashes: AtomicU64,
    cycle_done: AtomicU64,
    cur_coverage: AtomicU64,

    // CPU usage util past print
    cpu_usage: AtomicU64,
    all_cpu_usage: AtomicU64,
    all_cpu_record_times: AtomicU64,
    // Configuration
    total_coverage: u64,
    max_exec: u64,
    max_fuzzing_time: u64,
    cmd: String,
    num_executor: u64,
    num_mutator: u64,
    num_feedback_collector: u64,
    num_queue_manager: u64,
    num_core: u64,
    monitor_addr: SocketAddr,
    output_writer: Option<OutputWriter>,
}

lazy_static! {
    static ref FUZZER_INFO: Arc<Mutex<FuzzerInfo>> = Arc::new(Mutex::new(FuzzerInfo::new()));
    static ref CPU_PERCENT_COLLECTOR: Arc<Mutex<CpuPercentCollector>> =
        Arc::new(Mutex::new(CpuPercentCollector::new().unwrap()));
}

pub fn get_fuzzer_info() -> Arc<Mutex<FuzzerInfo>> {
    FUZZER_INFO.clone()
}

pub fn get_cpu_percentage_collector() -> Arc<Mutex<CpuPercentCollector>> {
    CPU_PERCENT_COLLECTOR.clone()
}

impl FuzzerInfo {
    pub fn new() -> Self {
        FuzzerInfo {
            start_time: SystemTime::now(),
            last_print_time: AtomicU64::new(0),
            last_num_exec: AtomicU64::new(0),
            num_exec: AtomicU64::new(0),
            cur_exec_speed: AtomicU64::new(0),
            num_timeout_exec: AtomicU64::new(0),
            num_crashes: AtomicU64::new(0),
            cycle_done: AtomicU64::new(0),
            total_coverage: 0,
            cur_coverage: AtomicU64::new(0),
            cmd: "".to_string(),
            num_executor: 0,
            num_mutator: 0,
            num_feedback_collector: 0,
            num_queue_manager: 0,
            num_core: 1,
            max_exec: u64::MAX,
            max_fuzzing_time: u64::MAX,
            monitor_addr: "127.0.0.1:12345".parse().unwrap(),
            output_writer: None,
            cpu_usage: AtomicU64::new(0),
            all_cpu_usage: AtomicU64::new(0),
            all_cpu_record_times: AtomicU64::new(0),
        }
    }

    pub fn reset_start_time(&mut self) {
        self.start_time = SystemTime::now();
    }

    fn simple_calculate(&self) {
        let elapsed_time = (self.start_time.elapsed().unwrap().as_millis() as u64).max(1);
        let last_print_time = self.last_print_time.swap(elapsed_time, Ordering::Relaxed);
        let millis = elapsed_time - last_print_time;

        let num_exec = self.num_exec.load(Ordering::Relaxed);
        let last_num_exec = self.last_num_exec.swap(num_exec, Ordering::Relaxed);

        self.cur_exec_speed.store(
            (num_exec - last_num_exec) * 1000 / millis.max(1),
            Ordering::Relaxed,
        );

        let cpu_percent: u64 = {
            (get_cpu_percentage_collector()
                .lock()
                .unwrap()
                .cpu_percent_percpu()
                .unwrap()
                .iter()
                .take(self.num_core as usize)
                .sum::<f32>()
                / self.num_core as f32) as u64
        };
        self.cpu_usage.store(cpu_percent, Ordering::Relaxed);
        self.all_cpu_usage.fetch_add(cpu_percent, Ordering::Relaxed);
        if cpu_percent != 0 {
            self.all_cpu_record_times.fetch_add(1, Ordering::Relaxed);
        }
    }

    pub fn simple_show(&self) {
        let elapsed_time = (self.start_time.elapsed().unwrap().as_millis() as u64).max(1);
        self.simple_calculate();
        let num_exec = self.num_exec.load(Ordering::Relaxed);
        let total_exec = pretty_format_num(num_exec);
        let cur_exec_speed = pretty_format_num(self.cur_exec_speed.load(Ordering::Relaxed));
        let average_speed = pretty_format_num(num_exec * 1000 / elapsed_time);
        let average_speed_per_core =
            pretty_format_num(num_exec * 1000 / elapsed_time.max(1) / self.num_core.max(1));
        let timeout_exec = self.num_timeout_exec.load(Ordering::Relaxed);
        let timeout_exec_str = pretty_format_num(timeout_exec);
        let current_coverage = pretty_format_num(self.cur_coverage.load(Ordering::Relaxed));
        let time = pretty_format_num(elapsed_time / 1000);
        let crash_num = pretty_format_num(self.num_crashes.load(Ordering::Relaxed));
        let timeout_rate = timeout_exec as f64 / num_exec.max(1) as f64 * 100f64;

        let cpu_percent = self.cpu_usage.load(Ordering::Relaxed);
        let all_cpu_usage = self.all_cpu_usage.load(Ordering::Relaxed);
        let all_cpu_usage_record_times = self.all_cpu_record_times.load(Ordering::Relaxed);
        let cpu_average = all_cpu_usage as f64 / all_cpu_usage_record_times.max(1) as f64;
        // Assume the cpu are occupied from lowest id to highest. Otherwise this number is wrong.
        println!(
            "Time: {time}, \
        Total exec: {total_exec}, current speed: {cur_exec_speed}/s, \
        average speed: {average_speed}/s, per core: {average_speed_per_core}/s, \
        timeout exec: {timeout_exec_str}, crash: {crash_num}, \
        Interesting inputs {current_coverage}, timeout rate: {timeout_rate:.3}%, cpu usage: {cpu_percent}%, average cpu usage {cpu_average:.1}%"
        );

        self.save_plot_data();
    }

    pub fn save_plot_data(&self) {
        if let Some(output_writer) = self.output_writer.as_ref() {
            output_writer.write_plot_data(self).unwrap();
        }
    }

    pub fn save_fuzzer_stat(&self) {
        if let Some(output_writer) = self.output_writer.as_ref() {
            output_writer.write_fuzzer_stat(self).unwrap();
        }
    }

    pub fn sync_with_json_config(&mut self, json_file_path: String) {
        let json_str = fs::read_to_string(json_file_path).unwrap();
        let config: Value = serde_json::from_str(&json_str).unwrap();

        if let Some(v) = config.get("monitor_addr") {
            self.set_monitor_addr(v.as_str().unwrap().parse().unwrap());
        }

        if let Some(v) = config.get("max_exec") {
            self.set_max_exec(v.as_u64().unwrap());
        }

        if let Some(v) = config.get("core") {
            self.set_core_num(v.as_u64().unwrap());
        }
    }

    fn set_monitor_addr(&mut self, addr: SocketAddr) {
        self.monitor_addr = addr;
    }

    pub fn sync_with_fuzzer_config(&mut self, fuzzer: &crate::FuzzerConfig) {
        self.set_cmd(fuzzer.cmd.clone())
            .set_total_coverage(fuzzer.map_size as u64)
            .set_core_num(fuzzer.core as u64)
            .set_max_exec(fuzzer.max_exec)
            .set_max_fuzzing_time(fuzzer.max_fuzzing_time)
            .set_queue_manager_num(fuzzer.queuemgr as u64)
            .set_executor_num(fuzzer.executor as u64)
            .set_feedback_collector_num(fuzzer.feedback as u64);
        if let Some(output_dir) = fuzzer.output.as_ref() {
            self.output_writer = Some(OutputWriter::new(output_dir.clone()).unwrap());
        }
    }

    pub fn get_output_dir(&self) -> Option<String> {
        if self.output_writer.is_some() {
            Some(self.output_writer.as_ref().unwrap().get_root_path())
        } else {
            None
        }
    }

    pub fn set_max_exec(&mut self, max_exec: u64) -> &mut Self {
        self.max_exec = max_exec;
        self
    }

    pub fn get_max_exec(&self) -> u64 {
        self.max_exec
    }
    pub fn set_executor_num(&mut self, executor_num: u64) -> &mut Self {
        self.num_executor = executor_num;
        self
    }

    pub fn get_executor_num(&self) -> u64 {
        self.num_executor
    }

    pub fn set_feedback_collector_num(&mut self, feedback_collector_num: u64) -> &mut Self {
        self.num_feedback_collector = feedback_collector_num;
        self
    }

    pub fn get_feebback_collector_num(&self) -> u64 {
        self.num_feedback_collector
    }

    pub fn set_queue_manager_num(&mut self, queue_manager_num: u64) -> &mut Self {
        self.num_queue_manager = queue_manager_num;
        self
    }

    pub fn get_queue_manager_num(&self) -> u64 {
        self.num_queue_manager
    }

    pub fn add_exec(&self, exec: u64) {
        self.num_exec.fetch_add(exec, Ordering::Relaxed);
    }

    pub fn get_exec(&self) -> u64 {
        self.num_exec.load(Ordering::Relaxed)
    }

    pub fn get_crash(&self) -> u64 {
        self.num_crashes.load(Ordering::Relaxed)
    }

    pub fn add_crash(&self, n_crash: u64) {
        self.num_crashes.fetch_add(n_crash, Ordering::Relaxed);
    }

    pub fn get_timeout_exec(&self) -> u64 {
        self.num_timeout_exec.load(Ordering::Relaxed)
    }

    pub fn add_timeout_exec(&self, n_timeout_exec: u64) {
        self.num_timeout_exec
            .fetch_add(n_timeout_exec, Ordering::Relaxed);
    }

    pub fn add_coverage(&self, coverage: u64) {
        self.cur_coverage.fetch_add(coverage, Ordering::Relaxed);
    }

    pub fn set_cmd(&mut self, cmd: String) -> &mut Self {
        self.cmd = cmd;
        self
    }

    pub fn set_core_num(&mut self, core_num: u64) -> &mut Self {
        self.num_core = core_num.min(*crate::util::AFFINITY_NUM as u64);
        self
    }

    pub fn get_core_num(&self) -> u64 {
        self.num_core
    }

    pub fn set_total_coverage(&mut self, total_coverage: u64) -> &mut Self {
        self.total_coverage = total_coverage;
        self
    }

    // Set max fuzzing time in seconds. The fuzzer will stop after fuzzing that period of time.
    pub fn set_max_fuzzing_time(&mut self, max_fuzzing_time: u64) -> &mut Self {
        self.max_fuzzing_time = max_fuzzing_time;
        self
    }

    pub fn get_max_fuzzing_time(&self) -> u64 {
        self.max_fuzzing_time
    }

    pub fn get_fuzzing_time(&self) -> u64 {
        self.start_time.elapsed().unwrap().as_secs()
    }

    // If the exec num or fuzzing time reach limit, then we should stop the fuzzer.
    pub fn stop_if_done(&self) {
        if self.get_exec() >= self.get_max_exec()
            || self.get_fuzzing_time() >= self.get_max_fuzzing_time()
        {
            self.save_fuzzer_stat();
            println!("Fuzzing done. Have a good day!");
            std::process::exit(0);
        }
    }
}

impl Default for FuzzerInfo {
    fn default() -> Self {
        Self::new()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::util;

    #[test]
    fn read_fuzzer_info_from_config() {
        let mut fuzzer_info = FuzzerInfo::new();
        fuzzer_info
            .sync_with_json_config(format!("{}/test_config.json", util::get_test_bin_path()));

        assert_eq!(fuzzer_info.get_max_exec(), 10);
        //assert_eq!(fuzzer_info.get_core_num(), 4);
    }
}
