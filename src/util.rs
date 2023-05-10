use super::datatype::{self, TestCase};
use core_affinity::CoreId;
use lazy_static::lazy_static;
use std::fs;
use std::fs::File;
use std::io;
use std::io::prelude::*;
use std::path::PathBuf;
use std::sync::{Arc, Mutex};
use std::time::Instant;

#[cfg(target_os = "linux")]
use libc::{cpu_set_t, sched_setaffinity, CPU_SET};
#[cfg(target_os = "linux")]
use std::mem;

#[cfg(not(target_os = "linux"))]
pub fn bind_to_cpu(_pid: i32) -> bool {
    true
}

// TODO: Make this portable.
#[cfg(target_os = "linux")]
pub fn bind_to_cpu(pid: i32) -> bool {
    let mut free_cpus = FREE_CPUS.lock().unwrap();
    if free_cpus.is_empty() {
        println!("No free cpu left");
        return false;
    }
    let core_id = free_cpus.pop().unwrap().id;
    // Turn `core_id` into a `libc::cpu_set_t` with only
    // one core active.
    let mut set = unsafe { mem::zeroed::<cpu_set_t>() };

    unsafe { CPU_SET(core_id, &mut set) };

    // Set the current thread's core affinity.
    unsafe {
        sched_setaffinity(
            pid, // Defaults to current thread
            mem::size_of::<cpu_set_t>(),
            &set,
        );
    }
    println!("Bind to cpu {}", core_id);
    true
}

lazy_static! {
    pub static ref FREE_CPUS: Arc<Mutex<Vec<CoreId>>> =
        Arc::new(Mutex::new(core_affinity::get_core_ids().unwrap()));
    pub static ref AFFINITY_NUM: usize = FREE_CPUS.lock().unwrap().len();
}

pub fn pretty_format_num(num: u64) -> String {
    num.to_string()
    /*
    if num < 10_000 {
        num.to_string()
    } else if num < 10_000_000 {
        format!("{}K", num / 1000)
    } else {
        format!("{}M", num / 1000_0000)
    }
        */
}

pub fn get_test_bin_path() -> String {
    PathBuf::from(env!("CARGO_MANIFEST_DIR"))
        .to_str()
        .unwrap()
        .to_string()
        + "/test_bins/"
}

pub fn get_test_bin_by_name(bin: &str) -> Option<String> {
    let mut path_str = PathBuf::from(env!("CARGO_MANIFEST_DIR"))
        .to_str()
        .unwrap()
        .to_string()
        + "/test_bins/"
        + bin;

    if !cfg!(target_os = "linux") {
        path_str += "_osx";
    }

    if std::path::Path::new(&path_str).is_file() {
        Some(path_str)
    } else {
        None
    }
}

pub fn read_corpus(path_str: String) -> io::Result<Vec<TestCase>> {
    let mut result = Vec::new();
    let paths = fs::read_dir(path_str).unwrap();
    for path in paths {
        let path = path.unwrap().path();
        let path = path.to_str().unwrap();

        let mut f = File::open(path)?;
        let mut buffer = Vec::new();

        f.read_to_end(&mut buffer)?;

        result.push(datatype::TestCase::new(buffer, 0));
    }
    Ok(result)
}

pub fn get_tcpdump_corpus() -> Vec<TestCase> {
    let corpus_path = get_test_bin_path() + "/test_corpus";
    read_corpus(corpus_path).unwrap()
}

pub enum Component {
    QueueManager,
    Executor,
    FeedbackCollector,
}

pub struct FuzzTimer {
    queue_manager_time: u128,
    executor_time: u128,
    feedback_collector_time: u128,
    now: Instant,
}

impl std::default::Default for FuzzTimer {
    fn default() -> Self {
        FuzzTimer {
            queue_manager_time: 0,
            executor_time: 0,
            feedback_collector_time: 0,
            now: Instant::now(),
        }
    }
}

impl FuzzTimer {
    pub fn start(&mut self) {
        self.now = Instant::now();
    }

    pub fn add_time(&mut self, t: Component) {
        match t {
            Component::QueueManager => self.queue_manager_time += self.now.elapsed().as_millis(),
            Component::Executor => self.executor_time += self.now.elapsed().as_millis(),
            Component::FeedbackCollector => {
                self.feedback_collector_time += self.now.elapsed().as_millis()
            }
        }
    }

    pub fn show(&self) {
        let queue_manager_time = self.queue_manager_time as f64;
        let executor_time = self.executor_time as f64;
        let feedback_time = self.feedback_collector_time as f64;
        let total = (queue_manager_time + executor_time + feedback_time).max(1.0);
        println!(
            "Total {:.2}, executor: {:.2}%, feedback: {:.2}%, queue manager: {:.2}% ",
            total,
            executor_time / total * 100.0,
            feedback_time / total * 100.0,
            queue_manager_time / total * 100.0
        );
    }
}

pub fn take_n_elements_from_vector<T>(v: &mut Vec<T>, num: Option<usize>) -> Vec<T> {
    let max_result_len = v.len();
    let n = match num {
        Some(num) => num.min(max_result_len),
        _ => max_result_len,
    };

    v.split_off(max_result_len - n)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn read_all_corpus() {
        let corpus_path = get_test_bin_path() + "/test_corpus";
        let corpus = read_corpus(corpus_path);
        assert!(corpus.is_ok());
        let corpus = corpus.unwrap();
        for test_case in corpus {
            println!("{:?}", test_case);
        }
    }
}
