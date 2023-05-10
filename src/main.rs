use clap::Parser;
use mufuzz::monitor::Monitor;
use mufuzz::FuzzerConfig;
use std::error::Error;
use tokio::runtime;

// In the future we will let users provide a config file instead.
fn main() -> Result<(), Box<dyn Error>> {
    let fuzzer = FuzzerConfig::parse();

    if !fuzzer.config.is_empty() {
        let fuzzer_stat = mufuzz::monitor::stats::get_fuzzer_info();
        fuzzer_stat
            .lock()
            .unwrap()
            .sync_with_json_config(fuzzer.config.clone());
    }

    match fuzzer.mode {
        2 => {
            let rt = runtime::Builder::new_multi_thread()
                .worker_threads(fuzzer.core as usize * 6)
                .enable_all()
                .build()?;
            rt.block_on(mufuzz::run_fuzzer_rpc_async_mode(fuzzer));
        }
        3 => {
            let rt = runtime::Builder::new_multi_thread()
                .worker_threads(fuzzer.core as usize * 5)
                .enable_all()
                .build()?;
            rt.block_on(mufuzz::run_fuzzer_local_async_mode(fuzzer));
        }
        4 => {
            mufuzz::monitor::get_monitor()
                .write()
                .unwrap()
                .get_fuzzer_info_mut()
                .sync_with_fuzzer_config(&fuzzer);
            for i in 0..fuzzer.core {
                let config = fuzzer.clone();
                std::thread::spawn(move || mufuzz::run_single_fuzzer_mode(config, i == 0));
            }

            std::thread::spawn(|| {
                let monitor = mufuzz::monitor::get_monitor();
                loop {
                    std::thread::sleep(std::time::Duration::from_millis(2000));
                    monitor.read().unwrap().show_statistics();
                }
            })
            .join()
            .unwrap();
        }
        5 => {
            let rt = runtime::Builder::new_multi_thread()
                .worker_threads(fuzzer.core as usize * 6)
                .enable_all()
                .build()?;
            rt.block_on(mufuzz::run_fuzzer_local_async_lock_free_mode(fuzzer));
        }
        6 => {
            let rt = runtime::Builder::new_multi_thread()
                .worker_threads(fuzzer.core as usize * 6)
                .enable_all()
                .build()?;
            rt.block_on(mufuzz::run_fuzzer_local_work_stealing_mode(fuzzer));
        }
        _ => {
            unreachable!();
        }
    }
    Ok(())
}
