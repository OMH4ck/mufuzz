# mufuzz, a parallel fuzzing framework

TODO: Add reference

## Build
1. Install `cargo` and `protoc`
```
curl --proto '=https' --tlsv1.2 -sSf https://sh.rustup.rs | sh -s -- -y
wget https://github.com/protocolbuffers/protobuf/releases/download/v21.5/protoc-21.5-linux-x86_64.zip && \
  sudo unzip protoc-21.5-linux-x86_64.zip -d /usr/local
```
2. `cargo build --release`


## Run
Basic run (for example, run on 20 cores):
```
taskset -c 0-19 cargo run --release --  -c "your/command @@" -i input_corpus --core 20
```

Check the usage:
```
cargo run --release -- -help
```

## Note
1. The code is still under cleanup. Some features are still missing (such as saving the corpus/crashes to disk).
2. Assign exclusive cores to `mufuzz` for better performance.
3. Part of the forkserver code is borrowed from [LibAFL](https://github.com/AFLplusplus/LibAFL), a great project for building fuzzers.
4. After you exit the fuzzer, you might need to run `ipcrm -a` to remove the share memory.
