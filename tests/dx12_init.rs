#![cfg(all(windows, feature = "gpu"))]

use std::process::{Command, Stdio};
use std::time::{Duration, Instant};

#[test]
fn dx12_init_smoke_finishes() {
    let exe = env!("CARGO_BIN_EXE_dx12_smoke");

    let mut child = Command::new(exe)
        .stdout(Stdio::piped())
        .stderr(Stdio::piped())
        .spawn()
        .expect("failed to spawn dx12_smoke binary");

    let timeout = Duration::from_secs(20);
    let deadline = Instant::now() + timeout;

    loop {
        match child.try_wait() {
            Ok(Some(status)) => {
                let out = child
                    .wait_with_output()
                    .expect("failed to collect dx12_smoke output");

                let stdout = String::from_utf8_lossy(&out.stdout);
                let stderr = String::from_utf8_lossy(&out.stderr);

                assert!(
                    status.success(),
                    "dx12_smoke failed (status={status}).\n--- stdout ---\n{stdout}\n--- stderr ---\n{stderr}\n"
                );
                return;
            }
            Ok(None) => {
                if Instant::now() >= deadline {
                    let _ = child.kill();
                    let out = child
                        .wait_with_output()
                        .expect("failed to collect dx12_smoke output after kill");

                    let stdout = String::from_utf8_lossy(&out.stdout);
                    let stderr = String::from_utf8_lossy(&out.stderr);

                    panic!(
                        "dx12_smoke timed out after {timeout:?} (process killed).\n--- stdout ---\n{stdout}\n--- stderr ---\n{stderr}\n"
                    );
                }
                std::thread::sleep(Duration::from_millis(50));
            }
            Err(err) => panic!("failed to poll dx12_smoke process: {err}"),
        }
    }
}
