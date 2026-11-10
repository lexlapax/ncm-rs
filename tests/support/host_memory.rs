//! Whole-process host memory observations, outside timed workloads.
use serde_json::{Value, json};
type Result<T> = std::result::Result<T, Box<dyn std::error::Error>>;
#[cfg(target_os = "linux")]
pub fn host() -> Result<Value> {
    let status = std::fs::read_to_string("/proc/self/status")?;
    let mut result = std::collections::BTreeMap::new();
    for key in ["VmRSS:", "VmHWM:"] {
        let line = status
            .lines()
            .find(|l| l.starts_with(key))
            .ok_or("missing memory counter")?;
        result.insert(
            key.trim_end_matches(':'),
            line.split_whitespace()
                .nth(1)
                .ok_or("missing memory value")?
                .parse::<u64>()?
                * 1024,
        );
    }
    Ok(json!(result))
}
#[cfg(target_os = "macos")]
#[allow(unsafe_code)]
pub fn host() -> Result<Value> {
    // Safety: rusage is a C struct of integer/timeval fields, valid zeroed. The
    // writable pointer lives through getrusage; inspect it only after success.
    // Darwin ru_maxrss is bytes (Linux's corresponding field would be KiB).
    let mut usage: libc::rusage = unsafe { std::mem::zeroed() };
    if unsafe { libc::getrusage(libc::RUSAGE_SELF, &mut usage) } != 0 {
        return Err(std::io::Error::last_os_error().into());
    }
    let output = std::process::Command::new("/bin/ps")
        .args(["-o", "rss=", "-p", &std::process::id().to_string()])
        .output()?;
    if !output.status.success() {
        return Err("ps RSS query failed".into());
    }
    let rss = std::str::from_utf8(&output.stdout)?.trim().parse::<u64>()? * 1024;
    Ok(json!({"VmRSS":rss,"VmHWM":u64::try_from(usage.ru_maxrss)?}))
}
#[cfg(not(any(target_os = "linux", target_os = "macos")))]
pub fn host() -> Result<Value> {
    Err("host memory probe supports Linux/macOS only".into())
}
