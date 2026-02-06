extern crate bindgen;

use sha2::{Digest, Sha256};
use std::collections::HashMap;
use std::env;
use std::fs;
use std::io::{self, Read, Write};
use std::path::{Path, PathBuf};
use std::thread;
use std::time::Duration;

const DEFAULT_XGBOOST_VERSION: &str = "3.1.1";
const MAX_HEADER_SIZE_BYTES: usize = 2 * 1024 * 1024;
const MAX_WHEEL_SIZE_BYTES: usize = 200 * 1024 * 1024;
const DOWNLOAD_CONNECT_TIMEOUT_SECS: u64 = 15;
const DOWNLOAD_READ_TIMEOUT_SECS: u64 = 120;
const DOWNLOAD_WRITE_TIMEOUT_SECS: u64 = 120;

fn get_xgboost_version() -> String {
    let raw_version =
        env::var("XGBOOST_VERSION").unwrap_or_else(|_| DEFAULT_XGBOOST_VERSION.to_string());
    let version = raw_version.trim().to_string();
    if let Err(message) = validate_version(&version) {
        panic!("Invalid XGBOOST_VERSION {}: {}", version, message);
    }
    version
}

fn validate_version(version: &str) -> Result<(), String> {
    let parts: Vec<&str> = version.split('.').collect();
    if parts.len() != 3 {
        return Err("expected semantic version MAJOR.MINOR.PATCH".to_string());
    }
    for part in parts {
        if part.is_empty() || !part.chars().all(|c| c.is_ascii_digit()) {
            return Err("version must be numeric in all components".to_string());
        }
    }
    Ok(())
}

// Known SHA256 checksums for header files by version
fn get_header_checksums() -> HashMap<&'static str, (&'static str, &'static str)> {
    let mut checksums = HashMap::new();
    // Format: version => (c_api.h SHA256, base.h SHA256)
    checksums.insert(
        "3.1.1",
        (
            "c0f0a98eb36fb5e451fdd3e9ead2d185f4c61be2a6997fc295e5d1a94f3096e2",
            "8d771fb20e03f3443e21cfdcd26ac5cd880be585b8817f2e0d146e7c5c7bb63a",
        ),
    );
    checksums.insert(
        "3.0.5",
        (
            "2ccec6e5301fa5a1324f60af48b9c6be5879e590ed583ec9d74297e6018860bc",
            "47f0148706907ccecb72b8484687524bc36d58b4c6fe5e7b81e59de157261ea7",
        ),
    );
    checksums.insert(
        "2.1.4",
        (
            "b804850ec6c7a00f8e36f139dfce7fe348fc9ad066ff4cb7ac44a4f5420ec1dd",
            "525c4a2ba2f6bd9b17a299978e16f91897d497d6ae0ae5df2335dd059f00d0ce",
        ),
    );
    checksums.insert(
        "1.7.6",
        (
            "145ed1df652937122b6f6bc31331051eabc02226a0b62349ea593cdbe841c20d",
            "b26e17eadbcc6350dc900b35d164eedc02b1cd2a64913c560d4d416c81a68935",
        ),
    );
    checksums.insert(
        "1.4.2",
        (
            "3f5de5d046a3c9576e0c560abe5fa1e889f72b4b18ff2bf73e5f98290d47d0dc",
            "e3abfcc730eee86acf44124d5496a2b41413f963c4bbf560513eeae0b7d12fb7",
        ),
    );
    checksums
}

fn compute_sha256(data: &[u8]) -> String {
    let mut hasher = Sha256::new();
    hasher.update(data);
    format!("{:x}", hasher.finalize())
}

fn normalize_sha256(value: &str) -> Result<String, String> {
    let trimmed = value.trim();
    if trimmed.len() != 64 || !trimmed.chars().all(|c| c.is_ascii_hexdigit()) {
        return Err("SHA256 must be 64 hex characters".to_string());
    }
    Ok(trimmed.to_lowercase())
}

fn build_http_agent() -> ureq::Agent {
    ureq::AgentBuilder::new()
        .timeout_connect(Duration::from_secs(DOWNLOAD_CONNECT_TIMEOUT_SECS))
        .timeout_read(Duration::from_secs(DOWNLOAD_READ_TIMEOUT_SECS))
        .timeout_write(Duration::from_secs(DOWNLOAD_WRITE_TIMEOUT_SECS))
        .build()
}

fn read_to_end_with_limit<R: Read>(
    reader: &mut R,
    max_size: usize,
) -> Result<Vec<u8>, Box<dyn std::error::Error>> {
    let mut buffer = Vec::new();
    let mut chunk = [0u8; 32 * 1024];

    loop {
        let read_bytes = reader.read(&mut chunk)?;
        if read_bytes == 0 {
            break;
        }

        if buffer.len() + read_bytes > max_size {
            return Err(format!("Download exceeded max size of {} bytes", max_size).into());
        }

        buffer.extend_from_slice(&chunk[..read_bytes]);
    }

    Ok(buffer)
}

fn get_default_wheel_sha256(version: &str, os: &str, arch: &str) -> Option<&'static str> {
    if version != DEFAULT_XGBOOST_VERSION {
        return None;
    }

    match (os, arch) {
        ("linux", "x86_64") => {
            Some("405e48a201495fe9474f7aa27419f937794726a1bc7d2c2f3208b351c816580a")
        }
        ("linux", "aarch64") => {
            Some("4347671aa8a495595f17135171aeae5f6d9ab4b4e7b02f191864cf2202e3c902")
        }
        ("darwin", "x86_64") => {
            Some("a51a2e488102a007b8c222d58bf855415002e8cdf06d104eea24b08dbf4eec4f")
        }
        ("darwin", "aarch64") => {
            Some("fac06c989f2cf11af7aa546b3bb78e7fa87595891e5dfde28edf3e7492e5440a")
        }
        ("windows", "x86_64") => {
            Some("2e1067489688ad99a410e8f2acdfe9d21a299c2f3b4b25dc8f094eae709c7447")
        }
        _ => None,
    }
}

fn resolve_wheel_sha256(
    version: &str,
    os: &str,
    arch: &str,
) -> Result<Option<String>, Box<dyn std::error::Error>> {
    if let Ok(value) = env::var("XGBOOST_WHEEL_SHA256") {
        let normalized = normalize_sha256(&value)
            .map_err(|message| format!("Invalid XGBOOST_WHEEL_SHA256: {}", message))?;
        return Ok(Some(normalized));
    }

    if let Some(default_hash) = get_default_wheel_sha256(version, os, arch) {
        return Ok(Some(default_hash.to_string()));
    }

    Ok(None)
}

fn get_wheel_download_url(wheel_filename: &str) -> String {
    if let Ok(url) = env::var("XGBOOST_WHEEL_URL") {
        let trimmed = url.trim();
        if !trimmed.is_empty() {
            if !trimmed.starts_with("https://") {
                println!(
                    "cargo:warning=XGBOOST_WHEEL_URL is not HTTPS; ensure the source is trusted"
                );
            }
            return trimmed.to_string();
        }
    }

    format!(
        "https://files.pythonhosted.org/packages/py3/x/xgboost/{}",
        wheel_filename
    )
}

fn verify_checksum(
    data: &[u8],
    expected: &str,
    filename: &str,
) -> Result<(), Box<dyn std::error::Error>> {
    let actual = compute_sha256(data);
    if actual != expected {
        return Err(format!(
            "SHA256 checksum mismatch for {}:\n  Expected: {}\n  Got:      {}",
            filename, expected, actual
        )
        .into());
    }
    println!("cargo:warning=✓ Verified SHA256 for {}", filename);
    Ok(())
}

fn parse_version(version: &str) -> (u32, u32, u32) {
    let parts: Vec<&str> = version.split('.').collect();
    let major = parts.first().and_then(|s| s.parse().ok()).unwrap_or(0);
    let minor = parts.get(1).and_then(|s| s.parse().ok()).unwrap_or(0);
    let patch = parts.get(2).and_then(|s| s.parse().ok()).unwrap_or(0);
    (major, minor, patch)
}

fn emit_version_cfg_flags(version: &str) {
    let (major, minor, _patch) = parse_version(version);

    // XGBoost 1.4.0+ has thread-safe predictions for tree models
    // See: https://github.com/dmlc/xgboost/issues/5339
    if major > 1 || (major == 1 && minor >= 4) {
        println!("cargo:rustc-cfg=xgboost_thread_safe");
        println!(
            "cargo:warning=XGBoost version {} supports thread-safe predictions",
            version
        );
    } else {
        println!(
            "cargo:warning=XGBoost version {} does NOT support thread-safe predictions",
            version
        );
    }
}

fn get_platform_info() -> (String, String) {
    let target = env::var("TARGET").unwrap();

    // Determine OS
    let os = if target.contains("apple-darwin") {
        "darwin"
    } else if target.contains("linux") {
        "linux"
    } else if target.contains("windows") {
        "windows"
    } else {
        panic!("Unsupported target: {}", target);
    };

    // Determine architecture
    let arch = if target.contains("x86_64") {
        "x86_64"
    } else if target.contains("aarch64") || target.contains("arm64") {
        "aarch64"
    } else if target.contains("i686") || target.contains("i586") {
        "i686"
    } else {
        panic!("Unsupported architecture for target: {}", target);
    };

    (os.to_string(), arch.to_string())
}

fn download_xgboost_headers(
    agent: &ureq::Agent,
    out_dir: &Path,
) -> Result<(), Box<dyn std::error::Error>> {
    let version = get_xgboost_version();
    let checksums = get_header_checksums();

    // Get expected checksums for this version
    let (c_api_expected, base_expected) = checksums.get(version.as_str()).ok_or_else(|| {
        format!(
            "No known SHA256 checksums for XGBoost version {}. \
             Please verify this version manually or add checksums to build.rs",
            version
        )
    })?;

    // Create the include/xgboost directory
    let include_dir = out_dir.join("include/xgboost");
    fs::create_dir_all(&include_dir)?;

    // Download and verify c_api.h
    let c_api_path = include_dir.join("c_api.h");
    download_and_verify_file(
        agent,
        &format!(
            "https://raw.githubusercontent.com/dmlc/xgboost/v{}/include/xgboost/c_api.h",
            version
        ),
        &c_api_path,
        c_api_expected,
        "c_api.h",
        MAX_HEADER_SIZE_BYTES,
    )?;

    // Download and verify base.h
    let base_path = include_dir.join("base.h");
    download_and_verify_file(
        agent,
        &format!(
            "https://raw.githubusercontent.com/dmlc/xgboost/v{}/include/xgboost/base.h",
            version
        ),
        &base_path,
        base_expected,
        "base.h",
        MAX_HEADER_SIZE_BYTES,
    )?;

    Ok(())
}

fn download_and_verify_file(
    agent: &ureq::Agent,
    url: &str,
    dest_path: &Path,
    expected_sha256: &str,
    filename: &str,
    max_size: usize,
) -> Result<(), Box<dyn std::error::Error>> {
    println!("cargo:warning=Downloading {} from: {}", filename, url);

    // Download into memory buffer
    let response = match agent.get(url).call() {
        Ok(response) => response,
        Err(ureq::Error::Status(code, _)) => {
            return Err(format!("Failed to download {}: HTTP {}", filename, code).into())
        }
        Err(e) => return Err(format!("Failed to download {}: {}", filename, e).into()),
    };
    let mut reader = response.into_reader();
    let buffer = read_to_end_with_limit(&mut reader, max_size)?;

    // Verify SHA256 checksum
    verify_checksum(&buffer, expected_sha256, filename)?;

    // Only write file after successful verification
    let mut file = fs::File::create(dest_path)?;
    file.write_all(&buffer)?;

    Ok(())
}

fn download_with_retry(
    agent: &ureq::Agent,
    url: &str,
    max_retries: u32,
    max_size: usize,
) -> Result<Vec<u8>, Box<dyn std::error::Error>> {
    let mut last_error = None;

    for attempt in 0..max_retries {
        if attempt > 0 {
            let backoff = Duration::from_millis(100 * 2_u64.pow(attempt));
            println!(
                "cargo:warning=Retry attempt {} after {:?}",
                attempt + 1,
                backoff
            );
            thread::sleep(backoff);
        }

        match agent.get(url).call() {
            Ok(response) => {
                let status = response.status();
                if !(200..300).contains(&status) {
                    last_error = Some(format!("HTTP {}", status));
                    continue;
                }

                let mut reader = response.into_reader();
                match read_to_end_with_limit(&mut reader, max_size) {
                    Ok(buffer) => return Ok(buffer),
                    Err(e) => {
                        last_error = Some(e.to_string());
                        continue;
                    }
                }
            }
            Err(ureq::Error::Status(code, _)) => {
                last_error = Some(format!("HTTP {}", code));
            }
            Err(e) => {
                last_error = Some(e.to_string());
            }
        }
    }

    Err(format!(
        "Failed to download after {} attempts. Last error: {}",
        max_retries,
        last_error.unwrap_or_else(|| "Unknown error".to_string())
    )
    .into())
}

fn download_and_extract_wheel(
    agent: &ureq::Agent,
    out_dir: &Path,
) -> Result<(), Box<dyn std::error::Error>> {
    let (os, arch) = get_platform_info();
    let version = get_xgboost_version();
    let (major, minor, _patch) = parse_version(&version);

    // Determine wheel filename based on platform and version
    // Different XGBoost versions use different manylinux tags
    let wheel_filename = match (os.as_str(), arch.as_str()) {
        ("linux", "x86_64") => {
            // Choose manylinux tag based on version
            let manylinux_tag = if major >= 3 {
                "manylinux_2_28"
            } else if major == 1 && minor == 4 {
                "manylinux2010"
            } else {
                "manylinux2014"
            };
            format!("xgboost-{}-py3-none-{}_x86_64.whl", version, manylinux_tag)
        }
        ("linux", "aarch64") => {
            let manylinux_tag = if major >= 3 {
                "manylinux_2_28"
            } else {
                "manylinux2014"
            };
            format!("xgboost-{}-py3-none-{}_aarch64.whl", version, manylinux_tag)
        }
        ("darwin", "x86_64") => {
            // macOS x86_64 wheel names changed between versions
            if major >= 3 {
                format!("xgboost-{}-py3-none-macosx_10_15_x86_64.whl", version)
            } else if major == 1 && minor == 4 {
                format!("xgboost-{}-py3-none-macosx_10_14_x86_64.macosx_10_15_x86_64.macosx_11_0_x86_64.whl", version)
            } else {
                // Versions 1.7.x and 2.x use multi-platform tag
                format!("xgboost-{}-py3-none-macosx_10_15_x86_64.macosx_11_0_x86_64.macosx_12_0_x86_64.whl", version)
            }
        }
        ("darwin", "aarch64") => {
            // macOS arm64 support started with version 1.5.0
            if major == 1 && minor < 5 {
                return Err(format!(
                    "XGBoost {} does not have macOS arm64 support. Minimum version for arm64 is 1.5.0",
                    version
                ).into());
            }
            format!("xgboost-{}-py3-none-macosx_12_0_arm64.whl", version)
        }
        ("windows", "x86_64") => format!("xgboost-{}-py3-none-win_amd64.whl", version),
        _ => return Err(format!("Unsupported platform: {}-{}", os, arch).into()),
    };

    let lib_filename = match os.as_str() {
        "windows" => "xgboost.dll",
        "darwin" => "libxgboost.dylib",
        _ => "libxgboost.so",
    };

    let expected_wheel_sha256 = resolve_wheel_sha256(&version, &os, &arch)?;
    if expected_wheel_sha256.is_none() {
        println!(
            "cargo:warning=No wheel SHA256 configured for XGBoost {} on {}-{}. Set XGBOOST_WHEEL_SHA256 to enable verification",
            version, os, arch
        );
    }

    // Setup paths
    let wheel_dir = out_dir.join("wheel");
    let lib_dir = out_dir.join("libs");
    fs::create_dir_all(&wheel_dir)?;
    fs::create_dir_all(&lib_dir)?;

    let wheel_path = wheel_dir.join(&wheel_filename);
    let lib_dest_path = lib_dir.join(lib_filename);

    // Check if library already exists
    if lib_dest_path.exists() {
        println!(
            "cargo:warning=Using cached XGBoost library at: {}",
            lib_dest_path.display()
        );
        if let Some(expected) = expected_wheel_sha256.as_deref() {
            if wheel_path.exists() {
                let wheel_buffer = fs::read(&wheel_path)?;
                if let Err(e) = verify_checksum(&wheel_buffer, expected, &wheel_filename) {
                    return Err(format!("{} (cached wheel at {})", e, wheel_path.display()).into());
                }
            } else {
                println!(
                    "cargo:warning=Cached library found but wheel cache missing; cannot verify wheel checksum"
                );
            }
        }
        return Ok(());
    }

    // Check if wheel is cached
    let wheel_cached = wheel_path.exists();
    let wheel_buffer = if wheel_cached {
        println!(
            "cargo:warning=Using cached wheel at: {}",
            wheel_path.display()
        );
        fs::read(&wheel_path)?
    } else {
        // Download wheel with retry
        let download_url = get_wheel_download_url(&wheel_filename);

        println!(
            "cargo:warning=Downloading XGBoost wheel from: {}",
            download_url
        );
        let buffer = download_with_retry(agent, &download_url, 3, MAX_WHEEL_SIZE_BYTES)?;
        buffer
    };

    if let Some(expected) = expected_wheel_sha256.as_deref() {
        if let Err(e) = verify_checksum(&wheel_buffer, expected, &wheel_filename) {
            if wheel_cached {
                return Err(format!("{} (cached wheel at {})", e, wheel_path.display()).into());
            }
            return Err(e);
        }
    }

    if !wheel_cached {
        // Write atomically (temp file + rename) after verification
        let temp_path = wheel_path.with_extension("tmp");
        {
            let mut temp_file = fs::File::create(&temp_path)?;
            temp_file.write_all(&wheel_buffer)?;
            temp_file.sync_all()?;
        }
        fs::rename(&temp_path, &wheel_path)?;

        println!("cargo:warning=✓ Downloaded and cached wheel");
    }

    // Extract library from wheel
    println!("cargo:warning=Extracting library from wheel");

    let cursor = io::Cursor::new(wheel_buffer);
    let mut archive = zip::ZipArchive::new(cursor)?;

    // Search for the library file in the wheel
    let mut found = false;
    let mut searched_paths = Vec::new();
    for i in 0..archive.len() {
        let mut file = archive.by_index(i)?;
        let file_path = file.name().to_string();

        // Collect paths for debugging
        if file_path.contains(".dll") || file_path.contains(".so") || file_path.contains(".dylib") {
            searched_paths.push(file_path.clone());
        }

        // Look for the library file (usually in xgboost/lib/)
        // Use contains check to handle path separators across platforms
        if file_path.ends_with(lib_filename)
            || file_path.ends_with(&format!("/{}", lib_filename))
            || file_path.ends_with(&format!("\\{}", lib_filename))
        {
            println!("cargo:warning=Found library at: {}", file_path);

            // Extract to temp file, then rename atomically
            let temp_dest_path = lib_dest_path.with_extension("tmp");
            {
                let mut dest = fs::File::create(&temp_dest_path)?;
                io::copy(&mut file, &mut dest)?;
                dest.sync_all()?;
            }
            fs::rename(&temp_dest_path, &lib_dest_path)?;

            found = true;
            break;
        }
    }

    if !found {
        let error_msg = if searched_paths.is_empty() {
            format!(
                "Library file {} not found in wheel. No library files found at all.",
                lib_filename
            )
        } else {
            format!(
                "Library file {} not found in wheel. Found these library files instead: {:?}",
                lib_filename, searched_paths
            )
        };
        return Err(error_msg.into());
    }

    println!(
        "cargo:warning=✓ Successfully extracted XGBoost library to: {}",
        lib_dir.display()
    );

    Ok(())
}

fn main() {
    // Tell cargo about custom cfg flags we emit
    println!("cargo:rustc-check-cfg=cfg(xgboost_thread_safe)");

    let out_dir = PathBuf::from(env::var("OUT_DIR").unwrap());
    let xgb_include_root = out_dir.join("include");

    // Get version and emit cfg flags for thread safety
    let version = get_xgboost_version();
    emit_version_cfg_flags(&version);

    let agent = build_http_agent();

    // Download the headers
    if let Err(e) = download_xgboost_headers(&agent, &out_dir) {
        eprintln!("Failed to download XGBoost headers: {}", e);
        panic!("Cannot proceed without headers");
    }

    // Download and extract the wheel
    if let Err(e) = download_and_extract_wheel(&agent, &out_dir) {
        eprintln!("Failed to download and extract wheel: {}", e);
        panic!("Cannot proceed without compiled library");
    }

    let bindings = bindgen::Builder::default()
        .header("wrapper.h")
        .clang_arg(format!("-I{}", xgb_include_root.display()))
        // Generate bindings for XGB and XGD functions (Booster and DMatrix)
        .allowlist_function("XGB.*")
        .allowlist_function("XGD.*")
        // Allowlist the main types we need
        .allowlist_type("BoosterHandle")
        .allowlist_type("DMatrixHandle")
        .allowlist_type("bst_ulong")
        .size_t_is_usize(true)
        // Disable doc comments to avoid doctest failures from C comments
        .generate_comments(false)
        .generate()
        .expect("Unable to generate bindings.");

    bindings
        .write_to_file(out_dir.join("bindings.rs"))
        .expect("Couldn't write bindings.");

    // Get platform info
    let (os, _arch) = get_platform_info();

    // Determine the library filename based on the OS
    let lib_filename = match os.as_str() {
        "windows" => "xgboost.dll",
        "darwin" => "libxgboost.dylib",
        _ => "libxgboost.so",
    };

    // Copy the library from OUT_DIR/libs to the final target directory
    let lib_source_path = out_dir.join("libs").join(lib_filename);

    // Find the final output directory (e.g., target/release)
    let target_dir = out_dir
        .ancestors()
        .find(|p| p.ends_with("target"))
        .unwrap()
        .join(env::var("PROFILE").unwrap());

    let lib_dest_path = target_dir.join(lib_filename);
    fs::copy(&lib_source_path, &lib_dest_path).expect("Failed to copy library to target directory");

    // On macOS/Linux, change the install name/soname to use @loader_path/$ORIGIN
    if os == "darwin" {
        if let Err(e) = run_install_name_tool(&lib_source_path, lib_filename) {
            panic!(
                "install_name_tool failed for {}: {}",
                lib_source_path.display(),
                e
            );
        }
        if let Err(e) = run_install_name_tool(&lib_dest_path, lib_filename) {
            panic!(
                "install_name_tool failed for {}: {}",
                lib_dest_path.display(),
                e
            );
        }
    } else if os == "linux" {
        // Use patchelf to set soname (if available)
        maybe_set_soname_with_patchelf(&lib_source_path, lib_filename);
        maybe_set_soname_with_patchelf(&lib_dest_path, lib_filename);
    }

    // Set the library search path for the build-time linker
    let lib_search_path = out_dir.join("libs");
    println!(
        "cargo:rustc-link-search=native={}",
        lib_search_path.display()
    );

    // Set the rpath for the run-time linker based on the OS
    match os.as_str() {
        "darwin" => {
            // For macOS, add multiple rpath entries for IDE compatibility
            println!("cargo:rustc-link-arg=-Wl,-rpath,@executable_path");
            println!("cargo:rustc-link-arg=-Wl,-rpath,@executable_path/../..");
            println!("cargo:rustc-link-arg=-Wl,-rpath,@loader_path");
            println!("cargo:rustc-link-arg=-Wl,-rpath,@loader_path/../..");
            println!(
                "cargo:rustc-link-arg=-Wl,-rpath,{}",
                lib_search_path.display()
            );
            // Add the target directory to rpath as well
            if let Some(target_root) = out_dir.ancestors().find(|p| p.ends_with("target")) {
                println!(
                    "cargo:rustc-link-arg=-Wl,-rpath,{}/debug",
                    target_root.display()
                );
                println!(
                    "cargo:rustc-link-arg=-Wl,-rpath,{}/release",
                    target_root.display()
                );
            }
        }
        "linux" => {
            // For Linux, use $ORIGIN
            println!("cargo:rustc-link-arg=-Wl,-rpath,$ORIGIN");
            println!("cargo:rustc-link-arg=-Wl,-rpath,$ORIGIN/../..");
            println!(
                "cargo:rustc-link-arg=-Wl,-rpath,{}",
                lib_search_path.display()
            );
            // Add the target directory to rpath as well
            if let Some(target_root) = out_dir.ancestors().find(|p| p.ends_with("target")) {
                println!(
                    "cargo:rustc-link-arg=-Wl,-rpath,{}/debug",
                    target_root.display()
                );
                println!(
                    "cargo:rustc-link-arg=-Wl,-rpath,{}/release",
                    target_root.display()
                );
            }
        }
        _ => {} // No rpath needed for Windows
    }

    println!("cargo:rustc-link-lib=dylib=xgboost");
}

fn run_install_name_tool(
    lib_path: &Path,
    lib_filename: &str,
) -> Result<(), Box<dyn std::error::Error>> {
    use std::process::Command;
    let status = Command::new("install_name_tool")
        .arg("-id")
        .arg(format!("@loader_path/{}", lib_filename))
        .arg(lib_path)
        .status()
        .map_err(|e| format!("Failed to run install_name_tool: {}", e))?;

    if !status.success() {
        return Err("install_name_tool returned a non-zero status".into());
    }

    Ok(())
}

fn maybe_set_soname_with_patchelf(lib_path: &Path, lib_filename: &str) {
    use std::process::Command;
    match Command::new("patchelf")
        .arg("--set-soname")
        .arg(lib_filename)
        .arg(lib_path)
        .status()
    {
        Ok(status) => {
            if !status.success() {
                println!(
                    "cargo:warning=patchelf returned a non-zero status for {}",
                    lib_path.display()
                );
            }
        }
        Err(e) => {
            println!(
                "cargo:warning=patchelf not available; skipping SONAME update for {} ({})",
                lib_path.display(),
                e
            );
        }
    }
}
